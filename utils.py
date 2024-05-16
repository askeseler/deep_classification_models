import numpy as np
import torch
import matplotlib.pyplot as plt 
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm
import importlib
import torch.nn.functional as F

def precision(tp, fp):
    return np.nan_to_num(tp / (tp + fp))

def recall(tp, fn):
    return np.nan_to_num(tp / (tp + fn))

def f1(tp, fp, fn, epsilon = 1e-25):
    """ tp, fp, fn in percent """
    return np.nan_to_num((2 * (precision(tp, fp) * recall(tp, fn))) / (precision(tp, fp) + recall(tp, fn)))

def evaluate(predicted, groundtruth):
    acc = np.mean(groundtruth == predicted)

    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    tp = np.mean(np.logical_and(predicted >= 1, groundtruth >= 1).astype(float))

    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    tn = np.mean(np.logical_and(predicted <= 0, groundtruth <= 0).astype(float))

    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    fp = np.mean(np.logical_and(predicted >= 1, groundtruth <= 0).astype(float))

    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    fn = np.mean(np.logical_and(predicted <= 0, groundtruth >= 1).astype(float))
    
    f1_ = f1(tp, fp, fn)
    recall_ = recall(tp, fn)
    precision_ = precision(tp, fp)
    
    return np.array([acc, fp, fn, tp, tn, precision_, recall_, f1_]).round(2)

def auroc(model, loader_name='val', N_classes=4, device = "cpu", class_names = [], figsize = None):
    """ Modified from source: https://gist.github.com/khizirsiddiqui"""
    model.eval()
    y_test = []
    y_score = []
    dataloaders =  {"val" : model.val_dataloader(), "train": model.train_dataloader(), "test":model.test_dataloader()}
    dataloader = dataloaders[loader_name]
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(tqdm(dataloader)):
            inputs = inputs.to(device)
            y_test.append(F.one_hot(classes, N_classes).numpy())
            
            try:
                bs, ncrops, c, h, w = inputs.size()
            except:
                bs, c, h, w = inputs.size()
                ncrops = 1
            if ncrops > 1:
                outputs = model.forward(inputs.view(-1, c, h, w))
                outputs = outputs.view(bs, ncrops, -1).mean(1)
            else:
                outputs = model(inputs)
            y_score.append(outputs.cpu().numpy())
    y_test = np.array([t.ravel() for t in y_test])
    y_score = np.array([t.ravel() for t in y_score])
    # print(y_true)
    # print(y_pred)

    """
    compute ROC curve and ROC area for each class in each fold
    """

    fpr = dict()
    tpr = dict()
    local_roc_auc = dict()
    for i in range(N_classes):
        fpr[i], tpr[i], _ = roc_curve(np.array(y_test[:, i]),np.array(y_score[:, i]))
        local_roc_auc[i] = auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    local_roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(N_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(N_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= N_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    local_roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    if figsize:
        plt.figure(dpi=300, figsize = figsize)
    else:
        plt.figure(dpi=300)

    plt.plot(fpr["micro"], tpr["micro"],
             label='$\overline{{{{ROC}}}}_{{micro}}={0:0.2f}$'
                   ''.format(local_roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='$\overline{{{{ROC}}}}_{{macro}}={0:0.2f}$'
                   ''.format(local_roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = [['black',u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'] for _ in range(100)]
    colors = [item for sublist in colors for item in sublist]

    for i, color in zip(range(N_classes), colors):
        if len(class_names) == 0:
            i1 = i
        else:
            try:
                i1 = class_names[i]
            except:
                print(len(class_names))
                i1 = i
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='$ROC_{{{0}}}={1:0.2f}$'
                       ''.format(i1, local_roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-1e-2, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristics')
    plt.legend(loc="lower right")
    plt.show()


def forward_pl_model(model, loader, seed = 42, verbose = 1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = model.to(device).eval()
    if verbose == 1:
        enum = tqdm
    else:
        enum = lambda x: x
    with torch.no_grad():
        groundtruth = None
        predictions = None
        for x, y in enum(loader):
            res = model(x.to(device))
            if type(predictions) == type(None):
                predictions = res
            else:
                predictions = torch.vstack([predictions, res])
            if type(groundtruth) == type(None):
                groundtruth = y
            else:
                groundtruth = torch.vstack([groundtruth, y])

    return groundtruth.to(device), predictions.to(device)