from multiline_progressbar import *
import os
import pandas as pd
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import argparse
from platform import architecture
import shutil
from models import *
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_python_file(filepath):
    args = importlib.import_module(
        filepath.replace("/", ".").replace(".py", ""))
    return args


def prepare_outputfolder(output_foldername, settings_file, dir="logs"):
    os.makedirs(dir, exist_ok=True)
    logs = np.array(os.listdir(dir), dtype=object)
    where = np.where([f.startswith(output_foldername) for f in logs])[0]
    print(where)
    if len(where) > 0:
        idxs = [int(f.split("_")[-1]) for f in logs[where]]
        new_idx = np.max(idxs) + 1
    else:
        new_idx = 0
    output_foldername = output_foldername + "_" + str(new_idx)
    os.makedirs(os.path.join(dir, output_foldername))
    shutil.copy(settings_file, os.path.join(
        dir, output_foldername, os.path.basename(settings_file)))
    return os.path.join(dir, output_foldername)


def train_categorization_model(opt, settings_file):
    # Training & Output
    architectures = opt.architectures
    prefix = opt.prefix
    learning_rate = opt.learning_rate
    size = opt.size
    num_epochs = opt.num_epochs
    num_classes = opt.num_classes
    transfer = opt.transfer
    batch_size = opt.batch_size
    optimizer = opt.optimizer
    gpus = opt.gpus

    train_set = opt.train_set
    vld_set = opt.vld_set
    test_set = opt.test_set
    tune_fc_only = opt.tune_fc_only

    num_epochs = opt.num_epochs
    gpus = 1

    for architecture in architectures:
        output_foldername = opt.output_foldername + "_" + architecture
        output_foldername = prepare_outputfolder(
            output_foldername, settings_file)

        callbacks = [callback_best_model]
        logger = TensorBoardLogger(save_dir=os.getcwd(), name=output_foldername.split(
            "/")[0], version=output_foldername.split("/")[1])
        callback_best_model = ModelCheckpoint(dirpath=output_foldername, monitor='acc_val', save_top_k=1,
                                              filename=prefix + architecture +
                                              '_{epoch}-{val_loss:.2f}-{acc_val:.2f}',
                                              save_last=True, mode="max")

        callbacks = [callback_best_model]

        # Instantiate Model
        model = DeepClassifier(num_classes=num_classes, architecture=architecture,
                               train_path=train_set, vld_path=vld_set, test_path=test_set,
                               optimizer=optimizer, lr=learning_rate,
                               batch_size=batch_size, transfer=transfer, tune_fc_only=tune_fc_only,
                               size=size)

        # Instantiate lightning trainer and train model
        trainer_args = {'gpus': gpus, 'max_epochs': num_epochs,
                        'callbacks': callbacks, "logger": logger}
        trainer = pl.Trainer(**trainer_args)
        trainer.fit(model)

        torch.cuda.empty_cache()


def test_categorization_model(folder):
    settings_file = [f for f in os.listdir(folder) if f.endswith(".py")][0]
    opt = load_python_file(os.path.join(folder, settings_file))
    architecture = folder.split("/")[-1].split("_")[-2]
    path = os.path.join(folder, "checkpoints", "last.ckpt")

    # Training & Output
    learning_rate = opt.learning_rate
    size = opt.size
    num_epochs = opt.num_epochs
    num_classes = opt.num_classes
    transfer = opt.transfer
    batch_size = opt.batch_size
    optimizer = opt.optimizer
    gpus = opt.gpus

    train_set = opt.train_set
    vld_set = opt.vld_set
    test_set = opt.test_set
    tune_fc_only = opt.tune_fc_only

    num_epochs = opt.num_epochs
    gpus = 1

    # Instantiate Model
    model = DeepClassifier(num_classes=num_classes, architecture=architecture,
                           train_path=train_set, vld_path=vld_set, test_path=test_set,
                           optimizer=optimizer, lr=learning_rate,
                           batch_size=batch_size, transfer=transfer, tune_fc_only=tune_fc_only,
                           size=size)
    model = model.load_from_checkpoint(path, num_classes=num_classes, architecture=architecture,
                                       train_path=train_set, vld_path=vld_set, test_path=test_set,
                                       optimizer=optimizer, lr=learning_rate,
                                       batch_size=batch_size, transfer=transfer, tune_fc_only=tune_fc_only,
                                       size=size)

    # Instantiate lightning trainer and train model
    trainer_args = {'gpus': gpus, 'max_epochs': num_epochs}
    trainer = pl.Trainer(**trainer_args)
    res = trainer.test(model)[0]
    res["model"] = architecture
    shutil.rmtree("lightning_logs")
    del (model)
    torch.cuda.empty_cache()
    return res


def train_multilabel_model(opt, settings_file, debug=False):
    # Dataset
    root = opt.root
    csv_filename = opt.csv_filename
    cols_features = opt.cols_features
    col_filename = opt.col_filename
    replace_in_filenames = opt.replace_in_filenames
    num_classes = opt.num_classes

    # Training & Output
    architectures = opt.architectures
    prefix = opt.prefix
    learning_rate = opt.learning_rate
    size = opt.size
    num_epochs = opt.num_epochs
    num_epochs_fc_only = opt.num_epochs_fc_only
    batch_size = opt.batch_size
    optimizer = opt.optimizer
    gpus = opt.gpus
    val_test_transform = opt.val_test_transform
    train_transform = opt.train_transform

    classes_of_categories = opt.classes_of_categories

    csv = pd.read_csv(os.path.join(root, csv_filename))
    for k, v in replace_in_filenames.items():
        csv["filename"] = [f.replace(k, v) for f in csv["filename"]]

    for architecture in architectures:
        output_foldername = opt.output_foldername + "_" + architecture

        output_foldername = prepare_outputfolder(
            output_foldername, settings_file)

        callback_best_model = ModelCheckpoint(dirpath=output_foldername, monitor='loss_val', save_top_k=1,
                                                filename=prefix + architecture +
                                                '_{epoch}-{loss_val:.2f}-{f1_val:.2f}-{acc_val:.2f}',
                                                save_last=True, mode="min")
        bar = MultilineProgressBar()
        callbacks = [callback_best_model, bar]
        logger = TensorBoardLogger(save_dir=os.getcwd(), name=output_foldername.split(
            "/")[0], version=output_foldername.split("/")[1])

        # Instantiate Model

        model = DeepClassifierMultilabel(architecture,
                                            csv, root, col_filename, cols_features,
                                            feature_categories=cols_features, classes_of_categories=classes_of_categories,
                                            val_test_transform=val_test_transform, train_transform=train_transform)

        if False:
            import matplotlib.pyplot as plt
            print(next(iter(model.train_dataloader())))
            plt.imshow(next(iter(model.train_dataloader()))[
                       "image"].cpu().detach().numpy()[1].transpose(1, 2, 0))
            plt.show()

        # Instantiate lightning trainer and train model
        if num_epochs_fc_only > 0:
            model.freeze_encoder(unfreeze=True)
            trainer_args = {'gpus': gpus, 'max_epochs': num_epochs_fc_only,
                            'callbacks': callbacks, "logger": logger}
            trainer = pl.Trainer(**trainer_args)
            trainer.fit(model)
            model.freeze_encoder(unfreeze=False)

        if num_epochs > 0:
            trainer_args = {'gpus': gpus, 'max_epochs': num_epochs,
                            'callbacks': callbacks, "logger": logger}
            trainer = pl.Trainer(**trainer_args)
            trainer.fit(model)

        del (model)
        torch.cuda.empty_cache()


def test_multilabel_model(opt, path, subset="val", verbose=0):
    # Dataset
    root = opt.root
    csv_filename = opt.csv_filename
    cols_features = opt.cols_features
    col_filename = opt.col_filename
    replace_in_filenames = opt.replace_in_filenames
    num_classes = opt.num_classes
    val_test_transform = opt.val_test_transform
    train_transform = opt.train_transform
    classes_of_categories = opt.classes_of_categories

    csv = pd.read_csv(os.path.join(root, csv_filename))
    for k, v in replace_in_filenames.items():
        csv["filename"] = [f.replace(k, v) for f in csv["filename"]]

    architecture = os.path.dirname(path).split("_")[-2]

    model = DeepClassifierMultilabel(architecture,
                                     csv, root, col_filename, cols_features,
                                     feature_categories=cols_features, classes_of_categories=classes_of_categories,
                                     val_test_transform=val_test_transform, train_transform=train_transform)
    model.load_state_dict(torch.load(path)['state_dict'])
    model = model.to(device)
    with torch.no_grad():
        trainer = pl.Trainer()
        if subset == "test":
            res = trainer.test(model)[0]
            
            f1s = {}#Add F1 globally (and not per epoch)
            precisions = {}
            recalls = {}
            accuracies = {}
            for name in model.intermediate_logs["tp"].keys():
                #for metric in ["tp", "tn", "fp", "fn"]:
                #    res[name + "_" + metric] = model.intermediate_logs[metric][name]

                tp = model.intermediate_logs["tp"][name]
                fp = model.intermediate_logs["fp"][name]
                fn = model.intermediate_logs["fn"][name]
                tn = model.intermediate_logs["tn"][name]

                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                #f1 = 2 * (precision * recall) / (precision + recall)

                accuracy = (tp+tn) / (fp+fn+tp+tn)
                tpr = tp / (tp+fn)
                tnr = tn / (tn+fp)

                f1 = (2 * tp) / (2*tp+fp+fn)#If there are positive samples F1 is defined as it is only undefined if all samples are tn.

                res["tpr_" + name] = tpr
                res["tnr_" + name] = tnr
                f1s["f1_" + name] = f1

                precisions["precision_" + name] = precision
                recalls["recall_" + name] = recall
                accuracies["acc_" + name] = accuracy

            for k, v in f1s.items():
                res[k] = v

            for k, v in precisions.items():
                res[k] = v

            for k, v in recalls.items():
                res[k] = v

            for k, v in accuracies.items():
                res[k] = v

            res["f1_macro"] = np.nanmean(list(f1s.values()))
        elif subset == "val":
            res = trainer.test(model)[0]
            #res = [v for k, v in res.items()]
        elif subset == "train":
            loader = model.train_dataloader()
            res = trainer.test(model, dataloaders=loader)[0]
            #res = [v for k, v in res.items()]
        else:
            assert False

    # y, preds = forward_pl_model(model, loader, verbose = verbose)#pl.Trainer().test(model, loader)
    # acc, fp, fn, tp, tn, precision_, recall_, f1_ = evaluate((y >.5).type(torch.uint8).cpu().detach().numpy(),
    #            preds.type(torch.uint8).cpu().detach().numpy())
    return res

def predict_and_save_multilabel(opt, path, verbose=0):
    # Dataset
    root = opt.root
    csv_filename = opt.csv_filename
    cols_features = opt.cols_features
    col_filename = opt.col_filename
    replace_in_filenames = opt.replace_in_filenames
    num_classes = opt.num_classes
    classes_of_categories = opt.classes_of_categories
    val_test_transform = opt.val_test_transform
    train_transform = opt.train_transform

    print(root)

    csv = pd.read_csv(os.path.join(root, csv_filename))

    for k, v in replace_in_filenames.items():
        csv["filename"] = [f.replace(k, v) for f in csv["filename"]]

    architecture = os.path.dirname(path).split("_")[-2]
    print(architecture)
    print(opt.cols_features)

    model = DeepClassifierMultilabel(architecture,
                                     csv, root, col_filename, cols_features,
                                     feature_categories=cols_features, classes_of_categories=classes_of_categories,
                                     val_test_transform=val_test_transform, train_transform=train_transform)
    model.load_state_dict(torch.load(path)['state_dict'])
    model = model.to(device)

    dataset = MultiLabelDataset(csv, model.root, model.col_filename,
                                    [], subset="all", transform=model.val_test_transform, shuffle = False)
    loader = DataLoader(dataset, batch_size=model.batch_size, shuffle=False,num_workers=4, drop_last=False)
    with torch.no_grad():
        trainer = pl.Trainer()
        predictions = trainer.predict(model, loader)
        predictions_keys = predictions[0].keys()
        predictions_flat = {}

        for key in predictions_keys:
            for preds in predictions:
                if not key in predictions_flat:
                    predictions_flat[key] = []
                predictions_flat[key].extend(list(preds[key].cpu().detach().numpy()))

        final_predictions = pd.DataFrame(predictions_flat)
        final_predictions.to_csv(os.path.join(os.path.dirname(path),
                        "predictions.csv"), index=False)

def get_last_ckpts(dir="logs", mode="last"):
    model_weights_pths = []
    for checkpoint_dir in os.listdir(dir):
        checkpoint_dir = os.path.join(dir, checkpoint_dir)
        if not os.path.isdir(checkpoint_dir):
            continue

        checkpoints = [f for f in os.listdir(
            checkpoint_dir) if f.endswith(".ckpt")]
        print(checkpoint_dir)
        checkpoints.remove("last.ckpt")  # keep only the best model
        if mode == "last":
            checkpoints = ["last.ckpt"]
        model_weights_pths.append(os.path.join(checkpoint_dir, checkpoints[0]))
    return model_weights_pths

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', action='store', type=str,
                        default="train_categorization")
    parser.add_argument('--settings', action='store', type=str,
                        default="settings/categorization_default.py")
    parser.add_argument('--logs', action='store', type=str, default="")
    parser.add_argument('--test_subset', action='store', type=str,
                        default="test")

    args = parser.parse_args()
    opt = load_python_file(args.settings)

    if args.task == "train_multilabel":
        prefix = "multilabel_"  # of outfilename
        train_multilabel_model(opt, args.settings)

    elif args.task == "train_categorization":
        prefix = "classification_"
        train_categorization_model(opt, args.settings)

    elif args.task == "test_multilabel":
        prefix = "multilabel_"  # of outfilename
        model_weights_path = get_last_ckpts(mode ="best")

        print("--------------")
        # Args for multilabel model
        csv = pd.read_csv(os.path.join(opt.root, opt.csv_filename))
        name_starts_with = prefix

        model_weights_pths = get_last_ckpts(dir="logs")

        columns = []
        rows = []
        for path in model_weights_pths:
            row = []
            columns = []
            res = test_multilabel_model(opt, path, subset = args.test_subset)
            if type(res) == type(None):
                continue
            for k,v in res.items():
                k_out = k
                k_out = k_out.replace("_test","")
                k_out = k_out.replace("_epoch","")
                columns.append(k_out)
                row.append(v)
            rows.append(row)
        table = pd.DataFrame(rows, columns =columns)
        print(os.listdir())
        table.to_csv("test_results.csv", index=False)

    elif args.task == "test_categorization":
        results = []
        for f in os.listdir(args.logs):
            if os.path.isfile(os.path.join(args.logs, f)):
                continue
            res = test_categorization_model(os.path.join(args.logs, f))
            results.append(res)
        column_names = list(results[0].keys())
        column_names.remove("model")
        column_names.insert(0, "model")  # make sure model is the first column
        values = [[res[k] for k in column_names] for res in results]
        df = pd.DataFrame(values, columns=column_names)
        df.to_csv(os.path.join(args.logs, "results_categorization.csv"))

    elif args.task == "predict_multilabel":
        model_weights_pths = get_last_ckpts(dir="logs", mode = "last")
        for path in model_weights_pths:
            predict_and_save_multilabel(opt, path)