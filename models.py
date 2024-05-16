from doctest import ELLIPSIS_MARKER
from inspect import ArgSpec
from os import R_OK
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from utils import *
from datasets import *
from tqdm import tqdm
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import torchvision.models as models
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.optim import SGD, Adam
import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
import warnings
from pathlib import Path
from argparse import ArgumentParser
from torcheval.metrics.functional import multiclass_f1_score
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ShallowCNN(nn.Module):
    def __init__(self, num_classes, size=(32,32)):
        super(ShallowCNN, self).__init__()
        self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),)
        self.output_shape = self.get_output_shape(size, self.features)
        self.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(self.output_shape, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        fts = self.features(x)
        clf = self.fc(fts)
        return clf

    def get_output_shape(self, size, encoder):
        out = encoder(torch.zeros([3, size[0], size[1]]).unsqueeze(0))
        return torch.prod(torch.tensor(out.shape))

class DeepClassifierBaseClass(pl.LightningModule):
    """ """
    def __init__(self, num_classes, architecture,
                 train_path, vld_path, test_path=None,
                 optimizer='adam', lr=1e-3, optimizer_specs=None, batch_size=16,
                 transfer=True, tune_fc_only=True, size=(500, 500)):
        super().__init__()
        self.__dict__.update(locals())
        self.architecture = architecture

        self.val_test_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.48232,), (0.23051,))])
        self.train_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.48232,), (0.23051,))])

        base_models = {'shallow': ShallowCNN,
                        'alexnet': models.alexnet,
                       'resnet18': models.resnet18,
                       'resnet34': models.resnet34,
                       'resnet50': models.resnet50,
                       'resnet101': models.resnet101,
                       'resnet152': models.resnet152,
                       'densenet121': models.densenet121,
                       'densenet161': models.densenet161,
                       'densenet169': models.densenet169,
                       'densenet201': models.densenet201,
                       'vgg11': models.vgg11,
                       'vgg11_bn': models.vgg11_bn,
                       'vgg13': models.vgg13,
                       'vgg13_bn': models.vgg13_bn,
                       'vgg16': models.vgg16,
                       'vgg16_bn': models.vgg16_bn,
                       'vgg19': models.vgg19,
                       'vgg19_bn': models.vgg19_bn}
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.optimizer = optimizers[optimizer]
        self.optimizer_specs = self.optimizer_specs

        self.criterion = nn.BCEWithLogitsLoss() if num_classes == 2 else nn.CrossEntropyLoss()

        if architecture.startswith("shallow"):
            self.base_model = ShallowCNN(num_classes, size)
        else:
            self.base_model = base_models[architecture](
                pretrained=transfer, num_classes=1000)

        if architecture.startswith("alexnet"):
            linear_size = 4096
            self.base_model._modules['classifier']._modules[str(
                6)] = nn.Linear(linear_size, num_classes)
            if tune_fc_only:
                self.freeze_encoder()

        elif architecture.startswith("resnet") or architecture.startswith("densenet"):
            linear_size = list(self.base_model.children())[-1].in_features

            last_layer_name = "fc" if architecture.startswith("resnet") else "classifier"
            self.base_model._modules[last_layer_name] = nn.Linear(
                linear_size, num_classes)
            self.last_layer = self.base_model._modules[last_layer_name]

            if tune_fc_only:
                self.freeze_encoder()

        elif architecture.startswith("vgg"):
            linear_size = self.base_model._modules["classifier"]._modules[str(
                6)].in_features
            self.base_model._modules["classifier"]._modules[str(
                6)] = nn.Linear(linear_size, num_classes)
            if tune_fc_only:
                self.freeze_encoder()
        elif architecture.startswith("shallow"):
            pass
        else:
            raise NotImplemented("Architecture not supported")

        self.train_path = train_path
        self.vld_path = vld_path
        self.test_path = test_path
        self.batch_size = batch_size

    def freeze_encoder(self, unfreeze=False):
        if self.architecture.startswith("resnet") or self.architecture.startswith("densenet"):
            for child in list(self.base_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = unfreeze
        elif self.architecture.startswith("vgg"):
            for child in list(self.base_model._modules["classifier"].children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = unfreeze
            for child in list(self.base_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = unfreeze
        elif self.architecture.startswith("alexnet"):
            for child in list(self.base_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = unfreeze
        elif self.architecture.startswith("shallow"):
            print("Not supported for shallow net")
        else:
            raise NotImplemented("Architecture not supported")

    def forward(self, X):
        return self.base_model(X)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)


class DeepClassifier(DeepClassifierBaseClass):
    def __init__(self, num_classes, architecture,
                 train_path=None, vld_path=None, test_path=None,
                 optimizer='adam', lr=1e-3, optimizer_specs=None, batch_size=16,
                 transfer=True, tune_fc_only=True, size=(500, 500)):
        super().__init__(num_classes, architecture,
                         train_path, vld_path, test_path,
                         optimizer, lr, optimizer_specs, batch_size,
                         transfer, tune_fc_only, size)

    def train_dataloader(self):
        img_train = ImageFolder(
            self.train_path, transform=self.train_transform)
        return DataLoader(img_train, batch_size=self.batch_size, shuffle=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        if self.num_classes == 2:
            y = F.one_hot(y, num_classes=2).float()
        loss = self.criterion(preds, y)

        acc = (y == torch.argmax(preds, axis=1)) \
            .type(torch.FloatTensor).mean()

        self.log("train_loss", loss.detach(), on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc.detach(), on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def val_dataloader(self):
        img_val = ImageFolder(self.vld_path, transform=self.val_test_transform)
        return DataLoader(img_val, batch_size=1, shuffle=False)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        if self.num_classes == 2:
            y = F.one_hot(y, num_classes=2).float()

        loss = self.criterion(preds, y)

        acc = (y == torch.argmax(preds, axis=1)) \
            .type(torch.FloatTensor).mean()
        self.log("loss_val", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("acc_val", acc, on_epoch=True, prog_bar=True, logger=True)

    def test_dataloader(self):
        img_test = ImageFolder(
            self.test_path, transform=self.val_test_transform)
        return DataLoader(img_test, batch_size=1, shuffle=False)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        if self.num_classes == 2:
            y = F.one_hot(y, num_classes=2).float()

        loss = self.criterion(preds, y)

        acc = (y == torch.argmax(preds, axis=1)) \
            .type(torch.FloatTensor).mean()
        self.log("test_loss", loss.detach(),
                 on_step=True, prog_bar=True, logger=True)
        self.log("test_acc", acc.detach(), on_step=True,
                 prog_bar=True, logger=True)

    def configure_optimizers(self):
        if type(self.optimizer_specs) != type(None):
            optimizer_specs = [{'params': self.base_model.parameters(), 'lr': 1e-4, 'weight_decay': 1e-3},
                               {'params': self.last_layer.parameters(), 'lr': 3e-3, 'weight_decay': 1e-3},]
            optimizer = torch.optim.Adam(optimizer_specs)
            scheduler = {'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step_size, gamma=0.1),
                         'interval': 'epoch', 'frequency': 1}
            scheduler["scheduler"].step()
            return ([optimizer], [scheduler])
        else:
            return self.optimizer(self.parameters(), lr=self.lr)

class DeepClassifierMultilabel(pl.LightningModule):
    def __init__(self, architecture,
                 csv, root, col_filename, cols_features, feature_categories = [], classes_of_categories = [],
                 lr=1e-4, optimizer_specs=None, batch_size=16,
                 transfer=True, tune_fc_only=True, size=(224, 224), val_test_transform = None, train_transform = None, col_train_or_test = None,):
        super().__init__()

        base_models = {'shallow': ShallowCNN,
                        'alexnet': models.alexnet,
                       'resnet18': models.resnet18,
                       'resnet34': models.resnet34,
                       'resnet50': models.resnet50,
                       'resnet101': models.resnet101,
                       'resnet152': models.resnet152,
                       'densenet121': models.densenet121,
                       'densenet161': models.densenet161,
                       'densenet169': models.densenet169,
                       'densenet201': models.densenet201,
                       'vgg11': models.vgg11,
                       'vgg11_bn': models.vgg11_bn,
                       'vgg13': models.vgg13,
                       'vgg13_bn': models.vgg13_bn,
                       'vgg16': models.vgg16,
                       'vgg16_bn': models.vgg16_bn,
                       'vgg19': models.vgg19,
                       'vgg19_bn': models.vgg19_bn}

        if architecture.startswith("resnet"):
            backbone = base_models[architecture](pretrained=True)
            in_features = list(backbone.children())[-1].in_features 
            self.encoder = nn.Sequential(*(list(backbone.children())[:-1]))
        elif architecture.startswith("densenet"):
            backbone = base_models[architecture](pretrained=True)
            in_features = list(backbone.children())[-1].in_features
            backbone._modules["classifier"] = nn.Linear(in_features, in_features)
            self.encoder = backbone
        elif architecture.startswith("alexnet"):
            in_features = 9216
            backbone = base_models[architecture](pretrained=True)
            self.encoder = nn.Sequential(*(list(backbone.children())[:-1]))           
        elif architecture.startswith("shallow"):
            in_features = 512
            self.encoder = ShallowCNN(in_features, size)
        elif architecture.startswith("vgg"):
            encoder = base_models[architecture](pretrained=True)
            in_features = encoder._modules["classifier"]._modules[str(6)].in_features
            encoder.classifier = nn.Sequential(*[encoder.classifier[i] for i in range(len(encoder.classifier)-1)])
            self.encoder = encoder
        else:
            raise NotImplementedError()

        self.classification_heads = []
        self.feature_categories = feature_categories
        self.classes_of_categories = classes_of_categories
        self.train_transform = train_transform
        self.val_test_transform = val_test_transform
        self.col_train_or_test = col_train_or_test
        
        for head_idx, n in enumerate(classes_of_categories):
            head = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features=in_features, out_features=n)).to(device)
            #self.__dict__["head_" + str(n)] = head
            for i, param in enumerate(head.parameters()):
                self.register_parameter(name='head_' + str(head_idx) +"_"+ str(i), param=param)
            self.classification_heads.append(head)

        self.csv = csv
        self.root = root
        self.col_filename = col_filename
        self.cols_features = cols_features
        self.loss_func = nn.CrossEntropyLoss()
        self.lr = lr
        self.batch_size = batch_size

        self.intermediate_logs = {}#To be averaged at end of testing
        self.intermediate_logs["tp"] = {}
        self.intermediate_logs["tn"] = {}
        self.intermediate_logs["fp"] = {}
        self.intermediate_logs["fn"] = {}

    def criterion(self, loss_func, outputs, batch):
        losses = 0
        for i, key in enumerate(outputs):
            losses += loss_func(outputs[key], batch['labels'][f'{key}'].to(device))
        return losses

    def forward(self, x):
        x = x.to(device)
        self.encoder = self.encoder.to(device)
        for i in range(len(self.classification_heads)):
            self.classification_heads[i] = self.classification_heads[i].to(device)
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        return {name:head(x) for name, head in zip(self.feature_categories, self.classification_heads)}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch["image"]
        y = batch["labels"]
        preds = self(x)
        for i, name in enumerate(preds.keys()):
            pred_int = torch.max(preds[name], axis=1)[1].detach()
            preds[name] = pred_int
        return preds

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["labels"]
        preds = self(x)
        loss = self.criterion(self.loss_func, preds, batch)
        accs = []
        f1s = []

        tp = {}
        fp = {}
        tn = {}
        fn = {}
        for i, name in enumerate(preds.keys()):
            pred_int = torch.max(preds[name], axis=1)[1].detach()
            true_int = y[name]

            acc = torch.mean(torch.tensor(pred_int==true_int).type(torch.float)).cpu()

            f1 = multiclass_f1_score(pred_int, true_int, num_classes=self.classes_of_categories[i]).cpu()
            f1s.append(f1)
            accs.append(acc)

            self.log("acc_train_" + name, acc, on_step=True,
                            on_epoch=False, prog_bar=True, logger=True)
            self.log("f1_train_" + name, f1, on_step=True,
                            on_epoch=False, prog_bar=False, logger=True)

            if self.classes_of_categories[i] == 2:
                tp[name] = torch.sum(torch.logical_and(pred_int==1, true_int == 1).type(torch.float))
                fp[name] = torch.sum(torch.logical_and(pred_int==1, true_int == 0).type(torch.float))
                tn[name] = torch.sum(torch.logical_and(pred_int==0, true_int == 0).type(torch.float))
                fn[name] = torch.sum(torch.logical_and(pred_int==0, true_int == 1).type(torch.float))

        self.log("acc_train", np.mean(accs), on_step=True,
                    on_epoch=False, prog_bar=True, logger=True)
        self.log("f1_train", np.mean(f1s), on_step=True,
                    on_epoch=False, prog_bar=False, logger=True)
        self.log("loss_train", np.mean(loss.detach().cpu().numpy()), on_step=True,
                    on_epoch=False, prog_bar=False, logger=True)
        return {"loss": loss, "tp":tp, "fp":fp, "tn": tn, "fn":fn}
    
    def log_tpr_tnr(self, outputs, subset = ""):
        if len(subset) != 0:
            subset = "_" + subset
        for name in self.cols_features:
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for r in outputs:
                assert "tp" + subset in r.keys()
                assert "tn" + subset in r.keys()
                assert "fp" + subset in r.keys()
                assert "fn" + subset in r.keys()

                tp += r["tp" + subset][name]
                tn += r["tn" + subset][name]
                fp += r["fp" + subset][name]
                fn += r["fn" + subset][name]
            tpr = tp / torch.tensor(tp+fn)
            tnr = tn / torch.tensor(tn+fp)
            self.log("tpr" + subset + "_" + name, tpr)
            self.log("tnr" + subset + "_" + name, tnr)
    
    def training_epoch_end(self, outputs):
        self.log_tpr_tnr(outputs)
        super().training_epoch_end(outputs)

    def test_epoch_end(self, outputs):
        for name in self.cols_features:
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            for r in outputs:
                assert "tp" in r.keys()
                assert "tn" in r.keys()
                assert "fp" in r.keys()
                assert "fn" in r.keys()

                tp += r["tp"][name]
                tn += r["tn"][name]
                fp += r["fp"][name]
                fn += r["fn"][name]

            if name not in self.intermediate_logs["tp"].keys():
                self.intermediate_logs["tp"][name] = tp.cpu().detach().numpy()
                self.intermediate_logs["tn"][name] = tn.cpu().detach().numpy()
                self.intermediate_logs["fp"][name] = fp.cpu().detach().numpy()
                self.intermediate_logs["fn"][name] = fn.cpu().detach().numpy()

            self.intermediate_logs["tp"][name] += tp.cpu().detach().numpy()
            self.intermediate_logs["tn"][name] += tn.cpu().detach().numpy()
            self.intermediate_logs["fp"][name] += fp.cpu().detach().numpy()
            self.intermediate_logs["fn"][name] += fn.cpu().detach().numpy()

        print(self.intermediate_logs)

        super().training_epoch_end(outputs)
        

    def val_epoch_end(self, outputs):
        self.log_tpr_tnr(outputs, subset = "val")
        super().training_epoch_end(outputs)

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = {k:v.to(device) for k,v in batch["labels"].items()}
        preds = self(x)
        loss = self.criterion(self.loss_func, preds, batch)
        accs = []
        f1s = []

        tp = {}
        fp = {}
        tn = {}
        fn = {}
        for i, name in enumerate(preds.keys()):
            pred_int = torch.max(preds[name], axis=1)[1].detach()
            true_int = y[name]
            acc = torch.mean(torch.tensor(pred_int==true_int).type(torch.float)).cpu()

            f1 = multiclass_f1_score(pred_int, true_int, num_classes=self.classes_of_categories[i]).cpu()
            f1s.append(f1)
            accs.append(acc)

            self.log("acc_val_" + name, acc, on_step=True,
                            on_epoch=True, prog_bar=True, logger=True)
            self.log("f1_val_" + name, f1, on_step=True,
                            on_epoch=True, prog_bar=False, logger=True)
            if self.classes_of_categories[i] == 2:
                tp[name] = torch.sum(torch.logical_and(pred_int==1, true_int == 1).type(torch.float))
                fp[name] = torch.sum(torch.logical_and(pred_int==1, true_int == 0).type(torch.float))
                tn[name] = torch.sum(torch.logical_and(pred_int==0, true_int == 0).type(torch.float))
                fn[name] = torch.sum(torch.logical_and(pred_int==0, true_int == 1).type(torch.float))
                
        self.log("acc_val", np.mean(accs), on_step=True,
                    on_epoch=True, prog_bar=True, logger=True)
        self.log("f1_val", np.mean(f1s), on_step=True,
                    on_epoch=True, prog_bar=False, logger=True)
        self.log("loss_val", np.mean(loss.detach().cpu().numpy()), on_step=True,
                    on_epoch=True, prog_bar=True, logger=True)
        return {"loss_val": loss, "tp_val": tp, "fp_val": fp, "tn_val": tn, "fn_val": fn}

    def test_step(self, batch, batch_idx):
        x = batch["image"]
        y = {k:v.to(device) for k,v in batch["labels"].items()}
        preds = self(x)
        loss = self.criterion(self.loss_func, preds, batch)
        accs = []

        tp = {}
        fp = {}
        tn = {}
        fn = {}

        for i, name in enumerate(preds.keys()):
            pred_int = torch.max(preds[name], axis=1)[1].detach()
            true_int = y[name].type(torch.int)

            acc = torch.mean(torch.tensor(pred_int==true_int).type(torch.float)).cpu()
            accs.append(acc)

            self.log("acc_test_" + name, acc, on_step=True,
                            on_epoch=False, prog_bar=True, logger=True)
            if self.classes_of_categories[i] == 2:
                tp[name] = torch.sum(torch.logical_and(pred_int==1, true_int == 1).type(torch.float))
                fp[name] = torch.sum(torch.logical_and(pred_int==1, true_int == 0).type(torch.float))
                tn[name] = torch.sum(torch.logical_and(pred_int==0, true_int == 0).type(torch.float))
                fn[name] = torch.sum(torch.logical_and(pred_int==0, true_int == 1).type(torch.float))

        self.log("acc_test", np.mean(accs), on_step=True,
                    on_epoch=True, prog_bar=True, logger=True)
        self.log("loss_test", np.mean(loss.detach().cpu().numpy()), on_step=True,
                    on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "tp": tp, "fp": fp, "tn": tn, "fn": fn}
        
    def all_dataloader(self):
        dataset = MultiLabelDataset(self.csv, self.root, self.col_filename,
                                    self.cols_features, subset="all", transform=self.val_test_transform, shuffle = False)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False,num_workers=4, drop_last=False)

    def train_dataloader(self):
        dataset = MultiLabelDataset(self.csv, self.root, self.col_filename,
                                    self.cols_features, subset="train", transform=self.train_transform, shuffle = False, col_train_or_test = self.col_train_or_test)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True,num_workers=4, drop_last=True)

    def val_dataloader(self):
        dataset = MultiLabelDataset(self.csv, self.root, self.col_filename,
                                    self.cols_features, subset="val", transform=self.train_transform, shuffle = False, col_train_or_test = self.col_train_or_test)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False,num_workers=4, drop_last=True)

    def test_dataloader(self):
        dataset = MultiLabelDataset(self.csv, self.root, self.col_filename,
                                    self.cols_features, subset="test", transform=self.train_transform, shuffle = True, col_train_or_test = self.col_train_or_test)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False,num_workers=4, drop_last=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
