from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import numpy as np
import torch

class MultiLabelDataset(Dataset):
    def __init__(self, csv, root_dir, col_img_pth = "filename", cols_features = [], subset = "train",
                 train_val_test = [.6,.2,.2], random_seed = 42, transform = None, shuffle = False, col_train_or_test=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if type(csv) == type(str):
            self.csv = pd.read_csv(csv)
        else:
            self.csv = csv
        if shuffle:
            self.csv.sample(frac=1, random_state = random_seed)
        self.root_dir = root_dir
            
        if not col_train_or_test:
            train_val_test = [int(len(csv) * train_val_test[0]),
                              int(len(csv) * train_val_test[1]),
                              int(len(csv) * train_val_test[2])]
            if subset == "train":
                self.csv = self.csv.iloc[:train_val_test[0]]
            elif subset == "val":
                self.csv = self.csv.iloc[train_val_test[0]:train_val_test[0]+train_val_test[1]]
            elif subset == "test":
                self.csv = self.csv.iloc[train_val_test[0]+train_val_test[1]:]
            elif subset == "all":
                self.csv = self.csv#.iloc[:100]
            else:
                assert False
        else:
            assert col_train_or_test in self.csv.keys()
            if subset == "train":
                self.csv = self.csv.loc[self.csv[col_train_or_test].values == 0]
            elif subset == "test" or subset == "val":
                self.csv = self.csv.loc[self.csv[col_train_or_test].values == 1]   
        self.imgs = [os.path.join(root_dir, img) for img in list(self.csv[col_img_pth])]
        self.features = [np.array(self.csv[c]>.5, dtype=int) for c in cols_features]

        self.transform = transform
        self.cols_features =cols_features

    def __len__(self):
        return len(self.imgs)#//100

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert('RGB')#torch.tensor(np.array(), dtype=np.float).permute(2,0,1).type(torch.float)
        if type(self.transform) != type(None):
            img = self.transform(img)
        #labels = torch.tensor(self.features[idx], dtype=torch.float)

        return {"image":img, "labels": {k:self.csv[k].values[idx].astype(int) for k in self.cols_features}, "path":self.imgs[idx]}

