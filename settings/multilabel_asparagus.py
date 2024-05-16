from torchvision import transforms
import pandas as pd

# Dataset
root = "/media/gerstenberger/Data1/datasets/asparagus/processed_images/"
csv_filename = "labels.csv"
cols_features = ["violet", "flower", "rust", "bent", "hallow", "green"]#["FreshApple", "RottenApple"]
classes_of_categories =[2,2,2,2,2,2]
col_filename = "filename"
replace_in_filenames = {}#{"train/":"", "test/":""}
num_classes = len(cols_features)

# Training & Output
architectures = ["alexnet", "resnet101", "vgg19","resnet18", "resnet50", "vgg11", "densenet121"]
prefix = "multilabel_"#must start with categorization_ or multilabel_
batch_size = 128
transfer = True
tune_fc_only = False
optimizer = "adam"
learning_rate = 1e-4
num_epochs = 2
num_epochs_fc_only = 0
size = (224, 224)
gpus = 1

val_test_transform = data_transforms = transforms.Compose([
        transforms.Resize((448,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
train_transform = data_transforms = transforms.Compose([
        transforms.Resize((448,128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

output_foldername = "multilabel_default"


