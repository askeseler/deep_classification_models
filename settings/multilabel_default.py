from torchvision import transforms
import pandas as pd

# Dataset
root = "./stanford_cars"#"/media/gerstenberger/Data1/datasets/asparagus/processed_images/"
csv_filename = "../stanford_cars/labels.csv"#"final_annotations_v5.csv"
cols_features = ["brand", "vehicle_type", "epoch"]#["FreshApple", "RottenApple"]
classes_of_categories =[6,5,2]
col_filename = "path"
replace_in_filenames = {}#{"imagefolder_v5":"imagefolder_v5_cropped_head"}
num_classes = len(cols_features)

# Training & Output
architectures = ["resnet34", "resnet50", "resnet101", "resnet152", "alexnet", "vgg11", "densenet121"]
prefix = "multilabel_"#must start with categorization_ or multilabel_
batch_size = 16
transfer = True
tune_fc_only = False
optimizer = "adam"
learning_rate = 1e-4
num_epochs = 10
num_epochs_fc_only = 0
size = (224, 224)
gpus = 1

val_test_transform = data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
train_transform = data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

output_foldername = "multilabel_default"


