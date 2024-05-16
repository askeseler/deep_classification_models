from torchvision import transforms

# Dataset
root = "/media/gerstenberger/Data1/datasets/asparagus/processed_images/"
csv_filename = "final_annotations_v6.csv"
cols_features = ["hallow","bent", "violet", "rust"]
col_filename = "filename"
replace_in_filenames = {}#{"imagefolder_v5":"imagefolder_v5_cropped_head"}
num_classes = len(cols_features)

# Training & Output
architectures = ["shallow","alexnet","resnet18", "resnet50", "resnet101", "vgg11", "densenet121"]
prefix = "multilabel_"#must start with categorization_ or multilabel_
batch_size = 128
transfer = False
tune_fc_only = False
optimizer = "adam"
learning_rate = 1e-3
num_epochs = 25
num_epochs_fc_only = 1
size = (256, 64)
gpus = 1

#val_test_transform = transforms.Compose([transforms.Resize(size),transforms.ToTensor(),transforms.Lambda(lambda x: x/2)])
#train_transform = transforms.Compose([transforms.Resize(size),transforms.RandomHorizontalFlip(0.3),transforms.RandomVerticalFlip(0.3),transforms.ToTensor(),transforms.Lambda(lambda x: x/2)])

output_foldername = "multilabel_default"


