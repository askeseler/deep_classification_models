num_classes = 20

#train_set = "/media/gerstenberger/Data1/projects/deep_classification_models/archive/dataset"
#vld_set = "/media/gerstenberger/Data1/projects/deep_classification_models/archive/dataset"
#test_set = "/media/gerstenberger/Data1/projects/deep_classification_models/archive/dataset"

train_set = "/media/gerstenberger/Data1/datasets/rotten_fruits/image_folder_split/train"
vld_set = "/media/gerstenberger/Data1/datasets/rotten_fruits/image_folder_split/test"
test_set = "/media/gerstenberger/Data1/datasets/rotten_fruits/image_folder_split/val"


batch_size = 128
transfer = True
tune_fc_only = False
optimizer = "adam"
learning_rate = 1e-3
size = (128, 128)
prefix = "categorization_"#must start with categorization_ or multilabel_


num_epochs = 25
gpus = 1

architectures = ["resnet18", "alexnet","resnet18", "resnet50", "vgg11", "densenet121"]
output_foldername = "categorization_default"