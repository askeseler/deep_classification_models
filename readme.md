This is a repository to fit the following models for multi class classification (categorization) and multi label classification (feature detection):

- a custom shallow network
- alexnet
- resnet18
- resnet34
- resnet50
- resnet101
- resnet152
- densenet121
- densenet161
- densenet169
- densenet201
- vgg11
- vgg11_bn
- vgg13
- vgg13_bn
- vgg16
- vgg16_bn
- vgg19
- vgg19_bn

Hyperparameters are specified in the python files in /settings. The path to the settings.py file must be specified upon calling main.py alongside the task the script should complete (e.g. train_multilabel).

For multilabel classification the labels are supposed to be provided in a .csv with a column containing the relative path to the image files.
For multiclass classification the path to the ImageFolders of the train, test, and validation subset must be specified.
This information is meant to be declared in the settings file.

The outputs are written to a log directory which will also contain a copy of the settings.py file such that the relationship between output performance and hyperparamters can be evaluated.
