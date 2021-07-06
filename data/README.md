# Dataset Pre-processing

## Generic Pre-processing
After downloading the dataset, please create a new folder named "images" under the folder "mini-Imagenet" or "tiered-ImageNet", and put all images in this folder. The provided data loader will read images from the "images" folder by default. Of course, it is also OK to change the read path. For example, for the mini-Imagenet dataset, please change the line 10 of "./feat/dataloader/mini_imagenet.py" as the path of the downloaded images.

### Mini-ImageNet
The Mini-ImageNet dataset is a subset of the ImageNet that includes a total number of 100 classes and 600 examples per class. We use the [Ravi's split](https://github.com/twitter/meta-learning-lstm), and use 64 classes as SEEN categories, 16 and 20 as two sets of UNSEEN categories for model validation and evaluation respectively. To download this dataset, please email [Sachin Ravi](http://www.cs.princeton.edu/~sachinr/) for further details and instructions.

### Tiered-ImageNet
Tiered-ImageNet includes a broader number of classes. There are 351, 97 and 160 classes for meta-training, meta-validation and meta-test respectively.


