<div align="center">
<img src="https://github.com/03axdov/muskie/assets/62298758/214bae89-6c9c-4e84-83cd-6a78bf42ca4b">
</div>

# Table of Contents
- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Data Handling](#data-handling)
- [Layers](#layers)
  - [Conv2D Layer](#conv2d-layer)
- [Models](#models)
- [GPU functionality](#gpu-functionality)

# Introduction
> A Computer Vision library for Python. The aim of this framework is to allow users to create datasets and use their models without having to write too much code. I.e. a simpler and more lightweight version of TensorFlow or PyTorch, specifically focused on Computer Vision. Currently in development


The framework uses Numpy, Matplotlib, Pillow, and some smaller libraries. To install these, run:
```
pip install -r requirements.txt
```


# Data Handling
Current syntax for creating a dataset from a folder of images, and then displaying the images in a grid:
```python
from muskie.data import create_dataset, display_data

path = "images/fish_images"
image_dimensions = (600, 500)

data = create_dataset(path, dimensions=image_dimensions, create_labels=True, split="_") # returns (images, labels, label_vector)
display_data(data,rows=3,cols=3)

images, labels, label_vector = data.as_tuple()
print(labels)
print(label_vector)
```
Which gives
<div align="center">
<img src="https://github.com/03axdov/muskie/assets/62298758/e0a5221b-d388-4f67-91d8-d9ea4b0950f1" width="600" height="400">
</div>

```
[0,0,0,1,1,1,2,2,2]
["arapaima", "marlin", "pike"]
```
where an individual label is an index in the label_vector, i.e. an image with the label 2 is of a pike. 
The labels are computed by studying the part of filenames in front of the 'split' value passed to create_dataset. I.e. arapaima_1.jpg, arapaima_2.jpg ... are all classified as arapaima (0 in labels and "arapaima" in label_vector)

# Layers
## Conv2D Layer
A Conv2D layer can be created like so
```python
from muskie.layers import Conv2D

layer = Conv2D(nbr_kernels=128, kernel_size=3, padding=1, std=0.01, mean=0.0)
convolution = layer.calculate(images[0])
print(convolution.shape)
```
Which gives
```
(600,500,128)
```

# Models
Models can be created with a list of layers, and layers can later be added as well.
```python
from muskie.models import ClassificationModel

layer1 = Conv2D(nbr_kernels=32, kernel_size=3, padding=1)
layer2 = Conv2D(nbr_kernels=64, kernel_size=3, padding=1)

model = ClassificationModel([layer1])
model.add(layer2)
prediction = model.predict(images[0])
print(prediction.shape)
```
which gives
```
(600,500,64)
```

# GPU functionality
For features that run faster with a GPU, such as processing of the convolutional layers, Muskie can be specified to use GPU. For this to work properly, 'cudatoolkit' must be installed. This can be installed by being in a conda environment and running
```
conda install cudatoolkit
```
To run on gpu, the following code must be run before operations:
```python
from muskie.core import use_gpu, dont_use_gpu
use_gpu()
```
Running on GPU is turned off by default, but to manually set it to false run
```python
dont_use_gpu()
```
