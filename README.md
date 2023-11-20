<div align="center">
<img src="https://github.com/03axdov/muskie/assets/62298758/214bae89-6c9c-4e84-83cd-6a78bf42ca4b">
</div>

# Table of Contents
- [Table of Contents](#table-of-contents)
- [Documentation](#documentation)
- [Data Handling](#data-handling)
  - [The 'Data' class](#the-data-class)
  - [Dataset Creation](#dataset-creation)
  - [Displaying Data](#displaying-data)
- [Layers](#layers)
  - [Conv2D Layer](#conv2d-layer)
- [Models](#models)
- [GPU functionality](#gpu-functionality)

# Documentation
> A Computer Vision library for Python. The aim of this framework is to allow users to create datasets and use their models without having to write too much code. I.e. a simpler and more lightweight version of TensorFlow or PyTorch, specifically focused on Computer Vision. Currently in development


The framework uses Numpy, Matplotlib, Pillow, and some smaller libraries. To install these, run:
```
pip install -r requirements.txt
```

It is important to note that the user's main script must have the following if-clause for Muskie to work properly
```python
# imports here

if __name__ == "__main__":
  # your code here
```

# Data Handling
## The 'Data' class
Muskie uses the Data class to store information that can be used to visualize images, train models etc. It essentially works as a dataset, containing 3 numpy arrays: images, labels, and label_vectors. Users can add data to data, or arrays to the individual arrays that Data classes contain. Additionally, the function equals() determines whether two instances of Data are equal. An instance of Data can be created by
```python
from muskie.data import Data
data1 = Data(images=np.array([1,2,3])) # all the arrays are empty when omitted from the constructor
data2 = Data(images=np.array([1,2,3]))

data1.add(data2)
data2.add_images(np.array([1,2,3]))
assert data1.equals(data2)
```
## Dataset Creation
There are currently two ways of creating datasets. One takes a folder that contains only images. The filenames can be used to generate labels. The other takes a folder with subdirectories that contain only images. The names of the subdirectories can be used to generate labels.
```python
from muskie.datasets import create_dataset, create_dataset_subdirectories

path1 = "images/fish_images"
path2 = "images/fish_images_subdirectories"
image_dimensions = (600, 500)

data1 = create_dataset(path1, dimensions=image_dimensions, create_labels=True, split="_")
data2 = create_dataset_subdirectories(path2, dimensions=image_dimensions, create_labels=True)
assert data1.equals(data2)
```
## Displaying Data
Data can be displayed using the 'display_data' function
```python
from muskie.data import display_data
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
