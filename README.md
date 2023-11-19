<div align="center">
<img src="https://github.com/03axdov/muskie/assets/62298758/ae9ee807-0e80-464e-98d0-fe75ffbb62c4">
</div>

# A Computer Vision Library
> A Computer Vision library for Python. The aim of this framework is to allow users to create datasets and use their models without having to write too much code. In development


The framework uses Numpy, Matplotlib, and Pillow. To install these, run:
```
pip install -r requirements.txt
```


## Data Handling
Current syntax for creating a dataset from a folder of images, and then displaying the images in a grid:
```python
from muskie.utils.create_dataset import create_dataset
from muskie.utils.display_data import display_data

path = "images/fish_images"
image_dimensions = (600, 500)

data = create_dataset(path, labels, image_dimensions) # returns (images, labels, label_vector)
display_data(data,rows=3,cols=3,label_vector=label_vector)

images, labels, label_vector = data
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

## Layers
A Conv2D layer can be created like so
```python
from muskie.layers.conv2d import Conv2D

layer = Conv2D(3, nbr_kernels=128, padding=1, std=0.01, mean=0.0)
convolution = layer.calculate(images[0])
print(convolution.shape)
```
Which gives
```
(600,500,128)
```
