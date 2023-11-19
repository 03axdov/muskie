<div align="center">
<img src="https://github.com/03axdov/muskie/assets/62298758/f3153aea-445f-4f09-a997-ad71f85f81f9">
</div>

# Description
> A Computer Vision library for Python. The aim of this framework is to allow users to create datasets and use their models without having to write too much code. I.e. a simpler and more lightweight version of TensorFlow or PyTorch, specifically focused on Computer Vision. Currently in development


The framework uses Numpy, Matplotlib, Pillow, and some smaller libraries. To install these, run:
```
pip install -r requirements.txt
```


## Data Handling
Current syntax for creating a dataset from a folder of images, and then displaying the images in a grid:
```python
from muskie.data import create_dataset, display_data

path = "images/fish_images"
image_dimensions = (600, 500)

data = create_dataset(path, dimensions=image_dimensions, create_labels=True, split="_") # returns (images, labels, label_vector)
display_data(data,rows=3,cols=3)

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
The labels are computed by studying the part of filenames in front of the 'split' value passed to create_dataset. I.e. arapaima_1.jpg, arapaima_2.jpg ... are all classified as arapaima (0 in labels and "arapaima" in label_vector)

## Layers
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

## GPU functionality
Features that run faster with a GPU, such as processing of the convolutional layers, can be used. For this, 'cudatoolkit' must be installed.
