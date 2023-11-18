<div align="center">
<img src="https://github.com/03axdov/muskie/assets/62298758/d2f3e5c1-dd2a-4982-ab17-8cd2b0bd31ac" width="315" height="315">
</div>

# Muskie Computer Vision
> A Computer Vision library for Python. The aim of this framework is to allow users to create datasets and use their models without having to write too much code. In development


The framework uses Numpy, Matplotlib, and Pillow. To install these, run:
```
pip install -r requirements.txt
```



Current syntax for creating a dataset from a folder of images, and then displaying the images in a grid:
```python
from .src.utils.create_dataset import create_dataset
from .src.utils.display_data import display_data

path = "images/fish_images"
labels = [0,0,0,1,1,1,2,2,2]  
image_dimensions = (600, 500)
label_vector = ("Arapaima", "Marlin", "Muskie")  # I.e. label 2 => Muskie

data = create_dataset(path, labels, image_dimensions) #(images, labels)
display_data(data,rows=3,cols=3,label_vector=label_vector)
```
