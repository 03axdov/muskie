<div align="center">
<img src="https://github.com/03axdov/muskie/assets/62298758/214bae89-6c9c-4e84-83cd-6a78bf42ca4b">
</div>

# Documentation
> A Machine Learning library for Python. The aim of this framework is to allow users to create datasets and use their models without having to write too much code. I.e. a simpler and more lightweight version of TensorFlow or PyTorch, intended to have additional features for Computer Vision, such as codeless dataset creation. Currently in development

- [Documentation](#documentation)
- [Code Structure](#code-structure)
- [Utils](#utils)
- [Data Handling](#data-handling)
  - [The 'Data' class](#the-data-class)
  - [The 'PredictionData' class](#the-predictiondata-class)
  - [The 'ImageData' class](#the-imagedata-class)
  - [Image Dataset Creation](#image-dataset-creation)
  - [Displaying Data](#displaying-data)
- [Layers](#layers)
  - [Dense Layer](#dense-layer)
  - [Conv2D Layer](#conv2d-layer)
- [Models](#models)
  - [Training](#training)
- [GPU functionality](#gpu-functionality)

The framework uses Numpy, Matplotlib, Pillow, and some smaller libraries. To install these, run:
```
pip install -r requirements.txt
```

# Code Structure
It is important to note that the user's main script must have the following if-clause for certain Muskie features (such as creating Image Datasets) to work properly
```python
# imports and non-Muskie code here

if __name__ == "__main__":
  # your code here
```

# Utils
There are several miscellaneous functions that may be useful when using the framework. These can be found in the 'utils.py' file.
```python
from muskie.utils import *

one_hot = to_one_hot(np.array([1.0, -2.5, 5.0, 2.3])) # takes a numpy array
# one_hot is now equal to [0 0 1 0]
label = to_label(one_hot) # takes a one_hot_encoded array
# label is now 2 (the index of the 1 in the one_hot)
```
Additionally, there's the 'convolution_output_shape' function, which computes the output shape of a list of convolutional layers.

# Data Handling
## The 'Data' class
Muskie uses the Data class to store image datasets that can be used to visualize images, train models etc. It contains 3 numpy arrays: images, labels, and label_vectors. Users can add data to data, or arrays to the individual arrays that ImageData classes contain. Additionally, the function equals() determines whether two instances of Data are equal. An instance of Data can be created by
```python
from muskie.data import Data
data1 = Data(images=np.array([1,2,3]), create_labels=True) # labels is now an array of zeros of equal length as images. Label_vector is empty
data2 = Data(images=np.array([1,2,3]), create_labels=True)

data1.add(data2)

data1.print()
```
Which gives
```
Data:
Inputs: [1 2 3 1 2 3]
Labels: [0 0 0 0 0 0]
Label Vector: []
```
Additionally, with the batch() method images and labels can be split into batches
```python
data1.add(data2)
data1.add(data2)
images, labels = data1.batch(batch_size=2)  # images and labels contain two batches of two elements
```
## The 'PredictionData' class
This class does not take labels as a parameter. It instead takes an instance of the 'Model' class that will be used to generate the labels based on the inputs.
```python
from muskie.data import PredictionData

data = PredictionData(inputs=np.array([1,2,3]), model=model)
# data.labels will now be the result of model.forward(1), model.forward(2), and model.forward(3)
```
## The 'ImageData' class
The ImageData class is a subclass of Data, specifically meant for storing image datasets. It comes with functionality such as displaying the images which it contains,
and can be created by, for example, scraping folders of images.
## Image Dataset Creation
There are currently two ways of creating image datasets (apart from manually adding the image arrays to the ImageData constructor). One takes a folder that contains only images. The filenames can be used to generate labels. The other takes a folder with subdirectories that contain only images. The names of the subdirectories can be used to generate labels.
```python
from muskie.datasets import create_dataset, create_dataset_subdirectories

path1 = "images/fish_images"
path2 = "images/fish_images_subdirectories"
image_dimensions = (600, 500)

data1 = create_dataset(path1, dimensions=image_dimensions, create_labels=True, split="_") 
data2 = create_dataset_subdirectories(path2, dimensions=image_dimensions, create_labels=True) 
# Both data1 and data2 are now instances of ImageData
assert data1.equals(data2)
data1.print()
```
Which gives:
```
ImageData:
Images (shape): (9, 600, 500)
Labels: [0 0 0 1 1 1 2 2 2]
Label Vector: ('arapaima', 'marlin', 'musky')
```
## Displaying Data
ImageData can be displayed using the 'display_data' function
```python
data1.display_data(rows=3,cols=3)

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
## Dense Layer
A Dense layer can be created by specifying the input and output sizes
```python
from muskie.layers import *
layer = Dense(input_size=3, output_size=32)
res = layer.forward(np.array([1,2,3]))  # The last dimension of the input must be equal to the input_size
print(res.shape)
print(layer.toString())
```
Which gives
```
(32,)
Dense(3, 32)
```
When adding Dense layers to a nonempty model with the Model.add() function, the input_size argument can be ignored as it is replaced with the output_shape of the previous layer. A Conv2D layer cannot lead directly into a Dense layer, as the output of the Conv2D must be flattened.
## Conv2D Layer
A Conv2D layer can be created like so
```python
layer = Conv2D(nbr_kernels=128, kernel_size=3, padding=1, std=0.01, mean=0.0)
convolution = layer.calculate(images[0])
print(convolution.shape)
print(layer.toString())
```
Which gives
```
(600,500,128)
Dense(128, kernel_size=3, padding=1)
```

# Models
Models can be created with a list of layers, and layers can later be added as well.
```python
from muskie.models import ClassificationModel
from muskie.activation_functions import Tanh

layer1 = Conv2D(nbr_kernels=32, kernel_size=3, padding=1)
layer2 = Conv2D(nbr_kernels=64, kernel_size=3, padding=1)

model = ClassificationModel([layer1])
model.add(Tanh())  # Will apply the ReLU activation function on the output of layer1 and pass it on
model.add(layer2)
prediction = model.predict(images[0])

model.summary()

print(prediction.shape)
```
Alternatively
```python
model = ClassificationModel([
  layer1,
  Tanh(),
  layer2
])
# rest of code
```
which gives
```
ClassificationModel:
1. Conv2D(32, kernel_size=3, padding=1)
2. Tanh()
3. Conv2D(64, kernel_size=3, padding=1)

(600,500,64)
```

## Training
Models can be trained using the train() function. From the 'dense_model_example.py' file:
```python
import numpy as np
from muskie.models import ClassificationModel
from muskie.layers import *
from muskie.data import Data
from muskie.activation_functions import Tanh
from muskie.processing import train
from muskie.optimizers import SGD
from muskie.loss_functions import MSE

inputs = np.reshape([[0,0], [0,1], [1,0], [1,1]], (4,2,1))
labels = np.reshape([[0], [1], [1], [0]], (4,1,1))
data = Data(inputs, labels)

model  = ClassificationModel([
    Dense(input_size=2, output_size=3),
    Tanh(),
    Dense(1)
])

train(model=model, data=data, epochs=10000, optimizer=SGD(), loss=MSE())
print("BEFORE TRAINING:")
print(x1)
print(x2)
print(x3)
print(x4)
print("")
print("AFTER TRAINING:")
print(model.forward(np.reshape([0,0], (2,1))))
print(model.forward(np.reshape([0,1], (2,1))))
print(model.forward(np.reshape([1,0], (2,1))))
print(model.forward(np.reshape([1,1], (2,1))))
```
Which gives something similar to this: 
```
Epoch: 1, Loss: 3.0368828673900987
Epoch: 2, Loss: 1.084968175641837
...
Epoch: 9999, Loss: 4.930380657631324e-32
Epoch: 10000, Loss: 4.930380657631324e-32

BEFORE TRAINING:
[[-1.11497663]]
[[0.12581535]]
[[-1.01016975]]
[[0.24266287]]

AFTER TRAINING:
[[0.]]
[[1.]]
[[1.]]
[[0.]]
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
