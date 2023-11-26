import numpy as np
from muskie.models import ClassificationModel
from muskie.layers import *
from muskie.data import process_image, display_data, Data
from muskie.datasets import create_dataset
from muskie.core import use_gpu, gpu
from muskie.activation_functions import ReLU, Tanh
from muskie.processing import train
from muskie.optimizers import SGD
from muskie.loss_functions import MSE

import numpy as np
import time


inputs = np.reshape([[0,0], [0,1], [1,0], [1,1]], (4,2,1))
labels = np.reshape([[0], [1], [1], [0]], (4,1,1))
data = Data(inputs, labels)

model  = ClassificationModel([
    Dense(input_size=2, output_size=3),
    Tanh(),
    Dense(input_size=3, output_size=1),
])
model.summary()

x1 = model.forward(np.reshape([0,0], (2,1)))
x2  = model.forward(np.reshape([0,1], (2,1)))
x3 = model.forward(np.reshape([1,0], (2,1)))
x4 = model.forward(np.reshape([1,1], (2,1)))

train(model=model, data=data, epochs=5000, optimizer=SGD(lr=0.1), loss=MSE())

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
