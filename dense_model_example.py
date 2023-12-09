import numpy as np
from muskie.models import *
from muskie.layers import *
from muskie.data import *
from muskie.activation_functions import *
from muskie.processing import *
from muskie.optimizers import *
from muskie.loss_functions import *

inputs = np.reshape([[0,0], [0,1], [1,0], [1,1]], (4,2,1))
labels = np.reshape([[1], [0], [0], [1]], (4,1,1))
data = Data(inputs, labels)

model  = ClassificationModel([
    Input(inputs.shape),
    Dense(3, activation=Tanh()),
    PrintShape(),
    Dense(1),
])
model.summary()

x1 = model.forward(np.reshape([0,0], (2,1)))
x2  = model.forward(np.reshape([0,1], (2,1)))
x3 = model.forward(np.reshape([1,0], (2,1)))
x4 = model.forward(np.reshape([1,1], (2,1)))

train(model=model, data=data, epochs=20000, optimizer=SGD(lr=0.1), loss=MSE())

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
