import numpy as np
from muskie.models import *
from muskie.layers import *
from muskie.data import *
from muskie.activation_functions import *
from muskie.processing import *
from muskie.optimizers import *
from muskie.loss_functions import *
from muskie.datasets import *


if __name__ == "__main__":
    data = create_image_dataset("images/fish_images", (100, 50))
    data.display_data(3,3)

    model  = ClassificationModel([
        Conv2D(3, kernel_size=3),
        Tanh(),
        PrintShape(),
        Flatten(),
        Dense(3),
        Tanh(),
        Dense(1)
    ])
    model.summary()

    x1 = model.forward(np.reshape([0,0], (2,1)))
    x2  = model.forward(np.reshape([0,1], (2,1)))
    x3 = model.forward(np.reshape([1,0], (2,1)))
    x4 = model.forward(np.reshape([1,1], (2,1)))

    train(model=model, data=data, epochs=10000, optimizer=SGD(lr=0.1), loss=MSE())

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
