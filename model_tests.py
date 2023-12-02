from muskie.models import *
from muskie.utils import *
from muskie.datasets import *
from muskie.layers import *
from muskie.core import *
from muskie.data import *
from muskie.utils import *
from muskie.processing import *
from muskie.optimizers import *
from muskie.loss_functions import *
from muskie.activation_functions import *

import numpy as np
import time
import unittest


class TestCases(unittest.TestCase):
    pike_path = "images/fish_images/pike_1.jpg"
    path = "images/fish_images"
    image_dimensions = (150, 75)

    def test_dense_model(self):
        inputs = np.reshape([[0,0], [0,1], [1,0], [1,1]], (4,2,1))
        labels = np.reshape([[1], [0], [0], [1]], (4,1,1))
        data = Data(inputs, labels)

        model  = ClassificationModel([
            Dense(input_size=2, output_size=3),
            Tanh(),
            Dense(input_size=3, output_size=1),
        ])
        train(model=model, data=data, epochs=5000, optimizer=SGD(lr=0.1), loss=MSE())

        data = PredictionData(inputs=inputs,
                              model=model)
        data_labels = data.labels
        print(f"data_labels: {data_labels}")
        difference = labels - data_labels


    def test_convolutional_model(self):
        kernel_size = 3
        nbr_kernels = 32
        padding = 0

        model1 = ClassificationModel()
        layer = Conv2D(nbr_kernels, kernel_size=kernel_size, padding=padding)
        model1.add(layer)
        assert model1.layers[0] == layer,"Adding layer not working"

        model2 = ClassificationModel([layer])
        image = process_image(self.pike_path, dimensions=self.image_dimensions)

        result = model2.predict(image, verbose=False)
        assert result.shape == convolution_output_shape(image.shape, [layer]),"predict gave the wrong output shape"

        data = create_image_dataset(path=self.path, create_labels=True, dimensions=(10,5))    # Small dimensions for faster testing
        model = ClassificationModel([
            Conv2D(8, 3, padding=1),
            Conv2D(8, 3, padding=1)
        ])



if __name__ == "__main__":

    use_gpu()

    unittest.main()