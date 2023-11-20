from muskie.models import ClassificationModel
from muskie.utils import convolution_output_shape
from muskie.datasets import create_dataset
from muskie.layers import Conv2D
from muskie.core import use_gpu
from muskie.data import process_image
from muskie.utils import convolution_output_shape

import numpy as np
import time
import unittest


class TestCases(unittest.TestCase):
    pike_path = "images/fish_images/pike_1.jpg"
    path = "images/fish_images"
    image_dimensions = (150, 75)

    def test_classification_model(self):
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

        data = create_dataset(path=self.path, create_labels=True, dimensions=(10,5))    # Small dimensions for faster testing
        model = ClassificationModel([
            Conv2D(8, 3, padding=1),
            Conv2D(8, 3, padding=1)
        ])

        result = model.predict_many(data.images[0:3], verbose=False)

        assert result.shape[0] == 3,"predict_many gave wrong shape"



if __name__ == "__main__":

    use_gpu()

    unittest.main()