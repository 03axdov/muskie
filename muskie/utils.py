from .layers import Conv2D, Layer
import numpy as np


array_type = type(np.array([]))


def convolution_output_shape(image_shape: list, layers: list[Layer]):
    for layer in layers:
        assert isinstance(layer, Layer),"layers must be an iterable of Layer subclasses"

    x,y = image_shape[:2]
    for layer in layers:
        if isinstance(layer, Conv2D):
            x,y = x - layer.kernel_size + 2 * layer.padding + 1, y - layer.kernel_size + 2 * layer.padding + 1
    return (x, y, len(layers[-1].params["w"]))  # The last layer may be Dense or other, therefore not nbr_kernels