from .layers import Conv2D, Layer
import numpy as np


def convolution_output_shape(image_shape: tuple, layers: list[Conv2D]) -> tuple:
    for layer in layers:
        assert isinstance(layer, Conv2D),"layers must be an iterable of Conv2D layers"

    x,y = image_shape[:2]
    for layer in layers:
        if isinstance(layer, Conv2D):
            x,y = x - layer.kernel_size + 2 * layer.padding + 1, y - layer.kernel_size + 2 * layer.padding + 1
    return (x, y, len(layers[-1].params["w"]))  # The last layer may be Dense or other, therefore not nbr_kernels