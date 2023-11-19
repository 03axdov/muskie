import numpy as np
from numba import jit, cuda
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings


def convolution_cpu(kernels, arr, padding, nbr: int = 0):
    result = 0
    for i in range(arr.shape[-1]):
        if len(arr.shape) == 3:
            arr = arr[:,:,i]
        kernelH, kernelW = kernels[nbr].shape
        arrH, arrW = arr.shape
        h, w = arrH + 1 - kernelH, arrW + 1 - kernelW

        filter1 = np.arange(kernelW) + np.arange(h)[:, np.newaxis]

        intermediate = arr[filter1]
        intermediate = np.transpose(intermediate, (0, 2, 1))

        filter2 = np.arange(kernelH) + np.arange(w)[:, np.newaxis]

        intermediate = intermediate[:, filter2]
        intermediate = np.transpose(intermediate, (0, 1, 3, 2))
        
        product = intermediate * kernels[nbr]
        convolved = product.sum(axis = (2,3))

        if padding:
            convolved = np.pad(convolved, (padding,), "constant", constant_values=(0,0))
        result += convolved

    return result


@jit(nopython=False)
def convolution_gpu(kernels, arr, padding, nbr: int = 0):
    result = 0
    for i in range(arr.shape[-1]):
        if len(arr.shape) == 3:
            arr = arr[:,:,i]
        kernelH, kernelW = kernels[nbr].shape
        arrH, arrW = arr.shape
        h, w = arrH + 1 - kernelH, arrW + 1 - kernelW

        filter1 = np.arange(kernelW) + np.arange(h)[:, np.newaxis]

        intermediate = arr[filter1]
        intermediate = np.transpose(intermediate, (0, 2, 1))

        filter2 = np.arange(kernelH) + np.arange(w)[:, np.newaxis]

        intermediate = intermediate[:, filter2]
        intermediate = np.transpose(intermediate, (0, 1, 3, 2))
        
        product = intermediate * kernels[nbr]
        convolved = product.sum(axis = (2,3))

        if padding:
            convolved = np.pad(convolved, (padding,), "constant", constant_values=(0,0))
        result += convolved

    return result