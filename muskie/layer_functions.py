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


@jit(nopython=True)  
def convolution_gpu(kernels, arr, padding, nbr: int = 0):
    result = 0
    for i in range(arr.shape[-1]):
        if len(arr.shape) == 3:
            new_arr = arr[:,:,i]
        kernelH, kernelW = kernels[nbr].shape
        arrH, arrW = new_arr.shape
        h, w = arrH + 1 - kernelH, arrW + 1 - kernelW

        filter1 = np.arange(kernelW) + np.arange(h)[:, np.newaxis]

        intermediate_0: list[int] = []

        for f in filter1:
            intermediate_0.append(arr[f])

        intermediate_1 = np.array(intermediate_0)

        intermediate_2 = np.transpose(intermediate_1, (0, 2, 1))

        filter2 = np.arange(kernelH) + np.arange(w)[:, np.newaxis]

        intermediate_3 = intermediate_2[:, filter2]
        intermediate_4 = np.transpose(intermediate_3, (0, 1, 3, 2))
        
        product = intermediate_4 * kernels[nbr]
        convolved = product.sum(axis = (2,3))

        if padding:
            convolved = np.pad(convolved, (padding,), "constant", constant_values=(0,0))
        result += convolved

    return result
    