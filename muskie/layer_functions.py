import numpy as np
from numba import jit, cuda
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

array_type = type(np.array([]))

def convolution_cpu(kernels: array_type, arr: array_type, padding: int, nbr: int = 0) -> array_type:
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
def pad_matrix_numba(matrix: array_type, padding: int, constant_values=0.0) -> array_type:
    original_shape = matrix.shape
    padded_shape = (original_shape[0] + 2 * padding, original_shape[1] + 2 * padding)
    
    padded_matrix = np.full(padded_shape, constant_values, dtype=matrix.dtype)
    
    for i in range(original_shape[0]):
        for j in range(original_shape[1]):
            padded_matrix[i + padding, j + padding] = matrix[i, j]

    return padded_matrix


@jit(nopython=True)
def convolution_gpu(kernels: array_type, a: array_type, padding: int, nbr: int = 0) -> array_type:
    result = np.zeros((a.shape[0] - kernels[nbr].shape[0] + 2*padding + 1, a.shape[1] - kernels[nbr].shape[0] + 2*padding + 1))  # Initialize result as a float

    for i in range(a.shape[-1]):
        if len(a.shape) == 3:
            arr = a[:, :, i]

        kernelH, kernelW = kernels[nbr].shape
        arrH, arrW = arr.shape
        h, w = arrH + 1 - kernelH, arrW + 1 - kernelW

        convolved_1 = np.zeros((h, w))  # Initialize convolved as zeros array

        for j in range(kernelW):
            for m in range(h):
                filter1_val = m + j
                for n in range(kernelH):
                    filter2_val = n + np.arange(w)
                    intermediate_val = arr[filter1_val, filter2_val]
                    convolved_1[m, :] += intermediate_val * kernels[i][n, j]

        if padding:
            convolved_2 = pad_matrix_numba(convolved_1, padding)
        else:
            convolved_2 = convolved_1

        result += convolved_2

    return result
