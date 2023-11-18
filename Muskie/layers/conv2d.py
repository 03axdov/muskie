from .layer import Layer
import numpy as np

class Conv2D(Layer):
    def __init__(self, kernel_size: int, std: float = 0.01, mean: float = 0):
        self.kernel = std * np.random.randn(kernel_size, kernel_size) + mean


    def calculate(self, arr):
        if arr.shape[-1] == 1 or len(arr.shape) == 2:
            return self.convolution(arr)
        else:
            return sum([self.convolution(arr[:,:,i]) for i in range(arr.shape[-1])])


    def convolution(self, arr):
        kernelH, kernelW = self.kernel.shape
        arrH, arrW = arr.shape
        h, w = arrH + 1 - kernelH, arrW + 1 - kernelW

        filter1 = np.arange(kernelW) + np.arange(h)[:, np.newaxis]
        
        intermediate = arr[filter1]
        
        intermediate = np.transpose(intermediate, (0, 2, 1))

        filter2 = np.arange(kernelH) + np.arange(w)[:, np.newaxis]
        
        intermediate = intermediate[:, filter2]
        
        intermediate = np.transpose(intermediate, (0, 1, 3, 2))
        
        product = intermediate * self.kernel
        
        convolved = product.sum(axis = (2,3))
        return convolved
        