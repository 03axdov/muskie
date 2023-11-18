from .layer import Layer
import numpy as np

class Conv2D(Layer):
    def __init__(self, kernel_size: int, nbr_kernels: int = 1, padding:int = 0, std: float = 0.01, mean: float = 0.0):
        assert type(kernel_size) == int and kernel_size > 0,"kernel_size must be a positive integer"
        assert type(nbr_kernels) == int and nbr_kernels > 0,"nbr_kernels must be a positive integer"
        assert type(padding) == int and padding >= 0,"padding must be a positive integer"
        assert type(std) == float,"std must be a float"
        assert type(mean) == float,"mean must be a float"
        
        self.kernels = [std * np.random.randn(kernel_size, kernel_size) + mean for _ in range(nbr_kernels)]
        self.padding = padding


    def calculate(self, arr):
        if arr.shape[-1] == 1 or len(arr.shape) == 2:
            conv = self.convolution(arr, 0)
            for i in range(len(self.kernels) - 1):
                conv = np.concatenate((conv, self.convolution(arr, i + 1)))
            return conv
        else:
            if len(self.kernels) > 1:
                conv = np.dstack(tuple([sum([self.convolution(arr[:,:,t], i) for t in range(arr.shape[-1])]) for i in range(len(self.kernels))]))
            else:
                conv = sum([self.convolution(arr[:,:,t], 0) for t in range(arr.shape[-1])])
            return conv


    def convolution(self, arr, nbr: int = 0):
        kernelH, kernelW = self.kernels[nbr].shape
        arrH, arrW = arr.shape
        h, w = arrH + 1 - kernelH, arrW + 1 - kernelW

        filter1 = np.arange(kernelW) + np.arange(h)[:, np.newaxis]

        intermediate = arr[filter1]
        intermediate = np.transpose(intermediate, (0, 2, 1))

        filter2 = np.arange(kernelH) + np.arange(w)[:, np.newaxis]

        intermediate = intermediate[:, filter2]
        intermediate = np.transpose(intermediate, (0, 1, 3, 2))
        
        product = intermediate * self.kernels[nbr]
        convolved = product.sum(axis = (2,3))

        if self.padding:
            return np.pad(convolved, (self.padding,), "constant", constant_values=(0,0))

        return convolved
        