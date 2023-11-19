import numpy as np
from abc import ABC, abstractmethod
import multiprocessing as mp


class Layer(ABC):
    @abstractmethod
    def calculate(self, inputs):
        pass
    

class Conv2D(Layer):
    def __init__(self, nbr_kernels: int, kernel_size: int = 3, padding:int = 0, std: float = 0.01, mean: float = 0.0):
        assert type(kernel_size) == int and kernel_size > 0,"kernel_size must be a positive integer"
        assert type(nbr_kernels) == int and nbr_kernels > 0,"nbr_kernels must be a positive integer"
        assert type(padding) == int and padding >= 0,"padding must be a positive integer"
        assert type(std) == float,"std must be a float"
        assert type(mean) == float,"mean must be a float"
        
        self.kernels = np.array([std * np.random.randn(kernel_size, kernel_size) + mean for _ in range(nbr_kernels)])
        self.padding = padding


    def calculate(self, arr):
        pool = mp.Pool(mp.cpu_count())
        processes = [pool.apply_async(self.convolution, args=(arr,t)) for t in range(len(self.kernels))]
        convolutions = [p.get() for p in processes]
        pool.close()
        return np.dstack(tuple(convolutions))
        # conv = np.dstack(tuple([sum([self.convolution(arr[:,:,t], i) for t in range(arr.shape[-1])]) for i in range(len(self.kernels))]))
        # return conv


    def convolution(self, arr, nbr: int = 0):
        result = 0
        for i in range(arr.shape[-1]):
            if len(arr.shape) == 3:
                arr = arr[:,:,i]
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
                convolved = np.pad(convolved, (self.padding,), "constant", constant_values=(0,0))
            result += convolved

        return result
        