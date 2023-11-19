from .layer_functions import convolution_cpu, convolution_gpu
import numpy as np
import multiprocessing as mp


class Conv2D():
    def __init__(self, nbr_kernels: int, kernel_size: int = 3, padding:int = 0, std: float = 0.01, mean: float = 0.0, gpu: bool = False):
        assert type(kernel_size) == int and kernel_size > 0,"kernel_size must be a positive integer"
        assert type(nbr_kernels) == int and nbr_kernels > 0,"nbr_kernels must be a positive integer"
        assert type(padding) == int and padding >= 0,"padding must be a positive integer"
        assert type(std) == float,"std must be a float"
        assert type(mean) == float,"mean must be a float"
        
        self.kernels = np.array([std * np.random.randn(kernel_size, kernel_size) + mean for _ in range(nbr_kernels)])
        self.padding = padding
        self.gpu = gpu


    def calculate(self, arr):
        if not self.gpu:
            pool = mp.Pool(mp.cpu_count())
            processes = [pool.apply_async(convolution_cpu, args=(self.kernels,arr,self.padding,t)) for t in range(len(self.kernels))]
            convolutions = [p.get() for p in processes]
            pool.close()
        else:
            convolutions = [convolution_gpu(self.kernels, a=arr, padding=self.padding, nbr=i) for i in range(len(self.kernels))]

        return np.dstack(tuple(convolutions))