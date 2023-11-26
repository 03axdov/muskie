from .layer_functions import convolution_cpu, convolution_gpu
from .core import gpu

import numpy as np
import multiprocessing as mp
from abc import ABC, abstractmethod
import sys


array_type = type(np.array([]))


class Layer(ABC):
    def __init__(self):
        self.params: Dict[str, array_type] = {}
        self.grads: Dict[str, array_type] = {}

    @abstractmethod
    def forward(self, arr: array_type) -> array_type:
        pass

    @abstractmethod
    def backward(self, arr: array_type) -> array_type:
        pass

    @abstractmethod
    def toString(self) -> str:
        pass


class Dense(Layer):
    def __init__(self, output_size: int, input_size: int = 1, std: float = 1, mean = 0.0):
        # Inputs: (batch_size, input_size)
        # Outputs: (batch_size, output_size)
        super().__init__()
        self.params["w"] = std * np.random.randn(output_size, input_size) + mean  # Initialize weights
       
        self.params["b"] = np.random.randn(output_size, 1)   # Initialize biases
        self.input = np.array([])

        self.output_size = output_size
        self.input_size = input_size

    def forward(self, input: array_type) -> array_type:
        # assert input.shape[0] == self.params["w"].shape[1],"Last dimension of inputs must be equal to the input shape of Dense layer"

        if len(input.shape) == 1:
            input = np.reshape(input, (input.shape[0], 1))

        self.input = input
        return np.dot(self.params["w"], self.input) + self.params["b"]


    def backward(self, grad: array_type) -> array_type:
        self.grads["b"] = grad 
        self.grads["w"] = np.dot(grad, self.input.T)
        
        return np.dot(self.params["w"].T, grad)

    
    def toString(self) -> str:
        return f"Dense({self.params['w'].shape[1]}, {self.params['w'].shape[0]})"



class Conv2D(Layer):
    def __init__(self, nbr_kernels: int, kernel_size: int = 3, padding:int = 0, std: float = 0.01, mean: float = 0.0):
        assert type(kernel_size) == int and kernel_size > 0,"kernel_size must be a positive integer"
        assert type(nbr_kernels) == int and nbr_kernels > 0,"nbr_kernels must be a positive integer"
        assert type(padding) == int and padding >= 0,"padding must be a positive integer"
        assert type(std) == float,"std must be a float"
        assert type(mean) == float,"mean must be a float"
        
        super().__init__()
        self.params["w"] = np.array([std * np.random.randn(kernel_size, kernel_size) + mean for _ in range(nbr_kernels)])
        self.kernel_size = kernel_size  # For computing the shape of outputs, etc.
        self.nbr_kernels = nbr_kernels  # For computing the shape of outputs, etc.
        self.padding = padding

        self.inputs = np.array([])
        self.c = 0


    def forward(self, inputs: array_type) -> array_type:
        assert type(inputs) == array_type
        self.inputs = inputs # Cache a[l-1]
        if not gpu():
            pool = mp.Pool(mp.cpu_count())
            processes = [pool.apply_async(convolution_cpu, args=(self.params["w"],inputs,self.padding,t)) for t in range(self.nbr_kernels)]
            convolutions = [p.get() for p in processes]
            pool.close()
        else:
            convolutions = np.array([convolution_gpu(self.params["w"], a=inputs, padding=self.padding, nbr=i) for i in range(self.nbr_kernels)])

        return np.dstack(tuple(convolutions))


    def backward(self, grads: array_type) -> array_type:
        self.c += 1
        if c == 2:
            sys.exit()
        pass


    def toString(self) -> str:
        if self.padding == 0:
            return f"Conv2D({self.nbr_kernels}, kernel_size={self.kernel_size})"
        else:
            return f"Conv2D({self.nbr_kernels}, kernel_size={self.kernel_size}, padding={self.padding})"


class Flatten(Layer):

    def __init__(self):
        self.inputs = np.array([])

    def forward(self, arr: array_type) -> array_type:
        self.inputs = inputs
        return arr.flatten()

    def backward(self, arr: array_type) -> array_type:
        return self.inputs

    def toString(self) -> str:
        return "Flatten()"
