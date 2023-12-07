from .layer_functions import convolution_cpu, convolution_gpu
from .core import gpu
from .activation_functions import Activation

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


class WeightedLayer(Layer, ABC):
    pass


class Input(Layer):
    pass


class Dense(WeightedLayer):
    def __init__(self, output_size: int, input_size: int = 1, activation: Activation = None, std: float = 1, mean = 0.0):
        # Inputs: (batch_size, input_size)
        # Outputs: (batch_size, output_size)
        super().__init__()
        self.params["w"] = std * np.random.randn(output_size, input_size) + mean  # Initialize weights
       
        self.params["b"] = np.random.randn(output_size, 1)   # Initialize biases
        self.input = np.array([])

        self.output_size = output_size
        self.input_size = input_size
        if activation:
            assert isinstance(activation, Activation), "activation must be an instance of the Activation abstract class"
        self.activation = activation

    def forward(self, input: array_type) -> array_type:
        # assert input.shape[0] == self.params["w"].shape[1],"Last dimension of inputs must be equal to the input shape of Dense layer"

        if len(input.shape) == 1:
            input = np.reshape(input, (input.shape[0], 1))

        self.input = input
        dot_product = np.dot(self.params["w"], self.input) + self.params["b"]

        if self.activation:
            dot_product = self.activation.forward(dot_product)

        return dot_product


    def backward(self, grad: array_type) -> array_type:

        if self.activation:
            grad = self.activation.backward(grad)

        self.grads["b"] = grad 
        self.grads["w"] = np.dot(grad, self.input.T)
        
        return np.dot(self.params["w"].T, grad)

    
    def toString(self) -> str:
        result = f"Dense({self.params['w'].shape[1]}, {self.params['w'].shape[0]}"
        if self.activation:
            result += ", activation=" + self.activation.toString()
        return result + ")"



class Conv2D(WeightedLayer):
    def __init__(self, nbr_kernels: int, kernel_size: int = 3, padding: int = 0, std: float = 0.01, mean: float = 0.0):
        assert type(kernel_size) == int and kernel_size > 0,"kernel_size must be a positive integer"
        assert type(nbr_kernels) == int and nbr_kernels > 0,"nbr_kernels must be a positive integer"
        assert type(padding) == int and padding >= 0,"padding must be a positive integer"
        assert type(std) == float,"std must be a float"
        assert type(mean) == float,"mean must be a float"
        
        super().__init__()
        self.params["w"] = np.array([std * np.random.randn(kernel_size, kernel_size) + mean for _ in range(nbr_kernels)])
        self.params["b"] = np.array([])
        self.kernel_size = kernel_size  # For computing the shape of outputs, etc.
        self.nbr_kernels = nbr_kernels  # For computing the shape of outputs, etc.
        self.padding = padding

        self.input = np.array([])
        self.c = 0


    def forward(self, input: array_type) -> array_type:
        assert type(input) == array_type
        self.input = input # Cache a[l-1]
        if not gpu():
            pool = mp.Pool(mp.cpu_count())
            processes = [pool.apply_async(convolution_cpu, args=(self.params["w"],input,self.padding,t)) for t in range(self.nbr_kernels)]
            convolutions = np.array([p.get() for p in processes])
            pool.close()
        else:
            convolutions = np.array([convolution_gpu(self.params["w"], a=input, padding=self.padding, nbr=i) for i in range(self.nbr_kernels)])

        if self.params["b"].size == 0:
            self.params["b"] = np.random.randn(convolutions.shape[0], convolutions.shape[1], convolutions.shape[2])

        return np.dstack(tuple(convolutions)) + np.dstack(tuple(self.params["b"]))  # May be slow


    def backward(self, grad: array_type) -> array_type:
        self.params["w"] = np.zeros(self.params["w"].shape)
        input_gradient = np.zeros(self.input.shape)

        for i in range(self.nbr_kernels):
            if not gpu():
                kernels_gradient[i, j] = convolution_cpu(self.input, grad[i])
                input_gradient[j] += convolution_cpu(grad[i], self.kernels[i])
            else:
                pass

        self.params["b"] = input_gradient
        return input_gradient


    def toString(self) -> str:
        result = f"Conv2D({self.nbr_kernels}, kernel_size={self.kernel_size}"
        if self.padding:
            result += f", padding={self.padding}"
        return result + ")"


class Flatten(Layer):

    def __init__(self):
        self.inputs = np.array([])

    def forward(self, inputs: array_type) -> array_type:
        self.inputs = inputs
        return inputs.flatten()

    def backward(self, grad: array_type) -> array_type:
        return self.inputs

    def toString(self) -> str:
        return "Flatten()"


class PrintShape(Layer):
    def __init__(self):
        self.inputs = np.array([])

    def forward(self, inputs: array_type) -> array_type:
        self.inputs = inputs
        print(f"PrintShape: {inputs.shape}")
        print("")
        return inputs

    def backward(self, grad: array_type) -> array_type:
        return self.inputs

    def toString(self) -> str:
        return "PrintShape()"