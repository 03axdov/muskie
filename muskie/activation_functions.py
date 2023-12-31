import numpy as np

array_type = type(np.array([]))

class Activation():
    def __init__(self, activation, activation_prime, activation_name: str):
        self.activation = activation
        self.activation_prime = activation_prime
        self.activation_name = activation_name
        

    def forward(self, input: array_type) -> array_type:
        self.input = input

        return self.activation(self.input)

    def backward(self, grad: array_type) -> array_type:
        return np.multiply(grad, self.activation_prime(self.input))

    def toString(self) -> str:
        return self.activation_name 


class ReLU(Activation):
    def __init__(self):
        relu = lambda x: np.maximum(0, x)
        relu_prime = lambda x: (x > 0) * 1
        name = "ReLU"
        super().__init__(relu, relu_prime, name)


class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x)**2
        name = "Tanh"
        super().__init__(tanh, tanh_prime, name)