from abc import ABC, abstractmethod
from .layers import Layer, Dense, Flatten
from .data import DataAbstract
import numpy as np
from alive_progress import alive_bar
from typing import Sequence
from .activation_functions import Activation


array_type = type(np.array([]))


class Model(ABC):

    @abstractmethod
    def add(self, layer: Layer):
        pass

    @abstractmethod
    def forward(self, inputs: array_type) -> array_type:
        pass

    @abstractmethod
    def backward(self, inputs: array_type) -> array_type:
        pass

    @abstractmethod 
    def predict(self, inputs: DataAbstract):
        pass

    @abstractmethod
    def params_and_grads(self):
        pass

    @abstractmethod
    def summary(self):
        pass


    
class ClassificationModel(Model):
    def __init__(self, layers: Sequence[Layer] = []):

        self.layers = np.array([])
        self.prev_output_size = 0
        for layer in layers:
            assert isinstance(layer, Layer),"layers must be an iterable of Layer subclasses"
            self.add(layer)
        self.weights = np.array([])


    def add(self, layer: Layer) -> None:
        assert isinstance(layer, Layer),"layer must be a subclass of Layer"

        if isinstance(layer, Dense) and self.prev_output_size > 0:
            new_layer = Dense(output_size=layer.output_size,input_size=self.prev_output_size)
            self.layers = np.append(self.layers, new_layer)
        else:
            self.layers = np.append(self.layers, layer)

        if isinstance(layer, Dense):
            self.prev_output_size = layer.output_size


    def forward(self, inputs: array_type) -> array_type:
        self.weights = []    # As to prevent large matrixes between epochs'
        for layer in self.layers:
            inputs = layer.forward(inputs)
            if not isinstance(layer, Activation) and not isinstance(layer, Flatten):    # W - The weight matrix will be used by the loss function for regularization
                self.weights = np.append(self.weights, layer.params['w'])

        return inputs


    def backward(self, grad: array_type) -> array_type:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad


    def predict(self, image: array_type, verbose: bool = True) -> array_type:
        assert type(image) == array_type,"image must be a numpy array"
        assert type(verbose) == bool,"verbose must be a boolean"

        current = image
        if verbose:
            with alive_bar(len(self.layers)) as bar:
                for layer in self.layers:
                    current = layer.forward(current)
                    bar()
        else:
            for layer in self.layers:
                current = layer.forward(current)
        return current
        

    def params_and_grads(self):
        for layer in self.layers:
            if not isinstance(layer, Activation) and not isinstance(layer, Flatten):
                for name, param in layer.params.items():

                    grad = layer.grads[name]
                    yield param, grad

    
    def summary(self) -> None:
        print("")
        print("ClassificationModel:")
        for t, layer in enumerate(self.layers):
            print(f"{t + 1}. {layer.toString()}")
        print("")