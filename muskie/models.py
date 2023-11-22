from abc import ABC, abstractmethod
from .layers import Layer
from .data import Data
import numpy as np
from alive_progress import alive_bar
from typing import Sequence


array_type = type(np.array([]))


class Model(ABC):

    @abstractmethod
    def add(self, layer: Layer):
        pass

    @abstractmethod
    def forward(self, inputs: array_type):
        pass

    @abstractmethod 
    def predict(self, inputs: Data):
        pass

    @abstractmethod
    def params_and_grads(self):
        pass

    @abstractmethod
    def print(self):
        pass


    
class ClassificationModel(Model):
    def __init__(self, layers: Sequence[Layer] = []):

        for layer in layers:
            assert isinstance(layer, Layer),"layers must be an iterable of Layer subclasses"
        self.layers = layers
        self.weights = []


    def add(self, layer: Layer) -> None:
        assert isinstance(layer, Layer),"layer must be a subclass of Layer"
        self.layers.append(layer)


    def forward(self, inputs: array_type) -> array_type:
        pass


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
        pass

    
    def summary(self) -> None:
        print("")
        print("ClassificationModel:")
        for t, layer in enumerate(self.layers):
            print(f"{t + 1}. {layer.toString()}")
        print("")