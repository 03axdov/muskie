from abc import ABC, abstractmethod
from .layers import Layer
from .data import Data
import numpy as np


array_type = type(np.array([]))


class Model(ABC):

    @abstractmethod
    def add(self, layer: Layer):
        pass

    @abstractmethod
    def predict(self, inputs: Data):
        pass

    @abstractmethod
    def train(self, data: Data):
        pass

    @abstractmethod
    def evaluate(self, data: Data):
        pass

    
class ClassificationModel(Model):
    def __init__(self, layers: list[Layer] = []):

        for layer in layers:
            assert isinstance(layer, Layer),"layers must be an iterable of Layer subclasses"
        self.layers = layers


    def add(self, layer: Layer):
        assert isinstance(layer, Layer),"layer must be a subclass of Layer"
        self.layers.append(layer)


    def predict(self, inputs: array_type):
        if type(inputs) == list:
            inputs = np.array(inputs)
        assert type(inputs) == array_type

        current = inputs
        for layer in self.layers:
            current = layer.calculate(current)
        return current



    def train(self, data: Data):
        assert isinstance(data, Data),"an instance of Data must be passed to 'train'"
        images, labels, label_vector = data


    def evaluate(self, data: Data):
        assert isinstance(data, Data),"an instance of Data must be passed to 'evaluate'"
        images, labels, label_vector = data