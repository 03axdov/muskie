from abc import ABC, abstractmethod
from .layers import Layer
from .data import Data

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
    def __init__(self, layers: list[Layer] = [], gpu: bool = False):
        assert type(layers) == list,"layers must be a list"
        assert type(gpu) == bool,"gpu must be a boolean"
        for layer in layers:
            assert isinstance(layer, Layer),"layers must be a list of Layer subclasses"

        self.layers = layers
        self.gpu = gpu

    def add(self, layer: Layer):
        assert isinstance(layer, Layer),"layer must be a subclass of Layer"
        layer.gpu = self.gpu
        self.layers.append(layer)


    def predict(self, data: Data):
        images, labels, label_vector = data


    def train(self, data: Data):
        images, labels, label_vector = data


    def evaluate(self, data: Data):
        images, labels, label_vector = data