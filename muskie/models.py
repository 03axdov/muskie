from abc import ABC, abstractmethod
from .layers import Layer
from .data import Data
import numpy as np
from alive_progress import alive_bar


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


    def predict(self, image: array_type, verbose: bool = True):
        assert type(image) == array_type
        assert type(verbose) == bool

        current = image
        if verbose:
            with alive_bar(len(self.layers)) as bar:
                for layer in self.layers:
                    current = layer.calculate(current)
                    bar()
        else:
            for layer in self.layers:
                current = layer.calculate(current)
        return current

    
    def predict_many(self, images: list, verbose : bool = True):
        if type(images) == list:
            images = np.array(images)
        assert type(images) == array_type
        assert type(verbose) == bool
        assert type(images[0]) == array_type
        
        current = np.array([self.predict(images[0], verbose)])
        for t,image in enumerate(images):
            if t != 0:
                prediction = self.predict(image, verbose)
                current = np.concatenate((current, np.array([prediction])))

        return current




    def train(self, data: Data):
        assert isinstance(data, Data),"an instance of Data must be passed to 'train'"
        images, labels, label_vector = data


    def evaluate(self, data: Data):
        assert isinstance(data, Data),"an instance of Data must be passed to 'evaluate'"
        images, labels, label_vector = data