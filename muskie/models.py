from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def train(self, data):
        pass

    @abstractmethod
    def evaluate(self, data):
        pass

    @abstractmethod
    def predict(self, inputs):
        pass

    
class ClassificationModel(Model):
    def __init__(self):
        pass

    def train(self, data):
        pass

    def evaluate(self, data):
        pass
    
    def predict(self, inputs):
        pass