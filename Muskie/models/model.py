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