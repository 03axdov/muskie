from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def calculate(self, inputs):
        pass