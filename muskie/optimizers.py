from .models import Model
from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def step(self, model: Model) -> None: # Update params
        pass


class SGD(Optimizer):
    def __init__(self, lr: float = 0.001) -> None:   # learning rate = 0.001, lambd --> lambda for regularization
        self.lr = lr

    def step(self, model: Model) -> None:
        for param, grad in model.params_and_grads():
            param -= self.lr * grad


class Adam(Optimizer):
    def __init__(self, lr: float = 0.001) -> None:
        self.lr = lr

    def step(self, model: Model) -> None:
        pass