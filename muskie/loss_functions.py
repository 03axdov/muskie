import numpy as np
from abc import ABC, abstractmethod


array_type = type(np.array([]))


class Loss(ABC): # Effectively Cost Functions as they apply to batches
    @abstractmethod
    def loss(self, predicted: array_type, actual: array_type) -> float: # Calculate loss
        pass

    @abstractmethod
    def grad(self, predicted: array_type, actual: array_type) -> array_type: # Get da[l] from y^ - The derivative of y^ = da[l]
        pass


class MSE(Loss):

    def loss(self, predicted: array_type, actual: array_type) -> float:
        return np.mean((predicted - actual) ** 2)

    def grad(self, predicted: array_type, actual: array_type) -> array_type:
        return 2 * (predicted - actual) / np.size(actual)


class CategoricalCrossentropy(Loss):

    def loss(self, predicted: array_type, actual: array_type) -> float:
        pass

    def grad(self, predicted: array_type, actual: array_type, weights: array_type) -> array_type:
        pass


class BinaryCrossentropy(Loss):

    def loss(self, predicted: array_type, actual: array_type) -> float:
        actual = np.expand_dims(actual, axis=1) # Required else cost will be of shape (batch_size, batch_size)
        return -predicted * np.log(actual) - (1 - predicted) * np.log(1-actual)

    def grad(self, predicted: array_type, actual: array_type, weights: array_type) -> array_type:
        actual = np.expand_dims(actual, axis=1) # Required else cost will be of shape (batch_size, batch_size)
        return (-predicted/actual) + (1-predicted) / (1-actual)


class Logistic(Loss):

    def loss(self, predicted: array_type, actual: array_type) -> float:
        actual = np.expand_dims(actual, axis=1) # Required else cost will be of shape (batch_size, batch_size)
        return -predicted * np.log(actual) - (1 - predicted) * np.log(1-actual)

    def grad(self, predicted: array_type, actual: array_type, weights: array_type) -> array_type:
        actual = np.expand_dims(actual, axis=1) # Required else cost will be of shape (batch_size, batch_size)
        return (-predicted/actual) + (1-predicted) / (1-actual)