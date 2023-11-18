from ..layers.layer import Layer
from .classification_model import ClassificationModel

class ClassificationModelBuilder():
    def __init__(self, layers: list = []):
        self.layers = layers

    def add_layer(layer: type(Layer)) -> None:
        self.layers.append(layer)

    def build() -> ClassificationModel:
        pass

