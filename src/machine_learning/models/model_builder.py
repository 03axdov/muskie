from .layers.layer import Layer
from .models import Model

class ModelBuilder():
    def __init__(self, layers: list = []):
        self.layers = layers

    def add_layer(layer: type(Layer)) -> None:
        self.layers.append(layer)

    def build() -> Model:
        pass