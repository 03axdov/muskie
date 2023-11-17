import numpy as np
from PIL import Image

def process_image(path: str, dimensions: tuple[int, int]) -> type(np.array([])):
        img = Image.open(path)
        img = img.resize(dimensions)
        img = np.array(img)
        return img