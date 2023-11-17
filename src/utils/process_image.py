import numpy as np
from PIL import Image

def process_image(path: str, dimensions: tuple[int, int]):
        img = Image.open(path)
        img = img.resize(dimensions)
        img.show()
        img = np.array(img)
        return img