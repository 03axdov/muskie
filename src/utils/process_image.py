import numpy as np
from PIL import Image

def process_image(path: str):
        img = Image.open(path)
        img = np.array(img)
        return img


if __name__ == "__main__":
    process_image("Logo.png")