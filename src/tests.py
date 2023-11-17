import numpy as np
from utils.process_image import process_image
from utils.create_dataset import create_dataset
import os


def test_process_image():
    img = process_image("images/Logo.png", (1224, 1020))

    assert img.shape == (1020, 1224, 4)
    assert np.array_equiv(img[500][500], np.array([250, 250, 248, 255]))


def test_create_dataset():
    labels_original = [1,1,1,2,2,2,3,3,3]
    image_dimensions = (500, 600)
    path = "images/fish_images"
    data = create_dataset(path, labels_original, image_dimensions=image_dimensions)
    images = data[0]
    labels = data[1]
    paths = list(map(lambda p: os.path.join(path, p), os.listdir(path)))
    
    assert np.array_equiv(images[2], process_image(paths[2], image_dimensions))
    assert labels == labels_original


if __name__ == "__main__":
    test_process_image()
    test_create_dataset()