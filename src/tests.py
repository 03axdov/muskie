import numpy as np
from utils.process_image import process_image


def test_process_image():
    img = process_image("Logo.png")
    assert img.shape == (1020, 1224, 4)
    assert np.array_equiv(img[500][500], np.array([250, 250, 248, 255]))



if __name__ == "__main__":
    test_process_image()