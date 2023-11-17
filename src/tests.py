import numpy as np
from utils.process_image import process_image
from utils.create_dataset import create_dataset
from utils.display_data import display_data
import os
import unittest


class TestCases(unittest.TestCase):

    labels_original = [1,1,1,2,2,2,3,3,3]
    image_dimensions = (600, 500)
    path = "images/fish_images"

    def test_process_image(self):
        img = process_image("images/Logo.png", (1224, 1020))

        assert img.shape == (1020, 1224, 4),"process_image gave an incorrect image shape"
        assert np.array_equiv(img[500][500], np.array([250, 250, 248, 255])),"process_image does gave an incorrect image"


    def test_create_dataset(self):
        data = create_dataset(self.path, self.labels_original, image_dimensions=self.image_dimensions)
        images = data[0]
        labels = data[1]
        paths = list(map(lambda p: os.path.join(self.path, p), os.listdir(self.path)))
        
        assert np.array_equiv(images[2], process_image(paths[2], self.image_dimensions)),"create_dataset gave an incorrect image"
        assert labels == self.labels_original,"create_dataset gave incorrect labels"


    def test_display_data(self):
        data = create_dataset(self.path, self.labels_original, image_dimensions=self.image_dimensions)
        display_data(data,3,3,label_vector=["Arapaima", "Marlin", "Muskie"])


if __name__ == "__main__":
    unittest.main()