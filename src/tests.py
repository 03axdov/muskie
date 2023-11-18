import numpy as np
from utils.process_image import process_image
from utils.create_dataset import create_dataset
from utils.display_data import display_data
from utils.paths_from_directory import paths_from_directory
from utils.labels_from_directory import labels_from_directory

import unittest


class TestCases(unittest.TestCase):

    labels_original = [0,0,0,1,1,1,2,2,2]
    image_dimensions = (600, 500)
    path = "images/fish_images"
    nbr_images = 9

    def test_paths_from_directory(self):
        paths = paths_from_directory(self.path)
        assert paths[:self.nbr_images] == ['images/fish_images\\arapaima_1.jpg', 'images/fish_images\\arapaima_2.jpg',
                         'images/fish_images\\arapaima_3.jpg', 'images/fish_images\\marlin_1.jpg', 
                         'images/fish_images\\marlin_2.jpg', 'images/fish_images\\marlin_3.jpeg', 
                         'images/fish_images\\pike_1.jpg', 'images/fish_images\\pike_2.jpg', 'images/fish_images\\pike_3.jpg'],"paths_from_directory returns incorrect paths"


    def test_labels_from_directory(self):
        labels, label_vector = labels_from_directory("images/fish_images", split="_")

        assert labels[:self.nbr_images] == [0,0,0,1,1,1,2,2,2],"labels_from_directory returns incorrect labels"
        assert label_vector == ["arapaima", "marlin", "pike"],"labels_from_directory returns incorrect label_vector"


    def test_process_image(self):
        img = process_image("images/Logo.png", (1224, 1020))

        assert img.shape == (1020, 1224, 4),"process_image gave an incorrect image shape"
        assert np.array_equiv(img[500][500], np.array([250, 250, 248, 255])),"process_image does gave an incorrect image"


    def test_create_dataset(self):
        images, labels = create_dataset(self.path, self.labels_original, image_dimensions=self.image_dimensions)
        paths = paths_from_directory(self.path)
        
        assert np.array_equiv(images[2], process_image(paths[2], self.image_dimensions)),"create_dataset gave an incorrect image"
        assert labels == self.labels_original,"create_dataset gave incorrect labels"


    def test_display_data(self):
        data = create_dataset(self.path, self.labels_original, image_dimensions=self.image_dimensions)
        # display_data(data,3,3,label_vector=["Arapaima", "Marlin", "Muskie"])


if __name__ == "__main__":
    unittest.main()