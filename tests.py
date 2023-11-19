import numpy as np
from muskie.data import process_image
from muskie.data import create_dataset
from muskie.data import display_data
from muskie.files import paths_from_directory
from muskie.files import labels_from_directory
from muskie.layers import Conv2D
import time
import unittest


class TestCases(unittest.TestCase):

    labels = [0,0,0,1,1,1,2,2,2]
    image_dimensions = (600, 500)
    path = "images/fish_images"
    pike_path = "images/fish_images/pike_1.jpg"
    nbr_images = 9

    def test_paths_from_directory(self):
        paths = paths_from_directory(self.path)
        assert paths[:self.nbr_images] == ['images/fish_images\\arapaima_1.jpg', 'images/fish_images\\arapaima_2.jpg',
                         'images/fish_images\\arapaima_3.jpg', 'images/fish_images\\marlin_1.jpg', 
                         'images/fish_images\\marlin_2.jpg', 'images/fish_images\\marlin_3.jpeg', 
                         'images/fish_images\\pike_1.jpg', 'images/fish_images\\pike_2.jpg', 'images/fish_images\\pike_3.jpg'],"paths_from_directory returns incorrect paths"

        assert paths_from_directory("made/up/directory", debug=True) == []


    def test_labels_from_directory(self):
        labels, label_vector = labels_from_directory("images/fish_images", split="_")

        assert labels[:self.nbr_images] == [0,0,0,1,1,1,2,2,2],"labels_from_directory returns incorrect labels"
        assert label_vector == ["arapaima", "marlin", "pike"],"labels_from_directory returns incorrect label_vector"
        assert labels_from_directory("made/up/directory", split="_", debug=True) == ([], [])


    def test_process_image(self):
        img = process_image(self.pike_path, dimensions=self.image_dimensions)

        assert img.shape == self.image_dimensions + (3,),"process_image gave an incorrect image shape"
        assert np.array_equiv(img[250][250], np.array([86, 84, 82])),"process_image gave an incorrect image"
        assert np.array_equiv(img, process_image(self.pike_path, list(self.image_dimensions)))
        assert process_image("made/up/path.jpg", self.image_dimensions, debug=True) == None
        assert process_image("images/fish_images", self.image_dimensions, debug=True) == None


    def test_create_dataset(self):
        images, labels, label_vector = create_dataset(self.path, split="_", dimensions=self.image_dimensions)
        paths = paths_from_directory(self.path)
        
        assert np.array_equiv(images[2], process_image(paths[2], dimensions=self.image_dimensions)),"create_dataset gave an incorrect image"
        assert labels == self.labels,"create_dataset gave incorrect labels"
        assert label_vector == ["arapaima", "marlin", "pike"]


    def test_display_data(self):
        data = create_dataset(self.path, dimensions=self.image_dimensions)
        display_data(data,3,3)

        image = process_image(self.pike_path, dimensions=self.image_dimensions)
        # display_data([image, [], []],3,3,many=False)


    def test_conv2d(self):
        image = process_image(self.pike_path, dimensions=self.image_dimensions)
        nbr_kernels = 1
        kernel_size = 4

        padding = 0
        output_size_x = (self.image_dimensions[0] - kernel_size + 2 * padding) + 1
        output_size_y = (self.image_dimensions[1] - kernel_size + 2 * padding) + 1

        layer = Conv2D(nbr_kernels, kernel_size=kernel_size)
        assert layer.kernels.shape == (nbr_kernels, kernel_size, kernel_size),"wrong shaped kernels"
        result = layer.calculate(image)
        assert result.shape == (output_size_x, output_size_y, nbr_kernels),"conv2d layer gives the wrong shape output"

        padding = 2
        output_size_x = (self.image_dimensions[0] - kernel_size + 2 * padding) + 1
        output_size_y = (self.image_dimensions[1] - kernel_size + 2 * padding) + 1

        layer = Conv2D(nbr_kernels, kernel_size=kernel_size, padding=padding)
        assert layer.kernels.shape == (nbr_kernels, kernel_size, kernel_size),"wrong shaped kernels with padding"
        result = layer.calculate(image)
        assert result.shape == (output_size_x, output_size_y, nbr_kernels),"conv2d layer with padding gives the wrong shape output"

        nbr_kernels = 3
        layer = Conv2D(nbr_kernels, kernel_size=kernel_size, padding=padding)
        assert layer.kernels.shape == (nbr_kernels, kernel_size, kernel_size),"wrong shaped kernels with many kernels"
        result = layer.calculate(image)
        assert result.shape == (output_size_x, output_size_y, nbr_kernels),"conv2d layer with kernel_size gives the wrong shape output"

        nbr_kernels_1 = 10
        nbr_kernels_2 = 5
        kernel_size_1 = 3
        kernel_size_2 = 5

        output_size_x = (self.image_dimensions[0] - kernel_size_1) + 1 - kernel_size_2 + 1
        output_size_y = (self.image_dimensions[1] - kernel_size_1) + 1 - kernel_size_2 + 1

        layer1 = Conv2D(nbr_kernels_1, kernel_size=kernel_size_1, gpu=True)
        layer2 = Conv2D(nbr_kernels_2, kernel_size=kernel_size_2, gpu=True)
        tic = time.time()
        result = layer2.calculate(layer1.calculate(image))
        toc = time.time()
        print(f"Time (ms): {int((toc - tic) * 1000)} ms")
        assert result.shape == (output_size_x, output_size_y, nbr_kernels_2)


if __name__ == "__main__":
    unittest.main()