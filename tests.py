from muskie.data import display_data, process_image, Data
from muskie.datasets import create_dataset, create_dataset_subdirectories
from muskie.system import paths_from_directory, labels_from_directory
from muskie.layers import Conv2D, Dense
from muskie.models import ClassificationModel
from muskie.utils import convolution_output_shape
from muskie.core import use_gpu
from muskie.activation_functions import relu

import numpy as np
import time
import unittest


class TestCases(unittest.TestCase):

    labels = [0,0,0,1,1,1,2,2,2]
    image_dimensions = (150, 75)
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
        assert np.array_equiv(img, process_image(self.pike_path, list(self.image_dimensions)))
        assert process_image("made/up/path.jpg", self.image_dimensions, debug=True) == None
        assert process_image("images/fish_images", self.image_dimensions, debug=True) == None


    def test_data(self):
        example_array = np.array([[1,2,3]])
        data1 = Data(images=example_array, create_labels=True)
        data2 = Data(images=example_array, create_labels=True)

        data1.add(data2)
        assert np.array_equiv(data1.images, np.array([[1,2,3],[1,2,3]])),"Data add() not working properly"
        assert data1.equals(data2),"Data equals not working"
        
        data1.add(data2)
        data1.add(data2)
        data1.add(data2)
        data1.add(data2)
        images, labels = data1.batch(batch_size=2)
        assert images.shape[0] == 3,"Data.batch() gives the wrong output shape"
        assert labels.shape[0] == 3,"Data.batch() gives the wrong output shape"


    def test_create_dataset(self):
        images, labels, label_vector = create_dataset(self.path, split="_", dimensions=self.image_dimensions).as_tuple()
        paths = paths_from_directory(self.path)
        
        assert np.array_equiv(images[2], process_image(paths[2], dimensions=self.image_dimensions)),"create_dataset gave an incorrect image"
        assert np.array_equiv(labels, np.array(self.labels)),"create_dataset gave incorrect labels"
        assert np.array_equiv(label_vector, np.array(["arapaima", "marlin", "pike"])),"create_dataset gave an incorrect label_vector"


    def test_create_dataset_subdirectories(self):
        images, labels, label_vector = create_dataset_subdirectories("images/fish_images_subdirectories", dimensions=self.image_dimensions).as_tuple()
        paths = paths_from_directory(self.path)

        assert np.array_equiv(images[2], process_image(paths[2], dimensions=self.image_dimensions)),"create_dataset_subdirectories gave an incorrect image"
        assert np.array_equiv(labels, np.array(self.labels)),"create_dataset_subdirectories gave incorrect labels"
        assert np.array_equiv(label_vector, np.array(["arapaima", "marlin", "pike"])),"create_dataset_subdirectories gave an incorrect label_vector"


    def test_display_data(self):
        data = create_dataset(self.path, dimensions=self.image_dimensions)
        display_data(data,3,3)

        image = process_image(self.pike_path, dimensions=self.image_dimensions)
        # display_data([image, [], []],3,3,many=False)


    def test_convolution_output_shape(self):    
        nbr_kernels = 128
        kernel_size = 3
        padding = 0

        layer1 = Conv2D(nbr_kernels, kernel_size=kernel_size, padding=padding)

        shape = convolution_output_shape(self.image_dimensions, [layer1])
        assert shape == (self.image_dimensions[0] - 2, self.image_dimensions[1] - 2, nbr_kernels)


    def test_dense(self):
        layer = Dense(input_size=3, output_size=32)
        arr = np.array([1,2,3])
        res = layer.forward(arr)
        assert res.shape == (32,),"dense layer gave an incorrect output shape"


    def test_conv2d(self):
        image = process_image(self.pike_path, dimensions=self.image_dimensions)
        nbr_kernels = 1
        kernel_size = 4

        padding = 0

        layer = Conv2D(nbr_kernels, kernel_size=kernel_size)
        assert layer.params["w"].shape == (nbr_kernels, kernel_size, kernel_size),"wrong shaped kernels"
        result = layer.forward(image)
        assert result.shape == convolution_output_shape(self.image_dimensions, [layer]),"conv2d layer gives the wrong shape output"
        assert not np.array_equiv(result, np.zeros(convolution_output_shape(self.image_dimensions, [layer]))),"conv2d layer gives a matrix of only zeros as output"

        padding = 2

        layer = Conv2D(nbr_kernels, kernel_size=kernel_size, padding=padding)
        assert layer.params["w"].shape == (nbr_kernels, kernel_size, kernel_size),"wrong shaped kernels with padding"
        result = layer.forward(image)
        assert result.shape == convolution_output_shape(self.image_dimensions, [layer]),"conv2d layer with padding gives the wrong shape output"
        assert not np.array_equiv(result, np.zeros(convolution_output_shape(self.image_dimensions, [layer]))),"conv2d layer with padding gives a matrix of only zeros as output"

        nbr_kernels = 3
        layer = Conv2D(nbr_kernels, kernel_size=kernel_size, padding=padding)
        assert layer.params["w"].shape == (nbr_kernels, kernel_size, kernel_size),"wrong shaped kernels with many kernels"
        result = layer.forward(image)
        assert result.shape == convolution_output_shape(self.image_dimensions, [layer]),"conv2d layer with kernel_size gives the wrong shape output"
        assert not np.array_equiv(result, np.zeros(convolution_output_shape(self.image_dimensions, [layer]))),"conv2d layer with kernel_size gives a matrix of only zeros as output"

        nbr_kernels_1 = 10
        nbr_kernels_2 = 5
        kernel_size_1 = 3
        kernel_size_2 = 5
        padding_1 = 2
        padding_2 = 2

        layer1 = Conv2D(nbr_kernels_1, kernel_size=kernel_size_1, padding=padding_1)
        layer2 = Conv2D(nbr_kernels_2, kernel_size=kernel_size_2, padding=padding_2)

        tic = time.time()
        result = layer2.forward(layer1.forward(image))
        toc = time.time()
        print("")
        print(f"Time: {int((toc - tic) * 1000)} ms")

        assert result.shape == convolution_output_shape(self.image_dimensions, [layer1, layer2]),"conv2d with gpu gives the wrong shape"
        assert not np.array_equiv(result, np.zeros(convolution_output_shape(self.image_dimensions, [layer1, layer2]))),"conv2d on gpu gives a matrix of only zeros as output"


    def test_relu(self):
        matrix = np.array([[1,-1,-10], [0,1,-9], [10, 2, -6]])
        matrix = relu(matrix)
        assert np.amin(matrix) == 0,"relu not working properly"
        

if __name__ == "__main__":

    use_gpu()

    unittest.main()