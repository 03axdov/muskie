import numpy as np
from PIL import Image
from typing import Iterable
import numpy as np
import os
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


array_type = type(np.array([]))


class Data(ABC):
    @abstractmethod
    def add(self, other) -> None:
        pass

    @abstractmethod
    def equals(self, other) -> bool:
        pass

    @abstractmethod
    def print(self) -> None:
        pass


class ImageData(Data):
    def __init__(self, images: array_type = np.array([]), 
                 labels: array_type = np.array([]), 
                 label_vector: array_type = np.array([]),
                 create_labels: bool = False,
                 default_label: int = 0):
        if type(images) == list:
            images = np.array(images)
        if type(labels) == list:
            labels = np.array(labels)
        if type(label_vector) == list:
            label_vector = np.array(label_vector)
        
        assert type(create_labels) == bool,"create_labels must be a boolean"
        assert type(default_label) == int
        if create_labels:
            labels = np.full((images.shape[0]), default_label)
        assert images.shape[0] == labels.shape[0],"images and labels must have the same first dimension"

        if label_vector.size > 0:
            assert label_vector.size > np.amax(labels),"label vector does not contain all possible labels"


        self.images = images
        self.labels = labels
        self.label_vector = label_vector
    
    def as_tuple(self) -> tuple:
        return (self.images, self.labels, self.label_vector)


    def add_images_labels(self, images: array_type, labels: array_type) -> None:
        if type(images) == list or type(images) == tuple:
            images = np.array(images)
        if type(labels) == list or type(labels) == tuple:
            labels = np.array(labels)
        assert type(images) == array_type,"images was not a numpy array, a tuple, or a list"
        assert type(labels) == array_type,"labels was not a numpy array, a tuple, or a list"
        assert images.shape[0] == labels.shape[0],"images and labels must have the same first dimension"


        self.images = np.concatenate((self.images, images)) 
        self.labels = np.concatenate((self.labels, labels))
        

    def add_label_vector(self, label_vector: array_type) -> None:
        if type(label_vector) == list or type(label_vector) == tuple:
            label_vector = np.array(label_vector)
        assert type(label_vector) == array_type,"label_vector was not a numpy array, a tuple, or a list"
        self.label_vector = np.concatenate((self.label_vector, label_vector))
    
    def add(self, other: Data) -> None:
        assert isinstance(other, ImageData)
        
        self.add_images_labels(other.images, other.labels)
        self.add_label_vector(other.label_vector)

    def equals(self, other: Data) -> bool:
        return np.array_equiv(self.images, other.images) and np.array_equiv(self.labels, other.labels) and np.array_equiv(self.label_vector, other.label_vector)

    def print(self) -> None:
        print("")
        print(f"Images (shape): {self.images.shape}")
        print(f"Labels: {self.labels}")
        print(f"Label Vector: {self.label_vector}")
        print("")

    def batch(self, batch_size: int) -> tuple[array_type]:
        assert type(batch_size) == int,"batch_size must be an integer"
        assert self.images.shape[0] >= batch_size,"batch_size cannot be greater than the first dimension of images"
        images = np.array(np.array_split(self.images, self.images.shape[0] // batch_size))
        labels = np.array(np.array_split(self.labels, self.labels.shape[0] // batch_size))
        return (images, labels)


def process_image(path: str, dimensions: tuple[int], debug: bool = False) -> array_type:
        assert type(path) == str,"Path must be a string"
        assert len(dimensions) == 2,"dimensions must be of length 2"
        assert type(dimensions[0]) == type(dimensions[1]) == int,"dimensions must be an iterable of two integers"
        assert type(debug) == bool
        if not os.path.isfile(path):
                if not debug:
                        print("ERROR: Nonexistent file")
                return None

        img = Image.open(path)
        img = img.resize((dimensions[-1], dimensions[0]))
        img = np.array(img)
        return img


def display_data(data: ImageData, 
                 rows: int, cols: int, axes: bool = False, many: bool = True, fig_x: int = 12, fig_y: int = 8) -> None:

    assert isinstance(data, ImageData),"data must be an instance of the Data class"
    assert type(rows) == int,"rows must be an integer"
    assert type(cols) == int,"cols must be an integer"
    assert type(axes) == bool,"axes must be a boolean"
    assert type(many) == bool,"many must be a boolean"
    assert type(fig_x) == int and fig_x > 0,"fig_x must be a positive integer"
    assert type(fig_y) == int and fig_y > 0,"fig_y must be a positive integer"

    
    images, labels, label_vector = data.as_tuple()
    if label_vector.size > 0 and labels.size > 0:
        assert label_vector.size > np.amax(labels),"Label vector does not contain all possible labels"
    
    if not many:
        rows = 1
        cols = 1
        f, axs = plt.subplots(rows,cols,figsize=(fig_x,fig_y),tight_layout=True)
        axs.imshow(images, aspect='auto')
        if len(label_vector) != 0:
            axs.set_title(str(label_vector[labels[i]]))
        elif len(labels) != 0:
            axs.set_title(labels[i])
        if not axes:
            axs.axis("off")
        plt.show()
        return

    f, axs = plt.subplots(rows,cols,figsize=(12,8),tight_layout=True)
    i = 0
    for r in range(rows):
        for c in range(cols):

            if (len(images) > i):
                if len(label_vector) != 0:
                    axs[r,c].set_title(str(label_vector[labels[i]]))
                elif len(labels) != 0:
                    axs[r,c].set_title(labels[i])

                if rows != 1 and cols != 1:
                    axs[r,c].imshow(images[i], aspect='auto')
                    if not axes:
                        axs[r,c].axis("off")
                else:
                    axs.imshow(images[i], aspect='auto')
                    if not axes:
                        axs.axis("off")
                
            i += 1
    plt.show()