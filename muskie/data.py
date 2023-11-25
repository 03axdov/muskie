import numpy as np
from PIL import Image
from typing import Iterable
import numpy as np
import os
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import NamedTuple, Iterator


array_type = type(np.array([]))
BATCH = NamedTuple("BATCH", [("inputs", array_type), ("targets", array_type)])



class DataAbstract(ABC):
    @abstractmethod
    def as_tuple(self) -> tuple[array_type]:
        pass

    @abstractmethod
    def add(self, other) -> None:
        pass

    @abstractmethod
    def equals(self, other) -> bool:
        pass

    @abstractmethod
    def print(self) -> None:
        pass

    @abstractmethod
    def get_batches(self, shuffle: bool) -> Iterator[BATCH]:
        pass


class Data(DataAbstract):
    def __init__(self, inputs: array_type = np.array([]),
                 labels: array_type = np.array([]),
                 label_vector: array_type = np.array([]),
                 create_labels: bool = False,
                 default_label: int = 0,
                 shuffle: bool = True):
        if type(inputs) == list:
            inputs = np.array(inputs)
        if type(labels) == list:
            labels = np.array(labels)
        if type(label_vector) == list:
            label_vector = np.array(label_vector)

        assert type(create_labels) == bool,"create_labels must be a boolean"
        assert type(default_label) == int,"default label must be an integer"
        assert type(shuffle) == bool,"shuffly must be a boolean"

        if create_labels:
            labels = np.full((inputs.shape[0]), default_label)
        assert inputs.shape[0] == labels.shape[0],"images and labels must have the same first dimension"

        if label_vector.size > 0:
            assert label_vector.size > np.amax(labels),"label vector does not contain all possible labels"


        self.inputs = inputs
        self.labels = labels
        self.label_vector = label_vector
        self.batch_size = len(inputs)
        self.shuffle = shuffle

    
    def as_tuple(self) -> tuple[array_type]:
        return (self.inputs, self.labels, self.label_vector)


    def add(self, other: DataAbstract) -> None:
        assert isinstance(other, Data)
        self.inputs = np.concatenate((self.inputs, other.inputs)) 
        self.labels = np.concatenate((self.labels, other.labels))
        self.label_vector = np.concatenate((self.label_vector, other.label_vector))


    def equals(self, other) -> bool:
        return np.array_equiv(self.inputs, other.inputs) and np.array_equiv(self.labels, other.labels) and np.array_equiv(self.label_vector, other.label_vector)


    def print(self) -> None:
        print("")
        print("Data:")
        print(f"Inputs (shape): {self.inputs.shape}")
        print(f"Labels: {self.labels}")
        print(f"Label Vector: {self.label_vector}")
        print("")


    def batch(self, batch_size: int):
        assert type(batch_size) == int and batch_size > 0,"batch_size must be a positive integer"
        assert batch_size <= len(self.inputs),"batch_size must be less than or equal to the length of inputs / labels"
        self.batch_size = batch_size


    def get_batches(self) -> Iterator[BATCH]:
        starts = np.arange(0, len(self.inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)

        for start in starts:
            end = start + self.batch_size
            batch_inputs = self.inputs[start:end]
            batch_labels = self.labels[start:end]

            yield BATCH(batch_inputs, batch_labels)


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


def display_data(data: Data, 
                 rows: int, cols: int, axes: bool = False, many: bool = True, fig_x: int = 12, fig_y: int = 8) -> None:

    assert isinstance(data, Data),"data must be an instance of the Data class"
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