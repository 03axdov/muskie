import numpy as np
from PIL import Image
from typing import Iterable
import numpy as np
import os
import multiprocessing as mp
import matplotlib.pyplot as plt

from .files import paths_from_directory
from .files import labels_from_directory


array_type = type(np.array([]))


class Data():
    def __init__(self, images: array_type, labels: array_type = np.array([]), label_vector: array_type = []):
        if len(labels) > 0:
            assert len(images) == len(labels),"there must be a label for every image but the length of 'labels' was not equal to that of 'images'"
        if len(label_vector) > 0:
            assert len(label_vector) > max(labels),"Label vector does not contain all possible labels"

        self.images = images
        self.labels = labels
        self.label_vector = label_vector

    
    def as_tuple(self) -> tuple:
        return (self.images, self.labels, self.label_vector)


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


def create_dataset(path: str,
                   dimensions: tuple[int],
                   create_labels: bool = True,
                   split: str = "_",
                   ) -> type(Data):

    assert type(path) == str, "path must be a string"    # paths_from directory tests if the path is valid
    assert len(dimensions) == 2,"dimensions must be an iterable of length 2"
    assert type(dimensions[0]) == type(dimensions[1]) == int,"dimensions must be an iterable of two integers"
    assert type(create_labels) == bool
    assert type(split) == str

    paths = paths_from_directory(path)

    if create_labels:
        labels, label_vector = labels_from_directory(path, split=split)
        assert len(paths) == len(labels)

    pool = mp.Pool(mp.cpu_count())
    processes = [pool.apply_async(process_image, args=(p,dimensions,)) for p in paths]
    images = [p.get() for p in processes]
    pool.close()

    if create_labels:
        return Data(images, labels, label_vector)
    else:
        return Data(images, [], [])


def display_data(data: Data, 
                 rows: int, cols: int, axes: bool = False, many: bool = True) -> None:

    assert isinstance(data, Data),"data must be an instance of the Data class"
    assert type(rows) == int,"rows must be an integer"
    assert type(cols) == int,"cols must be an integer"
    assert type(axes) == bool,"axes must be a boolean"
    assert type(many) == bool,"many must be a boolean"
    
    images, labels, label_vector = data.as_tuple()

    if (len(label_vector) != 0):
        assert len(label_vector) > max(labels),"Label vector does not contain all possible labels"
    
    if not many:
        rows = 1
        cols = 1
        f, axs = plt.subplots(rows,cols,figsize=(12,8),tight_layout=True)
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