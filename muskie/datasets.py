from .system import paths_from_directory
from .system import labels_from_directory
import multiprocessing as mp
import os
import numpy as np

from .data import Data, process_image


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
    images = np.array([p.get() for p in processes])
    pool.close()

    if create_labels:
        return Data(images, labels, label_vector)
    else:
        return Data(images, np.array([]), np.array([]))


def create_dataset_subdirectories(path: str,
                   dimensions: tuple[int],
                   create_labels: bool = True,
                   ) -> type(Data):

    assert type(path) == str, "path must be a string"    # paths_from directory tests if the path is valid
    assert len(dimensions) == 2,"dimensions must be an iterable of length 2"
    assert type(dimensions[0]) == type(dimensions[1]) == int,"dimensions must be an iterable of two integers"
    assert type(create_labels) == bool

    if create_labels:
        label_vector = os.listdir(path)
    else:
        label_vector = np.array([])

    dir_paths = paths_from_directory(path)
    if len(dir_paths) == 0:
        print("ERROR: No items in this directory")
        return
    
    data = 0
    for t,path in enumerate(dir_paths):
        sub_dset = create_dataset(path, dimensions, create_labels=False)
        if data == 0:
            data = Data()
            data.images = sub_dset.images
            data.labels = sub_dset.labels
            if create_labels:
                data.labels = np.full((len(sub_dset.images)), t)
                
        else:
            data.add_images(sub_dset.images)
            if create_labels:
                data.add_labels(np.full((len(sub_dset.images)), t))

    data.label_vector = label_vector

    return data