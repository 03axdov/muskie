from PIL import Image
from .process_image import process_image
from .paths_from_directory import paths_from_directory
from .labels_from_directory import labels_from_directory
import os
import numpy as np
import multiprocessing as mp
from multiprocessing import Process

def create_dataset(path: str,
                   dimensions: tuple[int],
                   create_labels: bool = True,
                   split: str = "_",
                   ) -> tuple:

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
        return (images, labels, label_vector)
    else:
        return (images, [], [])