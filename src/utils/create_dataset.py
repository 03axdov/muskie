from PIL import Image
from .process_image import process_image
from .paths_from_directory import paths_from_directory
import os
import numpy as np
import multiprocessing as mp
from multiprocessing import Process

def create_dataset(images_path: str,
                   labels: list[int],
                   image_dimensions: tuple[int, int],
                   new_path: str = "datasets/dataset",
                   ) -> tuple[list[type(np.array([]))], list[int]]:

    paths = paths_from_directory(images_path)
    assert len(paths) == len(labels)

    pool = mp.Pool(mp.cpu_count())
    processes = [pool.apply_async(process_image, args=(path,image_dimensions,)) for path in paths]
    images = [p.get() for p in processes]
    pool.close()

    return (images, labels)