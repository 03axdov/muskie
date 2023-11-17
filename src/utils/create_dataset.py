from PIL import Image
from process_image import process_image
from create_dataset_files import create_dataset_files
import os
import numpy as np
import multiprocessing as mp
from multiprocessing import Process

def create_dataset(images_path: str,
                   labels: list[int],
                   image_dimension: tuple[int, int],
                   new_path: str = "datasets/dataset",
                   ):

    paths = list(map(lambda path: os.path.join(images_path, path), os.listdir(images_path)))
    assert len(paths) == len(labels)

    pool = mp.Pool(mp.cpu_count())
    processes = [pool.apply_async(process_image, args=(path,image_dimension,)) for path in paths]
    images = [p.get() for p in processes]

    return (images, labels)


if __name__ == "__main__":
    create_dataset("images/fish_images", [1,1,1,2,2,2,3,3,3], image_dimension=(600, 500))