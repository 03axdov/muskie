from PIL import Image
from process_image import process_image
import os
import numpy as np

def create_dataset(images_path: str,
                   labels: list[int],
                   image_dimension: tuple[int, int],
                   new_path: str = "datasets",
                   filename: str = "dataset",
                   ):
    paths = list(map(lambda path: os.path.join(images_path, path), os.listdir(images_path)))
    images = np.array(list(map(lambda path: process_image(path, image_dimension), paths)))

    print(images.shape)


if __name__ == "__main__":
    create_dataset("images/fish_images", [1,1,1,2,2,2,3,3,3], image_dimension=(600, 500))