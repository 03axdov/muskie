from process_image import process_image

def process_image_decorator(path: str, dimensions: tuple[int, int], array: list):
        array.append(process_image(path, dimensions))
        print(process_image(path, dimensions).shape)