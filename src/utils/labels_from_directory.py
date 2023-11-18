from .paths_from_directory import paths_from_directory
import os

def labels_from_directory(path: str, split: str = "_", debug: bool = False) -> tuple:
    assert type(path) == str
    assert type(split) == str
    assert type(debug) == bool
    if not os.path.isdir(path):
        if not debug:
            print("ERROR: Nonexistent directory")
        return ([], [])

    paths = paths_from_directory(path)
    labels = []
    label_vector = []
    label_dict = {}

    for path in paths:
        label_name = os.path.basename(path).split(split)[0]
        if label_name not in label_vector:
            label_dict[label_name] = len(label_vector)
            label_vector.append(label_name)

        labels.append(label_dict[label_name])

    return (labels, label_vector)