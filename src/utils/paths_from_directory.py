import os

def paths_from_directory(path: str) -> list[str]:
    assert type(path) == str,"Path must be a string"
    if not os.path.isdir(path):
        print("ERROR: path must lead to a directory")
        return []

    paths = list(map(lambda f: os.path.join(path, f), os.listdir(path)))
    return paths