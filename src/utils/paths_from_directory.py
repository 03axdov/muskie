import os

def paths_from_directory(path: str) -> list[str]:
    paths = list(map(lambda f: os.path.join(path, f), os.listdir(path)))
    return paths