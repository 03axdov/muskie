import os

def paths_from_directory(path: str, debug: bool = False) -> list[str]:
    assert type(path) == str,"Path must be a string"
    assert type(debug) == bool
    if not os.path.isdir(path):
        if not debug:
            print("ERROR: Nonexistent directory")
        return []

    paths = list(map(lambda f: os.path.join(path, f), os.listdir(path)))
    return paths