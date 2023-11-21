USE_GPU = False


def use_gpu():
    global USE_GPU
    USE_GPU = True


def dont_use_gpu():
    global USE_GPU
    USE_GPU = False


def gpu() -> bool:
    return USE_GPU