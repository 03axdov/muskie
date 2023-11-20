USE_GPU = False


def use_gpu():
    USE_GPU = True


def gpu() -> bool:
    return USE_GPU