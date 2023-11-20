USE_GPU = False


def use_gpu():
    USE_GPU = True


def dont_use_gpu():
    USE_GPU = False


def gpu() -> bool:
    return USE_GPU