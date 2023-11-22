import numpy as np

array_type = type(np.array([]))

def relu(matrix: array_type) -> array_type:
    m = matrix
    m[m < 0] = 0
    return m


def activation_function(s: str, matrix: array_type) -> array_type:
    if s.lower():
        return relu(matrix)
    else:
        return matrix