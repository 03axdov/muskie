import numpy as np

array_type = type(np.array([]))

def relu(matrix: array_type) -> array_type:
    m = matrix
    m[m < 0] = 0
    return m


def activation_function(s: str, matrix: array_type) -> array_type:
    if s.lower() == "relu":
        return relu(matrix)
    else:
        return matrix


def activation_function_prime(s: str, matrix: array_type) -> array_type:
    if s.lower() == "relu":
        return (matrix > 0) * 1
    else:
        return matrix