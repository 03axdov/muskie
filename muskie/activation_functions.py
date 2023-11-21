import numpy as np

array_type = type(np.array([]))

def relu(matrix: array_type):
    m = matrix
    m[m < 0] = 0
    return m