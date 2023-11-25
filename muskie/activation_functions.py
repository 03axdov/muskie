import numpy as np

array_type = type(np.array([]))

def relu(matrix: array_type, prime: bool = False) -> array_type:
    if not prime:
        m = matrix
        m[m < 0] = 0
        return m
    else:
        return (matrix > 0) * 1
    

def tanh(matrix: array_type, prime: bool = False) -> array_type:
    if not prime:
        return np.tanh(matrix)
    else:
        y = self.tanh(x)
        return 1 - y**2
    

def activation_function(s: str, matrix: array_type, prime: bool = False) -> array_type:
    if s.lower() == "relu":
        return relu(matrix, prime)
    elif s.lower() == "tanh":
        return tanh(matrix, prime)
    else:
        return matrix