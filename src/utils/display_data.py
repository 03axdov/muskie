import numpy as np
import matplotlib.pyplot as plt

def display_data(data: tuple[list[type(np.array([]))], list[int]], 
                 rows: int, cols: int, axes: bool = False, label_vector: list = []) -> None:

    assert type(data) == tuple, "Incorrect data type, must be a tuple"
    assert type(rows) == int,"rows must be an integer"
    assert type(cols) == int,"cols must be an integer"
    assert type(axes) == bool,"axes must be a boolean"
    assert type(label_vector) == type([]),"label_vector must be a list"
    
    images = data[0]
    labels = data[1]

    if (len(label_vector) != 0):
        assert len(label_vector) > max(labels) - 1,"Label vector does not contain all possible labels"
    
    f, axarr = plt.subplots(rows,cols,figsize=(12,8),tight_layout=True)
    i = 0
    for r in range(rows):
        for c in range(cols):
            if (len(images) > i):
                axarr[r,c].imshow(images[i], aspect='auto')
                if (len(label_vector)) != 0:
                    axarr[r,c].set_title(str(label_vector[labels[i] - 1]))
                else:
                    axarr[r,c].set_title(labels[i])
                axarr[r,c].axis("off")
            i += 1
    
    plt.show()