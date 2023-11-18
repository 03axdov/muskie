import numpy as np
import matplotlib.pyplot as plt

def display_data(data: tuple[list], 
                 rows: int, cols: int, axes: bool = False) -> None:

    assert len(data) == 3,"data must be length 3"
    assert type(data[0]) == type(data[1]) == type(data[2]) == list,"data must be an iterable of three lists"
    assert type(rows) == int,"rows must be an integer"
    assert type(cols) == int,"cols must be an integer"
    assert type(axes) == bool,"axes must be a boolean"
    
    images = data[0]
    labels = data[1]
    label_vector = data[2]

    if (len(label_vector) != 0):
        assert len(label_vector) > max(labels),"Label vector does not contain all possible labels"
    
    f, axarr = plt.subplots(rows,cols,figsize=(12,8),tight_layout=True)
    i = 0
    for r in range(rows):
        for c in range(cols):
            if (len(images) > i):
                axarr[r,c].imshow(images[i], aspect='auto')
                if (len(label_vector)) != 0:
                    axarr[r,c].set_title(str(label_vector[labels[i]]))
                else:
                    axarr[r,c].set_title(labels[i])
                axarr[r,c].axis("off")
            i += 1
    
    plt.show()