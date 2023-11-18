import numpy as np
import matplotlib.pyplot as plt

def display_data(data: tuple[list], 
                 rows: int, cols: int, axes: bool = False, many: bool = True) -> None:

    assert len(data) == 3,"data must be length 3"
    if many:
        assert type(data[0]) == type(data[1]) == type(data[2]) == list,"data must be an iterable of three lists"
    assert type(rows) == int,"rows must be an integer"
    assert type(cols) == int,"cols must be an integer"
    assert type(axes) == bool,"axes must be a boolean"
    assert type(many) == bool,"many must be a boolean"
    
    images = data[0]
    labels = data[1]
    label_vector = data[2]

    if (len(label_vector) != 0):
        assert len(label_vector) > max(labels),"Label vector does not contain all possible labels"
    if not many:
        rows = 1
        cols = 1
    
    f, axs = plt.subplots(rows,cols,figsize=(12,8),tight_layout=True)
    i = 0
    if many:
        for r in range(rows):
            for c in range(cols):

                if (len(images) > i):
                    if len(label_vector) != 0:
                        axs[r,c].set_title(str(label_vector[labels[i]]))
                    elif len(labels) != 0:
                        axs[r,c].set_title(labels[i])

                    if rows != 1 and cols != 1:
                        axs[r,c].imshow(images[i], aspect='auto')
                        if not axes:
                            axs[r,c].axis("off")
                    else:
                        axs.imshow(images[i], aspect='auto')
                        if not axes:
                            axs.axis("off")
                    
                i += 1
    else:
        axs.imshow(images, aspect='auto')
        if len(label_vector) != 0:
            axs.set_title(str(label_vector[labels[i]]))
        elif len(labels) != 0:
            axs.set_title(labels[i])
        if not axes:
            axs.axis("off")
    
    plt.show()