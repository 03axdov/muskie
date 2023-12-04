from .models import Model
from .data import Data
from .loss_functions import MSE, Loss
from .optimizers import SGD, Optimizer
from progressbar import *

import numpy as np
import time


def train(model: Model,
          data: Data,
          epochs: int = 5000,
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD()) -> None:

    tic = time.time()
    widgets = ['Training:', ' ', Percentage(), ' ', Bar('#'), ' ', '']
    pbar = ProgressBar(widgets=widgets, maxval=epochs)
    pbar.start()
    for epoch in range(epochs):
        cost = 0.0
    
        for batch in data.get_batches():
            inputs = batch[0]
            labels = batch[1]
            predicted = model.forward(inputs) # Compute y^
            cost += loss.loss(predicted, labels)
            grad = loss.grad(predicted, labels) # Compute da[l]

            model.backward(grad) # Use da[l] to get dW[l-1], db[l-1], dW[l-2] etc.
            
            optimizer.step(model) # Update weights and biases according to the previously calculated gradients


        cost_rounded = np.around(cost / data.batch_size, 8)
        cost_str = str(cost_rounded)
        loss_str = cost_str + " " * (10 - len(cost_str))

        widgets[-1] = FormatLabel('loss: {0}'.format(loss_str))
        pbar.update(epoch)
        # print(f"Epoch: {epoch + 1}, Loss: {cost / data.batch_size}")

    pbar.finish()
    toc = time.time()
    print("")
    print(f"[ FINISHED TRAINING IN: {round(toc-tic, 2)} SECONDS ]")
    print("")