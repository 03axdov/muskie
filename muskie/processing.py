from .models import Model
from .data import Data
from .loss_functions import MSE, Loss
from .optimizers import SGD, Optimizer

import numpy as np
import time


def train(model: Model,
          data: Data,
          epochs: int = 5000,
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD()) -> None:

    tic = time.time()
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

        print(f"Epoch: {epoch + 1}, Loss: {cost / data.batch_size}")
    toc = time.time()
    print("")
    print(f"[ FINISHED TRAINING IN: {round(toc-tic, 2)} SECONDS ]")
    print("")