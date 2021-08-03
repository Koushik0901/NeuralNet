"""
XOR cannot be learnt by a simple linear function, so this is a nice way to test the network.
"""
import numpy as np
import NeuralNet.nn as nn
import NeuralNet.optim as optim
from NeuralNet import Trainer
from NeuralNet.layers import Linear
from NeuralNet.activations import Tanh
from NeuralNet.loss import MSE

inputs = np.array(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ]
)

targets = np.array(
    [
        [1, 0],
        [0, 1], 
        [0, 1],
        [1, 0]

    ]
)

net = nn.Sequential(
    [
        Linear(2, 2),
        Tanh(),
        Linear(2, 2)
    ]
)

Trainer.train(net=net, inputs=inputs, targets=targets, loss_fn=MSE(), optimizer=optim.SGD(lr=1e-3))