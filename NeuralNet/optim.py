import numpy as np
from NeuralNet.nn import Sequential

class Optimizer:
    def step(self, net: Sequential) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, net: Sequential) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr * grad