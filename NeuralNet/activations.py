import numpy as np
from NeuralNet.tensor import Tensor
from NeuralNet.layers import Layer
from typing import Callable

F = Callable[[Tensor], Tensor]

class Activation(Layer):
    def __init__(self, function: F, function_prime: F) -> None:
        super().__init__()
        self.function = function
        self.function_prime = function_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.function(self.inputs)

    def backward(self, grad: Tensor) -> Tensor:
        return grad * self.function_prime(self.inputs)

########################
# ACTIVATION FUNCTIONS #
########################

class ReLU(Activation):
    def __init__(self) -> None:
        relu = lambda x: np.maximum(0, x)
        relu_prime = lambda x: np.greater(x, 0).astype(int) # assume 0 for case x=0
        super().__init__(relu, relu_prime)

# TO BE IMPLEMENTED
class ReLU6(Activation):
    """ ReLU6(x) = min(max(0,x),6) """
    def __init__(self) -> None:
        relu6 = lambda x: np.minimum(np.maximum(0, x), 6)
        relu6_prime = lambda x: None
        super().__init__(relu6, relu6_prime)

class LeakyReLU(Activation):
    """ LeakyReLU(x) = max(negative_slope * 0, x) """
    def __init__(self, negative_slope: float) -> None:
        leakyrelu = lambda x: np.max(0, x) + negative_slope * np.minimum(x)
        leakyrelu_prime = lambda x: negative_slope if x <= 0 else 1 # assume negative slope for case x=0
        super().__init__(leakyrelu, leakyrelu_prime)

class Sigmoid(Activation):
    def __init__(self) -> None:
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        sigmoid_prime = lambda x: np.dot(sigmoid(x), (1 - sigmoid(x)))
        super().__init__(sigmoid, sigmoid_prime)

class Tanh(Activation):
    def __init__(self) -> None:
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2 # Derivative of tanh: 1 - [tanh(x)] ** 2
        super().__init__(tanh, tanh_prime)


class Softmax(Activation):
    def __init__(self):
        exp_values = lambda x: np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = lambda y: exp_values(y) / np.sum(exp_values(y), axis = 1, keepdims = True)
        softmax_prime = None # TODO: softmax_prime
        super().__init__(softmax, softmax_prime)