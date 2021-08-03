import numpy as np
from NeuralNet.tensor import Tensor
from scipy import signal


class Layer:
    def __init__(self) -> None:
        self.params: dict[str, Tensor] = {}
        self.grads: dict[str, Tensor] = {}
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError

##########
# LAYERS #
##########

class Linear(Layer):
    """ Regular densely-connected neural network layer

        This layer implements the operation:
            output = dot(inputs * weights) + bias

        Args:
            input_size (int): size of input sample
            output_size (int): size of output sample
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        # inputs.shape: (batch_size, input_size)
        # outputs.shape: (batch_size, output_size)
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return np.dot(inputs, self.params["w"]) + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        self.grads["w"] = np.dot(self.inputs.T, grad)
        self.grads["b"] = np.sum(grad, axis=0)
        input_gradient = np.dot(grad, self.params["w"].T)
        # self.params["w"] -= learning_rate * self.grads["w"]
        # self.params["b"] -= learning_rate * grad
        return input_gradient



class Conv2d(Layer):
    """ 2D convolution layer (e.g. spatial convolution over images)

        This layer creates a convolution kernel that is convolved with the layer
        input to produce a ndarray output.

        Args:
            kernels (int): Number of kernels to use
            kernel_size (int): Size of each kernels
            input_shape (tuple): Shape of the input data

    """
    def __init__(self, kernels: int, kernel_size: int, input_shape: int) -> None:
        super().__init__()
        input_depth, input_height, input_width = input_shape
        self.filters = filters
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (filters, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (filters, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.params["b"] = np.random.randn(*self.output_shape)

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        self.outputs = np.copy(self.params["b"])
        for i in range(self.filters):
            for j in range(self.input_depth):
                self.outputs[i] += signal.correlate2d(self.inputs[j], self.kernels[i, j], "valid")
        return self.outputs

    def backward(self, grad: Tensor) -> Tensor:
        # TODO: backward pass for conv
        ...

