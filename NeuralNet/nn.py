from NeuralNet.layers import Layer
from NeuralNet.tensor import Tensor
from typing import Sequence, Iterator, Tuple

class Sequential:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers
    
    def __call__(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad