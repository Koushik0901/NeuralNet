# NeuralNet
***A deep learning library from scratch using NumPy and SciPy.***

## Components
| Component | Description |
| ---- | ---- |
| NeuralNet.Trainer | Has the Trainer used to train any neural network |
|NeuralNet.nn | Has the sequential class to combine layers |
| NeuralNet.layers| Has the implementation of various layers |
| NeuralNet.activations | Has the implementation of activation functions |
| NeuralNet.optim | Has the implementation of various optimizers |
| NeuralNet.loss | Has the implementation of various loss functions |

## Example
```python
# Necessary imports
from NeuralNet.layers import Linear
from NeuralNet.activations import ReLU
import NeuralNet.optim as optim
from NeuralNet.losses import MSE
from NeuralNet import Trainer, Sequential

# Define the network
net = nn.Sequential(
    [
        Linear(1, 10),
        ReLU(),
        Linear(10, 1)
    ]
)
# Train the model
Trainer.train(net=net, inputs=inputs, targets=targets, loss_fn=MSE(), optimizer=optim.SGD(lr=1e-3))
```
