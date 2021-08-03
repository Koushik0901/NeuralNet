from NeuralNet.tensor import Tensor
from NeuralNet.nn import Sequential
from NeuralNet.loss import Loss
from NeuralNet.optim import Optimizer
from NeuralNet.data import DataIterator, BatchIterator
from tqdm import tqdm

def train(
    net: Sequential,
    inputs: Tensor,
    targets: Tensor,
    loss_fn: Loss,
    optimizer: Optimizer,
    num_epoch: int = 300,
    iterator: DataIterator = BatchIterator(),
) -> None:
    
    for epoch in range(num_epoch):
        epoch_loss = 0.0
        pbar = tqdm(iterator(inputs, targets))
        for batch in pbar:
            predicted = net(batch.inputs)
            epoch_loss += loss_fn(predicted, batch.targets)
            grad = loss_fn.backward()
            net.backward(grad)
            optimizer.step(net)
            pbar.set_description(f"epoch:{epoch}")
            pbar.set_postfix({"running loss": epoch_loss})

if __name__ == '__main__':
    train()