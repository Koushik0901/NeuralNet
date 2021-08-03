import numpy as np
from NeuralNet.tensor import Tensor

class Loss(object):
    def __call__(self, outputs, y) -> float:
        sample_losses = self.forward(outputs, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
    def forward(self, outputs, y):
        raise NotImplementedError


class MSE(Loss):

    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        mse = np.mean(np.power(y_true - y_pred, 2))
        return mse
    def backward(self):
        mse_prime = 2 * (self.y_pred - self.y_true) / np.size(self.y_true)
        return mse_prime

class CrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1: # for scalar y_true
            correct_confidences = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2: # for one hot encoded
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)         

        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood 

class BCE(Loss):
    ...

class BCEWithLogits(Loss):
    ...