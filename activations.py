"""
Activation functions and their derivatives.

Each activation needs two functions:
  - the forward function itself
  - its derivative, needed during backprop (chain rule)
"""

import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.greater(x, 0).astype(int)


def sigmoid(x):
    x=np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
