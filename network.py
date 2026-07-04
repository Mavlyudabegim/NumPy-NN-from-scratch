"""
Full network: wires Dense layers + ReLU activations + a softmax output
layer together, with a full forward pass and full backward pass.

Architecture (fixed for this project, feel free to change layer sizes):
  input (64 features, 8x8 digit images flattened)
    -> Dense -> ReLU
    -> Dense -> ReLU
    -> Dense -> softmax (10 classes)
"""

import numpy as np
from layers import Dense
from activations import relu, relu_derivative, softmax
from losses import cross_entropy_loss, cross_entropy_softmax_grad


class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        layer_sizes: list like [64, 32, 16, 10] meaning
          input dim 64 -> hidden 32 -> hidden 16 -> output 10

        Already implemented: builds a list of Dense layers based on
        consecutive pairs in layer_sizes.
        """
        self.layers = [
            Dense(layer_sizes[i], layer_sizes[i + 1])
            for i in range(len(layer_sizes) - 1)
        ]
        # Caches needed for backward pass -- filled in during forward()
        self._pre_activations = []  # pre-activation (z) for each hidden layer, for relu_derivative
        self._final_probs = None    # softmax output, for the loss gradient

    def forward(self, x):
        """
        x: (batch_size, input_dim)
        returns: (batch_size, num_classes) softmax probabilities

        Note: relu_derivative(z) needs the PRE-activation z, which is why
        we cache it here instead of recomputing during backward().
        """

        self._pre_activations = []
        for layer in self.layers[:-1]:
            z = layer.forward(x)
            self._pre_activations.append(z)
            x = relu(z)

        z = self.layers[-1].forward(x)
        self._final_probs = softmax(z)
        return self._final_probs

    def backward(self, labels_one_hot):
        """
        labels_one_hot: (batch_size, num_classes)

        This is the step gradient_check.py verifies. Do not move on to
        optimizers.py until gradient_check.py passes.
        """
        grad = cross_entropy_softmax_grad(self._final_probs, labels_one_hot)
        grad = self.layers[-1].backward(grad)
        for i in range(len(self.layers) - 2, -1, -1):
            grad = grad * relu_derivative(self._pre_activations[i])
            grad = self.layers[i].backward(grad)

    def loss(self, x, labels_one_hot):
        """Already implemented -- convenience wrapper for training/eval."""
        probs = self.forward(x)
        return cross_entropy_loss(probs, labels_one_hot)

    def predict(self, x):
        """Already implemented -- returns predicted class indices."""
        probs = self.forward(x)
        return np.argmax(probs, axis=1)

    def all_params_and_grads(self):
        """Already implemented -- flattens params/grads across all layers."""
        out = []
        for layer in self.layers:
            out.extend(layer.params_and_grads())
        return out