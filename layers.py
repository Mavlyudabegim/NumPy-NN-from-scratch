"""
Dense (fully-connected) layer.

This is the core of backprop: given the gradient flowing in from the layer
above (dL/dy), compute the gradient flowing out to the layer below (dL/dx),
plus the gradients on this layer's own parameters (dL/dW, dL/db).
"""

import numpy as np


class Dense:
    def __init__(self, input_dim, output_dim):
        # Small random init (e.g. scaled by sqrt(1/input_dim)) avoids
        # saturating activations at the start of training. Try a couple of
        # init scales later and see how it affects convergence -- that's a
        # good thing to mention in your README write-up.
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(1.0 / input_dim)
        self.b = np.zeros((1, output_dim))

        # Cache for use in the backward pass
        self.x = None

        # Populated by backward()
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        x: (batch_size, input_dim)
        returns: (batch_size, output_dim)
        """
        self.x = x
        return x @ self.W + self.b
    
    def backward(self, dL_dy):
        """
        dL_dy: (batch_size, output_dim) -- upstream gradient (dL/d[this layer's output])

        TODO:
          1. self.dW = self.x.T @ dL_dy
          2. self.db = np.sum(dL_dy, axis=0, keepdims=True)
          3. dL_dx = dL_dy @ self.W.T
          4. return dL_dx  -- this becomes the upstream gradient for the layer below

        Double check shapes as you go: self.dW must be the same shape as
        self.W, self.db the same shape as self.b, and dL_dx the same shape
        as self.x. Shape mismatches are the #1 source of bugs here.
        """

        self.dW = self.x.T @ dL_dy
        self.db = np.sum(dL_dy, axis=0, keepdims=True)
        dL_dx = dL_dy @ self.W.T

        assert self.dW.shape == self.W.shape, f"dW shape {self.dW.shape} != W shape {self.W.shape}"
        assert self.db.shape == self.b.shape, f"db shape {self.db.shape} != b shape {self.b.shape}"
        assert dL_dx.shape == self.x.shape, f"dL_dx shape {dL_dx.shape} != x shape {self.x.shape}"

        return dL_dx

    def params_and_grads(self):
        """Already implemented -- used by optimizers.py and network.py."""
        return [(self.W, self.dW), (self.b, self.db)]