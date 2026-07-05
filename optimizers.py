"""
Optimizers: given current parameters and their gradients, update the
parameters in place. All three share the same interface so train.py can
swap between them without changing anything else -- that's what makes
the convergence comparison plot possible.
"""

import numpy as np


class SGD:
    def __init__(self, lr=0.1):
        self.lr = lr

    def step(self, params_and_grads):
        """
        params_and_grads: list of (param, grad) tuples, e.g. [(W1, dW1), (b1, db1), ...]

        """
        for param, grad in params_and_grads:
            param -= self.lr * grad

class SGDMomentum:
    def __init__(self, lr=0.1, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocities = None  # lazily initialized on first step() call

    def step(self, params_and_grads):
        if self.velocities is None:
            self.velocities = [np.zeros_like(param) for param, grad in params_and_grads]

        for i, (param, grad) in enumerate(params_and_grads):
            v = self.velocities[i]
            v[:] = self.beta * v + grad
            param -= self.lr * v


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None  # first moment (mean of gradients)
        self.v = None  # second moment (mean of squared gradients)
        self.t = 0     # timestep, needed for bias correction

    def step(self, params_and_grads):
        self.m = self.m or [np.zeros_like(param) for param, grad in params_and_grads]
        self.v = self.v or [np.zeros_like(param) for param, grad in params_and_grads]
        self.t += 1
        for i, (param, grad) in enumerate(params_and_grads):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)