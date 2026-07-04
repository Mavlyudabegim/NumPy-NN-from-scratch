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

        TODO: for each (param, grad) pair, update in place:
          param -= self.lr * grad

        Careful: params are numpy arrays passed by reference. Use
        `param -= ...` (in-place) not `param = param - ...` (rebinds the
        local variable, doesn't affect the caller's array).
        """
        raise NotImplementedError


class SGDMomentum:
    def __init__(self, lr=0.1, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocities = None  # lazily initialized on first step() call

    def step(self, params_and_grads):
        """
        TODO:
          1. On the first call, initialize self.velocities as a list of
             zero arrays matching each param's shape.
          2. For each (param, grad), with matching velocity v:
               v[:] = self.beta * v + grad
               param -= self.lr * v

        Compare against plain SGD once implemented: momentum should reach
        a low loss in fewer epochs on this dataset, and you should be able
        to explain why (accumulating gradient direction across steps,
        damping oscillation in narrow ravines of the loss surface).
        """
        raise NotImplementedError


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
        """
        TODO:
          1. On first call, initialize self.m and self.v as lists of zero
             arrays matching each param's shape.
          2. self.t += 1
          3. For each (param, grad), with matching m_i, v_i:
               m_i[:] = self.beta1 * m_i + (1 - self.beta1) * grad
               v_i[:] = self.beta2 * v_i + (1 - self.beta2) * (grad ** 2)
               m_hat = m_i / (1 - self.beta1 ** self.t)
               v_hat = v_i / (1 - self.beta2 ** self.t)
               param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        Be ready to explain bias correction: without it, m and v are
        biased toward zero during the first few steps (since they start
        at zero), so dividing by (1 - beta^t) corrects for that.
        """
        raise NotImplementedError