"""
Numerical gradient checker

This is how verification of custom backward pass is actually correct, rather than
just "runs without crashing." It compares analytical gradient (from
network.backward()) against a numerical approximation using the definition
of a derivative:

    df/dx ~= (f(x + eps) - f(x - eps)) / (2 * eps)

If they agree to about 1e-5 or better, custom backward pass is correct. 
"""

import numpy as np
from network import NeuralNetwork


def numerical_gradient(network, param, x, labels_one_hot, eps=1e-5):
    """Finite-difference approximation of dLoss/dParam, elementwise."""
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original_value = param[idx]

        param[idx] = original_value + eps
        loss_plus = network.loss(x, labels_one_hot)

        param[idx] = original_value - eps
        loss_minus = network.loss(x, labels_one_hot)

        param[idx] = original_value  # restore
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)

        it.iternext()
    return grad


def relative_error(analytical, numerical):
    """
    Scale-invariant comparison: (|a - b|) / max(|a|, |b|, tiny)
    Returns the max elementwise relative error between the two gradient
    arrays -- a single number, easy to threshold against.
    """
    denom = np.maximum(np.abs(analytical), np.abs(numerical))
    denom = np.maximum(denom, 1e-8)
    return np.max(np.abs(analytical - numerical) / denom)


def check_network(network, x, labels_one_hot, param_sample_size=20, threshold=1e-4):
    """
    Runs analytical backward pass once, then spot-checks a random subset of
    entries in each parameter array against the numerical gradient (checking
    every single weight would be correct but slow for large layers).
    """
    # Analytical gradients: one forward + one backward pass
    network.forward(x)
    network.backward(labels_one_hot)

    all_passed = True
    for layer_idx, layer in enumerate(network.layers):
        for param_name, param, grad in [("W", layer.W, layer.dW), ("b", layer.b, layer.db)]:
            flat_indices = np.array(
                [np.unravel_index(i, param.shape) for i in range(param.size)]
            )
            sample = flat_indices[
                np.random.choice(len(flat_indices), size=min(param_sample_size, len(flat_indices)), replace=False)
            ]

            max_err = 0.0
            for idx in sample:
                idx = tuple(idx)
                original_value = param[idx]

                param[idx] = original_value + 1e-5
                loss_plus = network.loss(x, labels_one_hot)
                param[idx] = original_value - 1e-5
                loss_minus = network.loss(x, labels_one_hot)
                param[idx] = original_value

                numerical = (loss_plus - loss_minus) / (2 * 1e-5)
                analytical = grad[idx]
                denom = max(abs(numerical), abs(analytical), 1e-8)
                err = abs(numerical - analytical) / denom
                max_err = max(max_err, err)

            status = "PASS" if max_err < threshold else "FAIL"
            if max_err >= threshold:
                all_passed = False
            print(f"Layer {layer_idx} {param_name}: max relative error = {max_err:.2e}  [{status}]")

    return all_passed


if __name__ == "__main__":
    np.random.seed(0)

    # Small network, small batch -- gradient checking is O(num_params) forward
    # passes per param checked, so keep this tiny.
    net = NeuralNetwork([4, 5, 3])
    x = np.random.randn(3, 4)
    labels = np.zeros((3, 3))
    labels[np.arange(3), np.random.randint(0, 3, size=3)] = 1.0

    passed = check_network(net, x, labels)
    print("\nAll checks passed!" if passed else "\nSome checks FAILED -- fix your backward pass before training.")