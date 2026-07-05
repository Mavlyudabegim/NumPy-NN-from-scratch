"""
Training loop -- already fully implemented, do not need to modify.

Once activations.py, losses.py, layers.py, network.py, and optimizers.py
are filled in (and gradient_check.py passes), this script trains the same
network with three different optimizers and plots the loss curves side by
side. That comparison plot is the headline result for this project.

Run:
    python train.py
"""

import numpy as np
import matplotlib.pyplot as plt

from data import load_data
from network import NeuralNetwork
from optimizers import SGD, SGDMomentum, Adam


def accuracy(network, x, y_one_hot):
    preds = network.predict(x)
    true_labels = np.argmax(y_one_hot, axis=1)
    return np.mean(preds == true_labels)


def train_one_run(optimizer_name, optimizer, x_train, y_train, x_val, y_val,
                   layer_sizes, epochs=100, batch_size=32, seed=0):
    """
    Trains one network and watches VALIDATION accuracy every epoch -- this is
    what you use during development to compare optimizers, tune learning
    rates, decide when to stop, etc. The test set is never touched in here.
    """
    np.random.seed(seed)
    net = NeuralNetwork(layer_sizes)

    n = x_train.shape[0]
    history = {"loss": [], "val_acc": []}

    for epoch in range(epochs):
        # Shuffle each epoch
        perm = np.random.permutation(n)
        x_shuffled, y_shuffled = x_train[perm], y_train[perm]

        epoch_losses = []
        for start in range(0, n, batch_size):
            end = start + batch_size
            x_batch = x_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            probs = net.forward(x_batch)
            loss = net.loss(x_batch, y_batch)
            epoch_losses.append(loss)

            net.backward(y_batch)
            optimizer.step(net.all_params_and_grads())

        val_acc = accuracy(net, x_val, y_val)
        history["loss"].append(np.mean(epoch_losses))
        history["val_acc"].append(val_acc)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[{optimizer_name}] epoch {epoch:3d}  loss={history['loss'][-1]:.4f}  val_acc={val_acc:.4f}")

    return net, history


def main():
    # dataset="digits" (fast, ~1800 samples) or "mnist" (real, 70000 samples,
    # needs internet on first run; see data.py). Input layer size below
    # adapts automatically either way -- no need to hand-edit it.
    x_train, x_val, x_test, y_train, y_val, y_test = load_data(dataset="mnist")
    input_dim = x_train.shape[1]
    layer_sizes = [input_dim, 32, 16, 10]

    runs = {
        "SGD": SGD(lr=0.01),
        "SGD+Momentum": SGDMomentum(lr=0.02, beta=0.9),
        "Adam": Adam(lr=0.0005),
    }

    histories = {}
    trained_nets = {}
    for name, optimizer in runs.items():
        net, history = train_one_run(name, optimizer, x_train, y_train, x_val, y_val, layer_sizes)
        histories[name] = history
        trained_nets[name] = net

    test_accuracies = {name: accuracy(net, x_test, y_test) for name, net in trained_nets.items()}

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    for name, history in histories.items():
        ax_loss.plot(history["loss"], label=name)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Training loss")
    ax_loss.set_title("Training loss")
    ax_loss.legend()

    for name, history in histories.items():
        line, = ax_acc.plot(history["val_acc"], label=f"{name} (val)")
        ax_acc.axhline(y=test_accuracies[name], color=line.get_color(), linestyle="--", alpha=0.6)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Validation accuracy (solid) vs. final test accuracy (dashed)")
    ax_acc.legend(loc="lower right")

    fig.suptitle("Optimizer convergence comparison")
    fig.tight_layout()
    fig.savefig("optimizer_comparison.png", dpi=150)
    print("\nSaved optimizer_comparison.png")

    for name, history in histories.items():
        print(f"{name}: final val_acc = {history['val_acc'][-1]:.4f}")

    print("\nFinal test accuracy (test set touched once, here only):")
    for name, test_acc in test_accuracies.items():
        print(f"{name}: test_acc = {test_acc:.4f}")

if __name__ == "__main__":
    main()