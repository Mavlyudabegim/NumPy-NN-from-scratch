"""
Training loop 

This script trains the same
network with three different optimizers and plots the loss curves side by
side. That comparison plot is the headline result for this project.

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


def train_one_run(optimizer_name, optimizer, x_train, y_train, x_test, y_test,
                   layer_sizes, epochs=100, batch_size=32, seed=0):
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

        val_acc = accuracy(net, x_test, y_test)
        history["loss"].append(np.mean(epoch_losses))
        history["val_acc"].append(val_acc)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[{optimizer_name}] epoch {epoch:3d}  loss={history['loss'][-1]:.4f}  val_acc={val_acc:.4f}")

    return history


def main():
    x_train, x_test, y_train, y_test = load_data()
    layer_sizes = [64, 32, 16, 10]

    runs = {
        "SGD": SGD(lr=0.5),
        "SGD+Momentum": SGDMomentum(lr=0.1, beta=0.9),
        "Adam": Adam(lr=0.01),
    }

    histories = {}
    for name, optimizer in runs.items():
        histories[name] = train_one_run(name, optimizer, x_train, y_train, x_test, y_test, layer_sizes)

    # Plot loss convergence comparison -- this is the headline result
    plt.figure(figsize=(8, 5))
    for name, history in histories.items():
        plt.plot(history["loss"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Optimizer convergence comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("optimizer_comparison.png", dpi=150)
    print("\nSaved optimizer_comparison.png")

    for name, history in histories.items():
        print(f"{name}: final val_acc = {history['val_acc'][-1]:.4f}")


if __name__ == "__main__":
    main()