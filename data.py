"""
Data loading

Two options, both 10-class digit classification, both go through the same
train/val/test splitting logic:

  - "digits" (default): sklearn's built-in 8x8 images, ~1800 samples total.
    No download, trains in seconds -- use this while you're still debugging
    activations.py/layers.py/network.py/optimizers.py.

  - "mnist": the real 28x28 MNIST, 70000 samples total, fetched once via
    sklearn (cached locally afterward). Needs internet access on first run.
    Held-out sets are ~35x bigger (10500 vs 270 for a 15% split), so
    optimizer comparisons are far less noisy -- worth switching to once your
    implementation is verified and you want a real headline result.
"""

import numpy as np
from sklearn.datasets import load_digits, fetch_openml
from sklearn.model_selection import train_test_split


def _load_digits_raw():
    digits = load_digits()
    x = digits.data.astype(np.float64) / 16.0  # pixel values are 0-16
    y_labels = digits.target.astype(int)
    return x, y_labels


def _load_mnist_raw():
    # Downloads ~70MB on first call, cached by sklearn afterward (usually in
    # ~/scikit_learn_data). Requires internet access.
    x, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")
    x = x.astype(np.float64) / 255.0  # pixel values are 0-255
    y_labels = y.astype(int)
    return x, y_labels


def load_data(dataset="digits", val_size=0.15, test_size=0.15, seed=0):
    """
    Three-way split, not just train/test:
      - train: fit the weights
      - val:   watch every epoch, use to pick hyperparameters (lr, epochs, etc.)
      - test:  touch exactly ONCE, at the very end, for the number you report

    Watching "test" accuracy every epoch and using it to tune anything (like
    we did earlier when picking SGD+Momentum's learning rate) makes it a
    validation set in practice, not a real test set -- reporting that same
    number as your final result would be quietly overfitting to it through
    your own tuning choices. Keeping a genuinely untouched test set avoids
    that.

    dataset: "digits" (fast, ~1800 samples, default) or "mnist" (real,
      70000 samples, needs internet access on first call).

    Returns x_train, x_val, x_test, y_train, y_val, y_test where:
      x_* are (N, num_features) float arrays, pixel values scaled to [0, 1]
        (num_features is 64 for digits, 784 for mnist)
      y_* are (N, 10) one-hot label arrays
    """
    if dataset == "digits":
        x, y_labels = _load_digits_raw()
    elif dataset == "mnist":
        x, y_labels = _load_mnist_raw()
    else:
        raise ValueError(f"Unknown dataset '{dataset}', expected 'digits' or 'mnist'")

    num_classes = 10
    y_one_hot = np.zeros((len(y_labels), num_classes))
    y_one_hot[np.arange(len(y_labels)), y_labels] = 1.0

    # First carve off the test set (untouched until the final evaluation)
    x_trainval, x_test, y_trainval, y_test, labels_trainval, _ = train_test_split(
        x, y_one_hot, y_labels, test_size=test_size, random_state=seed, stratify=y_labels
    )

    # Then split the remainder into train/val. val_size is a fraction of the
    # ORIGINAL dataset, so rescale it relative to what's left after removing test.
    relative_val_size = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=relative_val_size, random_state=seed, stratify=labels_trainval
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


if __name__ == "__main__":
    for dataset in ["mnist"]:  
        x_train, x_val, x_test, y_train, y_val, y_test = load_data(dataset=dataset)
        print(f"--- {dataset} ---")
        print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
        print(f"x_val:   {x_val.shape}, y_val:   {y_val.shape}")
        print(f"x_test:  {x_test.shape}, y_test:  {y_test.shape}")