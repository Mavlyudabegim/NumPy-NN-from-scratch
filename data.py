"""
Data loading

Uses sklearn's built-in digits dataset (8x8 grayscale images, 10 classes,
~1800 samples) so there's no download step and training is fast on CPU.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_data(test_size=0.2, seed=0):
    """
    Returns x_train, x_test, y_train, y_test where:
      x_* are (N, 64) float arrays, pixel values scaled to [0, 1]
      y_* are (N, 10) one-hot label arrays
    """
    digits = load_digits()
    x = digits.data.astype(np.float64) / 16.0  # pixel values are 0-16
    y_labels = digits.target

    num_classes = 10
    y_one_hot = np.zeros((len(y_labels), num_classes))
    y_one_hot[np.arange(len(y_labels)), y_labels] = 1.0

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_one_hot, test_size=test_size, random_state=seed, stratify=y_labels
    )
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_data()
    print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"x_test:  {x_test.shape}, y_test:  {y_test.shape}")