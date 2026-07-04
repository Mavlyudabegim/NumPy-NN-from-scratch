"""
Loss functions.

We use cross-entropy loss on top of softmax outputs, for multi-class
classification (10 digit classes).
"""

import numpy as np


def cross_entropy_loss(predictions, labels_one_hot):
    """
    predictions: (batch_size, num_classes) -- softmax probabilities, rows sum to 1
    labels_one_hot: (batch_size, num_classes) -- one-hot true labels

    """
    predictions = np.clip(predictions, 1e-12, 1.0)
    loss = -np.mean(np.sum(labels_one_hot * np.log(predictions), axis=1))
    return loss


def cross_entropy_softmax_grad(predictions, labels_one_hot):
    """
    Gradient of cross-entropy loss w.r.t. the PRE-softmax logits, when the
    network's final layer is softmax and the loss is cross-entropy.

    predictions: (batch_size, num_classes) -- softmax probabilities
    labels_one_hot: (batch_size, num_classes)

    """

    batch_size = predictions.shape[0]
    return (predictions - labels_one_hot) / batch_size