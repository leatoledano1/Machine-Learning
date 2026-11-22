import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    score = np.sum(y_true == y_pred)
    return score / len(y_true)


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples,) or (n_samples, input_dim)
            True labels. Can be categorical indices or one-hot encoded vectors.
    y_pred: ndarray of shape (n_samples, input_dim)
        Predicted distribution (softmax) of each sample

    Returns
    -------
    output: float
        Cross entropy of given y_true, y_pred
    """
    n_samples = y_pred.shape[0]

    if y_true.ndim == 1:
        prob_correct = y_pred[np.arange(n_samples), y_true.astype(int)]
        return -np.mean(np.log(prob_correct + 1e-15))
    else:
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-15), axis=-1))


def softmax(X: np.ndarray) -> np.ndarray:
    """
    Compute the Softmax function for each sample in given data

    Parameters:
    -----------
    X: ndarray of shape (n_samples, input_dim)

    Returns:
    --------
    output: ndarray of shape (n_samples, input_dim)
        Softmax(x) for every sample x in the given data X
    """
    max_X = np.max(X, axis=-1, keepdims=True)
    exp_X = np.exp(X - max_X)
    return exp_X / np.sum(exp_X, axis=-1, keepdims=True)