import numpy as np

def vectorized_result(j: np.int64) -> np.ndarray:
    """
    Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Apply the sigmoid function elementwise."""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z: np.ndarray) -> np.ndarray:
    """Apply the derivative of the sigmoid function elementwise."""
    return sigmoid(z) * (1 - sigmoid(z))
