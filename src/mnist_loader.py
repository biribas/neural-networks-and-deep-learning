"""
mnist_loader

A library to load the MNIST image data. For details of the data
structures that are returned, see the docstrings for ``load_data``
and ``load_data_wrapper``. In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

import pickle
import gzip
import numpy as np
from utils import vectorized_result

TrainingSample = tuple[np.ndarray, np.ndarray]
EvalSample = tuple[np.ndarray, np.int64]

def load_data() -> tuple[
    list[TrainingSample],
    list[EvalSample],
    list[EvalSample]
]:
    """
    Return a tuple containing ``(training_data, validation_data,
    test_data)`` in a format more convenient for use in our neural
    network implementation.
    """
    with gzip.open('../data/mnist.pkl.gz', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')

    training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    training_outputs = [vectorized_result(y) for y in training_data[1]]
    training_set = list(zip(training_inputs, training_outputs))

    validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    validation_set = list(zip(validation_inputs, validation_data[1]))

    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_set = list(zip(test_inputs, test_data[1]))

    return training_set, validation_set, test_set
