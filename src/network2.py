"""
network2.py

Adapted for educational purposes.
"""

# Standard libraries
import json
import random
import sys
from typing import Protocol, Optional

# Third party libraries
import numpy as np

# Local libraries
from utils import sigmoid, sigmoid_prime, vectorized_result
from mnist_loader import TrainingSample, EvalSample

class CostFunction(Protocol):
    @staticmethod
    def fn(a: np.ndarray, y: np.ndarray) -> np.float64: ...

    @staticmethod
    def delta(z: np.ndarray, a: np.ndarray, y: np.ndarray) -> np.ndarray: ...


class QuadraticCost:

    @staticmethod
    def fn(a: np.ndarray, y: np.ndarray) -> np.float64:
        """Return the cost associated with an output `a` and desired output `y`. """
        return np.float64(0.5 * np.linalg.norm(a - y) ** 2)

    @staticmethod
    def delta(z: np.ndarray, a: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the error delta from the output layer."""
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost:

    @staticmethod
    def fn(a: np.ndarray, y: np.ndarray) -> np.float64:
        """
        Return the cost associated with an output `a` and desired output
        `y`.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both `a` and `y` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).
        """
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z: np.ndarray, a: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Return the error delta from the output layer.  Note that the
        parameter `z` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.
        """
        return (a - y)

class Network:

    def __init__(self, sizes: list[int], cost: CostFunction = CrossEntropyCost):
        """
        The list `sizes` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron. The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes;
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        """
        Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron. Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """
        Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison. It
        will usually be better to use the default weight initializer
        instead.
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a: np.ndarray) -> np.ndarray:
        """Return the output of the network if `a` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(w @ a + b)
        return a

    def SGD(
        self,
        training_data: list[TrainingSample],
        epochs: int,
        mini_batch_size: int,
        eta: float,
        lmbda: float = 0.0,
        evaluation_data: Optional[list[EvalSample]] = None,
        monitor_evaluation_cost=False,
        monitor_evaluation_accuracy=False,
        monitor_training_cost=False,
        monitor_training_accuracy=False
    ) -> tuple[list, list, list, list]:
        """
        Train the neural network using mini‑batch stochastic gradient descent (SGD).

        Parameters:
            training_data : list[TrainingSample]
                A list of (sample, label) tuples. Each `sample` is an array of shape
                (n_input, 1) and each `label` is a one‑hot array

            epochs : int
                The number of full passes over the training dataset.

            mini_batch_size : int
                The size of each mini‑batch. At each step, the training data is split
                into consecutive slices of this length (the last batch may be smaller).

            eta : float
                The learning rate (step size) for gradient descent updates.

            lmbda : float
                Regularization parameter

            evaluation_data : Optional[list[EvalSample]]
                If provided, after each epoch the network will be evaluated on this data
                (a list of (input, label) tuples) and the accuracy printed. If omitted,
                only epoch completion is reported.
        """
        n = len(training_data)
        n_eval = len(evaluation_data) if evaluation_data else 0
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        for j in range(epochs):
            # Randomly shuffle `training_data`.
            random.shuffle(training_data)

            # Partition into mini‑batches of size `mini_batch_size`.
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]

            # For each mini‑batch, adjust weights and biases.
            for mini_batch in mini_batches:
                self.apply_mini_batch_update(mini_batch, eta, lmbda, len(training_data))

            print(f"Epoch {j} training complete")
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print(f"Cost on training data: {cost}")
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print(f"Accuracy on training data: {accuracy} / {n} ({(100 * accuracy / n):.2f}%)")
            if monitor_evaluation_cost and evaluation_data:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print(f"Cost on evaluation data: {cost}")
            if monitor_evaluation_accuracy and evaluation_data:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print(f"Accuracy on evaluation data: {accuracy} / {n_eval} ({(100 * accuracy / n_eval):.2f}%)")

        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def apply_mini_batch_update(
        self,
        mini_batch: list[TrainingSample],
        eta: float,
        lmbda: float,
        n: int,
        vectorized: bool = True
    ) -> None:
        """
        Update network parameters by applying gradient descent on one mini‑batch.

        Parameters:
            mini_batch : list[TrainingSample]
                A list of (sample, label) tuples.

            eta : float
                The learning rate.

            lmbda : float
                The regularization parameter

            n : int
                The total size of the training data set.

            vectorized : bool
                If True (default), compute gradients in one shot for the entire mini‑batch by
                stacking samples as columns and running a single call to `backprop(...)`.
                If False, compute each sample’s gradient individually and sum them.
        """
        if vectorized:
            samples, labels = zip(*mini_batch)
            samples = np.hstack(samples)
            labels = np.hstack(labels)

            bias_grad, weight_grad = self.backprop(samples, labels)
        else:
            bias_grad = [np.zeros(b.shape) for b in self.biases]
            weight_grad = [np.zeros(w.shape) for w in self.weights]

            for sample, label in mini_batch:
                delta_bias_grad, delta_weight_grad = self.backprop(sample, label)

                # Accumulate bias and weights gradients
                for i in range(self.num_layers - 1):
                    bias_grad[i] += delta_bias_grad[i]
                    weight_grad[i] += delta_weight_grad[i]

        # Scaling (averaging)
        k1 = 1 - eta * lmbda / n
        k2 = eta / len(mini_batch)
        for i in range(self.num_layers - 1):
            self.biases[i] -= k2 * bias_grad[i]
            self.weights[i] = k1 * self.weights[i] - k2 * weight_grad[i]

    def backprop(self, x: np.ndarray, y: np.ndarray) -> tuple[list, list]:
        """
        Compute the gradient of the cost function with respect to the
        network's weights and biases, for either a single input sample
        or an entire mini-batch of samples.

        Parameters:
            x : np.ndarray
                The input vector (shape: (n_input, 1)) or a matrix of inputs
                (shape: (n_input, m)) where each column is a sample.
            y : np.ndarray
                The target output vector (shape: (n_output, 1)) or a matrix
                of target outputs (shape: (n_output, m)).

        Returns:
            tuple (bias_grad, weight_grad):
                - bias_grad: List of arrays representing the gradients of the cost
                    with respect to each layer's biases.
                - weight_grad: List of arrays representing the gradients of the cost
                    with respect to each layer's weights.

            If the input is a mini-batch, each array in both returned lists contains
            the sum of the gradients over all samples in the batch. If the input is
            a single sample, the arrays contain the gradient for that sample alone.

        Note:
            Scaling (e.g., averaging) should be handled outside this function
            if needed, typically during the parameter update step.
        """

        bias_grad = [np.zeros(b.shape) for b in self.biases]
        weight_grad = [np.zeros(w.shape) for w in self.weights]

        activations = [x]   # list to store all the activations, layer by layer
        zs = []             # list to store all the weighted inputs z, layer by layer

        # feedforward
        a = activations[0]
        for b, w in zip(self.biases, self.weights):
            z = w @ a + b
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)

        # output error
        delta = self.cost.delta(zs[-1], activations[-1], y)

        # gradients for the last layer
        bias_grad[-1] = np.sum(delta, axis=1, keepdims=True)
        weight_grad[-1] = delta @ activations[-2].T

        # backpropagate the error
        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = (self.weights[-l+1].T @ delta) * sigmoid_prime(z)
            bias_grad[-l] = np.sum(delta, axis=1, keepdims=True)
            weight_grad[-l] = delta @ activations[-l-1].T

        return bias_grad, weight_grad

    def accuracy(self, data: list, convert: bool = False):
        """
        Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.

        The flag `convert` should be set to `False` if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results `y` are
        represented in the different data sets. In particular, it
        flags whether we need to convert between the different
        representations. It may seem strange to use different
        representations for the different data sets. Why not use the
        same representation for all three data sets? It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up. More details on the
        representations can be found in `mnist_loader.load_data`.
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]

        return sum(int(x == y) for (x, y) in results)

    def total_cost(self, data: list, lmbda: float, convert: bool = False):
        """
        Return the total cost for the data set `data`. The flag `convert`
        should be set to `False` if the data set is the training data
        (the usual case), and to `True` if the data set is the validation
        or test data. See comments on the similar (but reversed) convention
        for the `accuracy` method, above.
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)

        # L2 regularization
        cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file `filename`."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": self.cost.__class__.__name__}
        with open(filename, "w") as f:
            json.dump(data, f)


def load(filename):
    """
    Load a neural network from the file `filename`.
    Returns an instance of Network.
    """
    with open(filename, "r") as f:
        data = json.load(f)
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net
