"""
network.py

A simple implementation of a feedforward neural network based on the
code from Chapters 1 and 2 of "Neural Networks and Deep Learning" by Michael Nielsen.

This module defines:
- `Network`: a class for creating, training (via SGD), and evaluating a fully-connected
  neural network with sigmoid activations.
- Utility functions `sigmoid` and `sigmoid_prime` for the activation and its derivative.

The implementation follows the pedagogical examples in the book, including:
- Random Gaussian initialization of weights and biases (mean 0, variance 1).
- Mini-batch stochastic gradient descent (SGD) with backpropagation.
- Evaluation routine for measuring accuracy on test data.

Adapted for educational purposes.
"""

import numpy as np
import random
from utils import sigmoid, sigmoid_prime
from typing import Optional
from mnist_loader import TrainingSample, EvalSample

class Network:

    def __init__(self, sizes: list[int]):
        """
        The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron. The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes;
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

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
        test_data: Optional[list[EvalSample]] = None
    ) -> None:
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

            test_data : Optional[list[EvalSample]]
                If provided, after each epoch the network will be evaluated on this data—
                a list of (input, label) tuples—and the accuracy printed. If omitted,
                only epoch completion is reported.
        """
        n = len(training_data)

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
                self.apply_mini_batch_update(mini_batch, eta)

            if test_data:
                accuracy = self.evaluate(test_data)
                n_test = len(test_data)
                print(f"Epoch {j}: {accuracy} / {n_test} ({(100 * accuracy / n_test):.2f}%)")
            else:
                print(f"Epoch {j} complete")

    def apply_mini_batch_update(
        self,
        mini_batch: list[TrainingSample],
        eta: float,
        vectorized: bool = True
    ) -> None:
        """
        Update network parameters by applying gradient descent on one mini‑batch.

        Parameters:
            mini_batch : list[TrainingSample]
                A list of (sample, label) tuples.

            eta : float
                The learning rate.

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
        k = eta / len(mini_batch)
        for i in range(self.num_layers - 1):
            self.biases[i] -= k * bias_grad[i]
            self.weights[i] -= k * weight_grad[i]

    def backprop(self, X: np.ndarray, Y: np.ndarray) -> tuple[list, list]:
        """
        Compute the gradient of the cost function with respect to the
        network's weights and biases, for either a single input sample
        or an entire mini-batch of samples.

        Parameters:
            X : np.ndarray
                The input vector (shape: (n_input, 1)) or a matrix of inputs
                (shape: (n_input, m)) where each column is a sample.
            Y : np.ndarray
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

        activations = [X]   # list to store all the activations, layer by layer
        zs = []             # list to store all the weighted inputs z, layer by layer

        # feedforward
        a = activations[0]
        for b, w in zip(self.biases, self.weights):
            z = w @ a + b
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)

        # output error
        delta = self.cost_derivative(activations[-1], Y) * sigmoid_prime(zs[-1])

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

    def evaluate(self, test_data: list[EvalSample]) -> int:
        """
        Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (output_activations - y)
