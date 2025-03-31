from helpers.math_helpers import (
    generate_random_weight,
    gradient_descent,
    get_activation_function
)
from config import global_config as cfg


class Neuron:
    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = [generate_random_weight() for _ in range(input_size)]
        self.bias_weight = generate_random_weight()
        self.bias = 1.0

        # Load activation function + its derivative from config
        self.activation_fn, self.activation_derivative = get_activation_function(cfg.NN_ACTIVATION)

    def input_signal(self, input_values):
        assert len(input_values) == self.input_size
        total = sum(x * w for x, w in zip(input_values, self.weights))
        return total + self.bias * self.bias_weight

    def output(self, input_values):
        z = self.input_signal(input_values)
        return self.activation_fn(z)

    def classify(self, input_values):
        return 1 if self.output(input_values) >= cfg.NN_OUTPUT_THRESHOLD else 0

    def adjust_weights(self, inputs, expected, prediction):
        z = self.input_signal(inputs)
        derivative = self.activation_derivative(z)
        error = expected - prediction

        for i in range(self.input_size):
            grad = inputs[i] * error * derivative
            self.weights[i] += gradient_descent(grad)

        self.adjust_bias(error, derivative)

    def adjust_bias(self, error, derivative):
        grad = 1 * error * derivative
        self.bias_weight += gradient_descent(grad)
