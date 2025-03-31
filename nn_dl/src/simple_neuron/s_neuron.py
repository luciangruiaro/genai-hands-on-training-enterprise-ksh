import logging

from helpers.math_helpers import (
    generate_random_weight,
    gradient,
    gradient_descent,
    sigmoid
)
from config import global_config as cfg

from helpers.logger_config import get_logger

logger = get_logger(__name__)


class SNeuron:
    """
    A simple binary classifier neuron using sigmoid activation.
    """

    def __init__(self, input_size: int):
        self.no_inputs = input_size
        self.weights = [generate_random_weight() for _ in range(self.no_inputs)]
        self.bias = 1.0
        self.bias_weight = 0  # generate_random_weight()
        self.log_creation()

    def input_signal(self, input_values):
        assert len(input_values) == self.no_inputs
        weighted_sum = sum(x * w for x, w in zip(input_values, self.weights))
        return weighted_sum + self.bias * self.bias_weight

    def activation_function(self, input_signal):
        return sigmoid(input_signal)

    def output(self, input_values):
        return self.activation_function(self.input_signal(input_values))

    def classify(self, input_values):
        return 1 if self.output(input_values) >= cfg.SN_OUTPUT_THRESHOLD else 0

    def adjust_weights(self, inputs, expected, prediction):
        for i in range(self.no_inputs):
            grad = gradient(inputs[i], expected, prediction)
            self.weights[i] += gradient_descent(grad)
        self.adjust_bias_weight(expected, prediction)

    def adjust_bias_weight(self, expected, prediction):
        grad = gradient(1, expected, prediction)
        self.bias_weight += gradient_descent(grad)

    def log_creation(self):
        logger.info("Neuron created.")
        self.log_state()

    def log_state(self, epoch=None):
        state_str = f"Weights: {self.weights}, Bias: {self.bias_weight}"
        prefix = f"[Epoch {epoch}] " if epoch is not None else ""
        level = logging.DEBUG if logger.isEnabledFor(logging.DEBUG) else logging.INFO
        logger.log(level, f"{prefix}Neuron state: {state_str}")
