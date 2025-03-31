import math
import random
from config import global_config as cfg


# -----------------------------
# Activation Functions + Derivatives
# -----------------------------

def sigmoid(x):
    """Sigmoid activation"""
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    """Derivative of sigmoid applied to raw input (z)"""
    s = sigmoid(x)
    return s * (1 - s)


def sigmoid_derivative_from_output(output):
    """Derivative of sigmoid using its output (for reuse efficiency)"""
    return output * (1 - output)


def relu(x):
    return max(0, x)


def relu_derivative(x):
    return 1 if x > 0 else 0


def tanh(x):
    return math.tanh(x)


def tanh_derivative(x):
    return 1 - math.tanh(x) ** 2


# Registry
activation_functions = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "relu": (relu, relu_derivative),
    "tanh": (tanh, tanh_derivative)
}


def get_activation_function(name: str):
    """Returns (activation_fn, derivative_fn)"""
    if name not in activation_functions:
        raise ValueError(f"Unsupported activation function: {name}")
    return activation_functions[name]


# -----------------------------
# Gradient and Loss Functions
# -----------------------------

def raw_loss(expected, predicted):
    return expected - predicted


def mean_squared_loss(expected, predicted):
    return 0.5 * (expected - predicted) ** 2


def cross_entropy_loss(expected, predicted):
    return -sum(e * math.log(p) for e, p in zip(expected, predicted))


def gradient(input_value, expected, predicted):
    """Generic gradient using sigmoid-based delta"""
    return input_value * raw_loss(expected, predicted) * sigmoid_derivative_from_output(predicted)


def gradient_vector(inputs, expected, predictions):
    assert len(inputs) == len(expected) == len(predictions)
    return [gradient(i, e, p) for i, e, p in zip(inputs, expected, predictions)]


def gradient_descent(value, learning_rate=None):
    """Apply learning rate to gradient value"""
    lr = learning_rate if learning_rate is not None else cfg.SN_LEARNING_RATE
    return lr * value


def gradient_descent_vector(gradients, learning_rate=None):
    return [gradient_descent(g, learning_rate) for g in gradients]


# -----------------------------
# Delta Calculations (Backprop)
# -----------------------------

def compute_delta_for_output(expected, prediction):
    """Used in output layer neurons"""
    return raw_loss(expected, prediction) * sigmoid_derivative_from_output(prediction)


def compute_delta_for_hidden(activated_input, next_layer_deltas, next_layer_weights):
    """Used in hidden layer neurons"""
    weighted_sum = sum(d * w for d, w in zip(next_layer_deltas, next_layer_weights))
    return weighted_sum * sigmoid_derivative_from_output(activated_input)


# -----------------------------
# Utility Math Functions
# -----------------------------

def softmax(logits):
    exps = [math.exp(x) for x in logits]
    sum_exps = sum(exps)
    if sum_exps == 0:
        raise ZeroDivisionError("Denominator is zero in softmax computation.")
    return [x / sum_exps for x in exps]


def get_max_index(values):
    return max(range(len(values)), key=lambda i: values[i])


def generate_random_weight():
    return random.uniform(-0.5, 0.5)


def flatten_matrix(matrix):
    return [value for row in matrix for value in row]
