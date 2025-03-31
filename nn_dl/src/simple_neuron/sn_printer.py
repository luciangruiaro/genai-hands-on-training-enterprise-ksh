import logging
from helpers.math_helpers import mean_squared_loss, gradient, gradient_descent

logger = logging.getLogger(__name__)


def print_neuron_state(s_neuron, expected=None, prediction=None):
    logger.info("---------- Neuron state ----------")
    logger.info(f"Weights: {s_neuron.weights}")
    logger.info(f"Bias Weight: {s_neuron.bias_weight}")
    logger.info("----------------------------------")
    if expected is not None and prediction is not None:
        loss = mean_squared_loss(expected, prediction)
        grad = gradient(expected, prediction, prediction)
        descent = gradient_descent(grad)
        logger.info(f"Loss: {loss}")
        logger.info(f"Gradient: {grad}")
        logger.info(f"Gradient Descent: {descent}")
        logger.info(f"Activation Output: {prediction}")


def print_neuron_creation(s_neuron):
    logger.info(f"Neuron created with {s_neuron.no_inputs} inputs.")
