import logging
import json
import os

from simple_neuron.s_neuron import SNeuron
from simple_neuron.sn_state import SNState
from simple_neuron.neuron_visualiser import NeuronVisualiser
from visualizations.neuron_evolution import plot_neuron_states
from config import global_config as cfg

from helpers.logger_config import get_logger

logger = get_logger(__name__)

# --- Global state ---
_sn = None
_sn_states = []
_sn_epoch_counter = 0

# Hardcoded default input
_new_input = [1, 0, 0]

# Simple binary dataset
_training_data_set = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 1]
]
_expected_outputs = [0, 0, 0, 0, 1]


def get_or_create_neuron():
    """Returns the current neuron instance or initializes a new one."""
    global _sn
    if _sn is None:
        _sn = SNeuron(cfg.SN_INPUT_SIZE)
    return NeuronVisualiser(_sn, _new_input)


def train_one_epoch():
    global _sn_epoch_counter, _sn
    get_or_create_neuron()

    for inputs, expected in zip(_training_data_set, _expected_outputs):
        prediction = _sn.output(inputs)
        _sn.adjust_weights(inputs, expected, prediction)

    _sn_epoch_counter += 1
    _sn.log_state(epoch=_sn_epoch_counter)
    return {
        "epoch": _sn_epoch_counter,
        "neuron": NeuronVisualiser(_sn, _new_input)
    }


def train_full():
    global _sn, _sn_states, _sn_epoch_counter
    get_or_create_neuron()

    _sn_epoch_counter = 0
    _sn_states = [SNState(_sn)]

    for _ in range(cfg.SN_EPOCHS):
        for inputs, expected in zip(_training_data_set, _expected_outputs):
            prediction = _sn.output(inputs)
            _sn.adjust_weights(inputs, expected, prediction)
        _sn_epoch_counter += 1
        _sn_states.append(SNState(_sn))

    # Save training history
    os.makedirs("resources", exist_ok=True)
    with open("resources/neuronModel.json", "w") as f:
        json.dump([s.__dict__ for s in _sn_states], f, indent=2)

    # Plot learning
    plot_neuron_states([s.__dict__ for s in _sn_states])

    # Log only last state unless debug mode
    if logging.getLogger().isEnabledFor(logging.DEBUG):
        for i, state in enumerate(_sn_states):
            _sn.log_state(epoch=i)
    else:
        _sn.log_state(epoch=_sn_epoch_counter)

    return {
        "epoch": _sn_epoch_counter,
        "neuron": NeuronVisualiser(_sn, _new_input)
    }


def reset_neuron():
    """Reset neuron and counters."""
    global _sn, _sn_states, _sn_epoch_counter
    _sn = SNeuron(cfg.SN_INPUT_SIZE)
    _sn_states = []
    _sn_epoch_counter = 0
    return {
        "epoch": 0,
        "neuron": NeuronVisualiser(_sn, _new_input)
    }
