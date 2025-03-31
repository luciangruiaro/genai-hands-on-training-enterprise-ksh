from simple_neuron.sn_state import SNState


def train_more_epochs(training_data, expected_outputs, s_neuron, epochs):
    """
    Train a single neuron for a given number of epochs.
    Tracks the state (weights, bias) after each full epoch.
    """
    sn_states = [SNState(s_neuron)]  # Initial state
    for _ in range(epochs):
        for inputs, expected in zip(training_data, expected_outputs):
            _train(inputs, expected, s_neuron)
        sn_states.append(SNState(s_neuron))  # Capture post-epoch state
    return sn_states


def _train(input_values, expected, s_neuron):
    """
    Single forward + backward pass for one training pair.
    """
    prediction = s_neuron.output(input_values)
    s_neuron.adjust_weights(input_values, expected, prediction)
