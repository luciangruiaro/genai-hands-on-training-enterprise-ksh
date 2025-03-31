class NeuronVisualiser:
    """
    Data wrapper for a trained or untrained neuron
    to feed values to the GUI.
    """

    def __init__(self, s_neuron, inputs):
        self.no_inputs = s_neuron.no_inputs
        self.weights = s_neuron.weights
        self.bias_weight = s_neuron.bias_weight
        self.inputs = inputs

        self.output = s_neuron.output(inputs)
        self.output_classified = s_neuron.classify(inputs)
        self.activated = self.output >= 0.5

    def is_activated(self):
        return self.activated
