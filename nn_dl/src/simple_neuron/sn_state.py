class SNState:
    def __init__(self, s_neuron):
        self.weights = s_neuron.weights[:]
        self.bias_weight = s_neuron.bias_weight
