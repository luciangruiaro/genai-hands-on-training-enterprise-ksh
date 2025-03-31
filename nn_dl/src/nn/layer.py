from nn.neuron import Neuron


class Layer:
    def __init__(self, number_of_neurons, input_size):
        self.neurons = [Neuron(input_size) for _ in range(number_of_neurons)]

    def forward(self, inputs):
        return [neuron.output(inputs) for neuron in self.neurons]

    def classify(self, inputs):
        return [neuron.classify(inputs) for neuron in self.neurons]

    def adjust_weights(self, inputs, expected_outputs, predictions):
        for neuron, expected, prediction in zip(self.neurons, expected_outputs, predictions):
            neuron.adjust_weights(inputs, expected, prediction)

    def get_weights(self):
        return [neuron.weights + [neuron.bias_weight] for neuron in self.neurons]
