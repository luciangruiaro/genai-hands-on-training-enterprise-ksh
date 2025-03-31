import numpy as np
from nn.layer import Layer
from config import global_config as cfg


class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, neurons_per_hidden_layer, output_size):
        self.input_size = input_size
        self.hidden_layers_config = hidden_layers
        self.neurons_per_hidden_layer = neurons_per_hidden_layer
        self.output_size = output_size

        self.inputs_vector = [0.0] * input_size
        self.hidden_layers = []
        self.hidden_layer_outputs = None
        self.output_layer_outputs = None

        # Build hidden layers
        for i in range(hidden_layers):
            input_dim = input_size if i == 0 else neurons_per_hidden_layer
            self.hidden_layers.append(Layer(neurons_per_hidden_layer, input_dim))

        # Build output layer
        self.output_layer = Layer(output_size, neurons_per_hidden_layer)

    def forward(self, inputs):
        self.inputs_vector = inputs
        output = inputs
        self.hidden_layer_outputs = []

        for layer in self.hidden_layers:
            output = layer.forward(output)
            self.hidden_layer_outputs.append(output)

        self.output_layer_outputs = self.output_layer.forward(output)
        return self.output_layer_outputs

    def classify(self, inputs):
        output = self.forward(inputs)
        return [1 if val >= cfg.NN_OUTPUT_THRESHOLD else 0 for val in output]

    def train(self, inputs, expected_outputs):
        prediction = self.forward(inputs)
        self.output_layer.adjust_weights(self.hidden_layer_outputs[-1], expected_outputs, prediction)

    def get_all_weights(self):
        return {
            "hidden": [layer.get_weights() for layer in self.hidden_layers],
            "output": self.output_layer.get_weights()
        }

    @staticmethod
    def create_from_config(cfg):
        return NeuralNetwork(
            input_size=cfg.NN_INPUT_NEURONS,
            hidden_layers=cfg.NN_NO_HIDDEN_LAYERS,
            neurons_per_hidden_layer=cfg.NN_NO_NEURONS_PER_HIDDEN_LAYERS,
            output_size=cfg.NN_OUTPUT_NEURONS
        )
