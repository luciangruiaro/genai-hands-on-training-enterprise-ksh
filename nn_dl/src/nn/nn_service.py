import os
import json
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from my_models.training_data import get_training_data, get_default_inference_sample
from nn.nn import NeuralNetwork
from config import global_config as cfg
from helpers.logger_config import get_logger

logger = get_logger(__name__)

# --- Global State ---
_nn_instance: Optional[NeuralNetwork] = None
_current_epoch: int = 0
_training_history = []


def get_or_create_nn():
    global _nn_instance
    if _nn_instance is None:
        _nn_instance = load_model_from_file()
        if _nn_instance:
            logger.info("Loaded neural network from saved model.")
        else:
            _nn_instance = NeuralNetwork.create_from_config(cfg)
            logger.info("Initialized new neural network instance from config.")
    return _nn_instance


def load_model_from_file():
    path = cfg.APP_CONFIG["nn"]["json_model"]
    if not os.path.exists(path):
        logger.warning("Model file not found, skipping load.")
        return None

    try:
        with open(path, "r") as f:
            data = json.load(f)

        model = NeuralNetwork(
            input_size=data["config"]["input_neurons"],
            hidden_layers=data["config"]["hidden_layers"],
            neurons_per_hidden_layer=data["config"]["neurons_per_hidden_layer"],
            output_size=data["config"]["output_neurons"]
        )

        weights = data["weights"]

        # Set weights for hidden layers
        for layer, layer_weights in zip(model.hidden_layers, weights["hidden"]):
            for neuron, neuron_weights in zip(layer.neurons, layer_weights):
                neuron.weights = neuron_weights[:-1]
                neuron.bias_weight = neuron_weights[-1]

        # Set weights for output layer
        for neuron, neuron_weights in zip(model.output_layer.neurons, weights["output"]):
            neuron.weights = neuron_weights[:-1]
            neuron.bias_weight = neuron_weights[-1]

        return model

    except Exception as e:
        logger.error(f"Failed to load model from file: {e}")
        return None


def save_model_to_file(model: NeuralNetwork, suffix: str = ""):
    try:
        os.makedirs("resources", exist_ok=True)
        model_data = {
            "config": {
                "input_neurons": cfg.NN_INPUT_NEURONS,
                "hidden_layers": cfg.NN_NO_HIDDEN_LAYERS,
                "neurons_per_hidden_layer": cfg.NN_NO_NEURONS_PER_HIDDEN_LAYERS,
                "output_neurons": cfg.NN_OUTPUT_NEURONS,
            },
            "weights": model.get_all_weights()
        }

        base_path = cfg.APP_CONFIG["nn"]["json_model"]
        if suffix:
            name, ext = os.path.splitext(base_path)
            path = f"{name}{suffix}{ext}"
        else:
            path = base_path

        with open(path, "w") as f:
            json.dump(model_data, f, indent=2)
        logger.info(f"💾 Model saved to {path}")
    except Exception as e:
        logger.error(f"❌ Failed to save model to file: {e}")


def reset_network():
    global _nn_instance, _current_epoch, _training_history
    _nn_instance = None
    _current_epoch = 0
    _training_history = []
    logger.info("🔄 Neural network state has been reset.")
    return get_or_create_nn()


def get_current_epoch():
    logger.info("🔍 Returning current epoch. {}".format(_current_epoch))
    return _current_epoch


def train_one_epoch():
    global _current_epoch, _training_history

    logger.info("🚀 Starting one epoch of training...")
    model = get_or_create_nn()
    dataset = get_training_data()

    for i, sample in enumerate(dataset):
        model.train(sample.input, sample.output)
        if i % 10000 == 0:
            logger.debug(f"🔄 Trained on sample {i + 1}/{len(dataset)}")

    _current_epoch += 1
    weights = model.get_all_weights()
    _training_history.append(weights)

    logger.info(f"✅ Epoch {_current_epoch} completed and weights updated.")

    # Save model after single epoch
    try:
        save_model_to_file(model)
        logger.info(f"💾 Model state saved after epoch {_current_epoch}.")
    except Exception as e:
        logger.error(f"❌ Failed to save model after epoch {_current_epoch}: {e}")

    return model


def train_full_epochs():
    model = get_or_create_nn()
    dataset = get_training_data()
    epochs = cfg.APP_CONFIG["nn"]["epochs"]
    global _current_epoch

    weight_evolution = []

    logger.info(f"🚀 Starting full training for {epochs} epochs...")

    for epoch in range(epochs):
        _current_epoch += 1
        for sample in dataset:
            model.train(sample.input, sample.output)

        weights = model.get_all_weights()
        weight_evolution.append(weights)
        logger.debug(f"✅ Epoch {epoch + 1}/{epochs} completed.")

        # Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_model_to_file(model, suffix=f"_epoch_{epoch + 1}")
            logger.info(f"💾 Model checkpoint saved at epoch {epoch + 1}.")
            plot_weight_evolution(weight_evolution, suffix=f"_epoch_{epoch + 1}")
            logger.info(f"📊 Plot checkpoint saved at epoch {epoch + 1}.")

    # Final weight evolution save
    os.makedirs("resources", exist_ok=True)
    weight_file_path = "resources/weight_evolution.json"
    with open(weight_file_path, "w") as f:
        json.dump(weight_evolution, f, indent=2)
        logger.info(f"📈 Full weight evolution saved to {weight_file_path}.")

    # Final model + plots
    save_model_to_file(model)
    logger.info("✅ Final model saved after full training.")
    plot_weight_evolution(weight_evolution)
    logger.info("✅ Final plots saved after full training.")

    return model


def infer_current_state(model=None, force_input=None):
    model = model or get_or_create_nn()

    if force_input is not None:
        logger.info("🔎 Inference using provided input vector.")
        model.forward(force_input)
    elif model.inputs_vector and any(model.inputs_vector):
        logger.info("🔎 Inference using model.inputs_vector.")
        model.forward(model.inputs_vector)
    else:
        sample = get_default_inference_sample()
        logger.info(f"🔍 Inference on default digit {cfg.NN_DEFAULT_DIGIT}")
        model.forward(sample.input)

    return {
        "input": model.inputs_vector,
        "hidden": get_hidden_layer_outputs(model),
        "output": get_output_layer_outputs(model)
    }


def get_hidden_layer_outputs(nn) -> str:
    if not hasattr(nn, "hidden_layer_outputs") or nn.hidden_layer_outputs is None:
        return ""
    flattened = [val for layer_out in nn.hidden_layer_outputs for val in layer_out]
    return ",".join(map(str, flattened))


def get_output_layer_outputs(nn) -> str:
    if not hasattr(nn, "output_layer_outputs") or nn.output_layer_outputs is None:
        return ""
    return ",".join(map(str, nn.output_layer_outputs))


def plot_weight_evolution(weight_history, suffix=""):
    import matplotlib.pyplot as plt
    import matplotlib

    os.makedirs("resources/plots", exist_ok=True)
    num_epochs = len(weight_history)
    max_legend_items = 20

    try:
        for layer_idx in range(len(weight_history[0]["hidden"])):
            layer_weights = [weight_history[epoch]["hidden"][layer_idx] for epoch in range(num_epochs)]
            plt.figure(figsize=(12, 6))
            num_neurons = len(layer_weights[0])

            for neuron_idx in range(num_neurons):
                neuron_weights = [layer[neuron_idx] for layer in layer_weights]
                weights_per_input = list(zip(*neuron_weights))

                for input_idx, input_weight_trace in enumerate(weights_per_input):
                    # Only plot every second input for readability
                    if input_idx % 2 == 0:
                        label = f"Neuron {neuron_idx} - W{input_idx + 1}"
                        plt.plot(input_weight_trace, label=label)

            plt.title(f"Hidden Layer {layer_idx + 1} - Weight Evolution")
            plt.xlabel("Epochs")
            plt.ylabel("Weight Value")
            plt.grid(True)
            plt.legend(loc="upper right", fontsize="small", ncol=2)
            plt.tight_layout()
            filename = f"resources/plots/hidden_layer_{layer_idx + 1}_weights{suffix}.png"
            plt.savefig(filename)
            plt.close()
            logger.info(f"✅ Saved plot: {filename}")

        # Output layer
        output_layer_weights = [weight_history[epoch]["output"] for epoch in range(num_epochs)]
        plt.figure(figsize=(12, 6))
        num_output_neurons = len(output_layer_weights[0])

        for neuron_idx in range(num_output_neurons):
            neuron_weights = [epoch_weights[neuron_idx] for epoch_weights in output_layer_weights]
            weights_per_input = list(zip(*neuron_weights))

            for input_idx, input_weight_trace in enumerate(weights_per_input):
                # Again, skip every other weight to reduce clutter
                if input_idx % 2 == 0:
                    label = f"Output Neuron {neuron_idx} - W{input_idx + 1}"
                    plt.plot(input_weight_trace, label=label)

        plt.title("Output Layer - Weight Evolution")
        plt.xlabel("Epochs")
        plt.ylabel("Weight Value")
        plt.grid(True)
        plt.legend(loc="upper right", fontsize="small", ncol=2)
        plt.tight_layout()
        output_plot = f"resources/plots/output_layer_weights{suffix}.png"
        plt.savefig(output_plot)
        plt.close()
        logger.info(f"✅ Saved plot: {output_plot}")

    except Exception as e:
        logger.error(f"❌ Error during weight evolution plotting: {e}")
