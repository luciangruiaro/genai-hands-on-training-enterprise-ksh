from flask import Blueprint, render_template, jsonify
from simple_neuron.simple_neuron_service import (
    get_or_create_neuron,
    train_one_epoch,
    train_full,
    reset_neuron
)

sn_bp = Blueprint("sn", __name__, url_prefix="/neuron")


@sn_bp.route("/visualize")
def visualize():
    neuron = get_or_create_neuron()
    return render_template("simpleNeuron.html", **_neuron_visual_data(neuron))


@sn_bp.route("/train/step")
def train_step():
    result = train_one_epoch()
    neuron = result["neuron"]
    return jsonify(_neuron_response_payload(neuron, result["epoch"]))


@sn_bp.route("/train/full")
def train_full_route():
    result = train_full()  # This also plots evolution
    neuron = result["neuron"]
    return jsonify(_neuron_response_payload(neuron, result["epoch"]))


@sn_bp.route("/reset")
def reset():
    result = reset_neuron()
    neuron = result["neuron"]
    return jsonify(_neuron_response_payload(neuron, result["epoch"]))


# --- Helpers ---
def _neuron_visual_data(neuron):
    return {
        "weights": neuron.weights,
        "noInputs": neuron.no_inputs,
        "activated": neuron.is_activated(),
        "output": neuron.output,
        "outputClassified": neuron.output_classified,
        "biasWeight": neuron.bias_weight,
        "inputValues": neuron.inputs
    }


def _neuron_response_payload(neuron, epoch):
    return {
        "epoch": epoch,
        "weights": neuron.weights,
        "biasWeight": neuron.bias_weight,
        "inputValues": neuron.inputs,
        "output": neuron.output,
        "outputClassified": neuron.output_classified,
        "activated": neuron.is_activated()
    }
