from flask import Blueprint, request, jsonify, render_template
from helpers.data_helper import (array_to_csv)
from nn.nn_service import *
from config import global_config as cfg

nn_bp = Blueprint('nn', __name__, url_prefix='/nn')


@nn_bp.route("/network-d3")
def view_network_d3():
    model = get_or_create_nn()

    # Run inference with current weights and current test image
    infer_current_state(model)

    input_values = model.inputs_vector
    hidden_values = get_hidden_layer_outputs(model)
    output_values = get_output_layer_outputs(model)

    hidden_neurons_total = cfg.NN_NO_HIDDEN_LAYERS * cfg.NN_NO_NEURONS_PER_HIDDEN_LAYERS
    output_neurons_total = cfg.NN_OUTPUT_NEURONS

    return render_template("network-d3.html", **{
        "noInputNeurons": cfg.NN_INPUT_NEURONS,
        "noHiddenLayers": cfg.NN_NO_HIDDEN_LAYERS,
        "noNeuronsPerHiddenLayer": cfg.NN_NO_NEURONS_PER_HIDDEN_LAYERS,
        "noOutputNeurons": cfg.NN_OUTPUT_NEURONS,
        "inputNeuronValues": array_to_csv(input_values),
        "hiddenNeuronValues": hidden_values or ",".join(["0.0"] * hidden_neurons_total),
        "outputNeuronValues": output_values or ",".join(["0.0"] * output_neurons_total),
        "currentEpoch": get_current_epoch()
    })


@nn_bp.route("/reset", methods=["POST"])
def reset_route():
    model = reset_network()
    return jsonify({
        "message": "Network reset.",
        "input": model.inputs_vector
    })


@nn_bp.route("/train/epoch", methods=["POST"])
def train_step_route():
    train_one_epoch()
    return jsonify({
        "epoch": get_current_epoch(),
        **infer_current_state()
    })


@nn_bp.route("/train/full", methods=["POST"])
def train_full_route():
    train_full_epochs()
    return jsonify({
        "epoch": get_current_epoch(),
        **infer_current_state()
    })


@nn_bp.route("/test/submit", methods=["POST"])
def test_input_route():
    data = request.get_json()
    input_values = data.get("input", [])

    logger.info(f"🔎 Received input with {len(input_values)} values.")

    if not input_values or len(input_values) != cfg.NN_INPUT_NEURONS:
        logger.warning("❌ Invalid input received — wrong length or empty.")
        return jsonify({"error": "Invalid input length"}), 400

    model = get_or_create_nn()
    model.inputs_vector = input_values  # 🟢 fix: update model state
    logger.info("🧠 Using existing model for inference.")

    prediction = model.forward(input_values)
    infer_current_state(model)

    logger.debug(f"📥 Input preview (first 10): {input_values[:10]}")
    logger.debug(f"📤 Raw output: {prediction}")

    predicted_class = prediction.index(max(prediction))
    logger.info(f"✅ Predicted digit: {predicted_class}")

    return jsonify({
        "input": input_values,
        "raw_output": prediction,
        "predicted_digit": predicted_class
    })


@nn_bp.route("/test")
def view_test_page():
    return render_template("test.html")


@nn_bp.route("/visualizeInputImage")
def vizualize_input_image():
    return render_template("visualizeInputImage.html")
