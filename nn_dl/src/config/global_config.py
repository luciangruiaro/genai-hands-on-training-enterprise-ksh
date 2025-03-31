import os
from config import APP_CONFIG

# ---- PATHS ----
DYNAMIC_PATH = os.getcwd()

RES_PYTHON_PATH = os.path.join(DYNAMIC_PATH, APP_CONFIG["paths"]["base"])
RES_JS_PATH = os.path.join(DYNAMIC_PATH, APP_CONFIG["paths"]["static"]["js"])
RES_CSS_PATH = os.path.join(DYNAMIC_PATH, APP_CONFIG["paths"]["static"]["css"])
PYTHON_VENV_COMMAND = os.path.join(DYNAMIC_PATH, APP_CONFIG["paths"]["python_venv"])

# ---- SIMPLE NEURON CONFIG ----
SN_OUTPUT_THRESHOLD = APP_CONFIG["sn"]["output_threshold"]
SN_LEARNING_RATE = APP_CONFIG["sn"]["learning_rate"]
SN_INPUT_SIZE = APP_CONFIG["sn"]["input_size"]
SN_EPOCHS = APP_CONFIG["sn"]["epochs"]

# ---- NEURAL NETWORK CONFIG ----
NN_INPUT_NEURONS = APP_CONFIG["nn"]["input_neurons"]
NN_OUTPUT_NEURONS = APP_CONFIG["nn"]["output_neurons"]
NN_OUTPUT_THRESHOLD = APP_CONFIG["nn"]["output_threshold"]
NN_NO_HIDDEN_LAYERS = APP_CONFIG["nn"]["no_hidden_layers"]
NN_NO_NEURONS_PER_HIDDEN_LAYERS = APP_CONFIG["nn"]["neurons_per_hidden_layer"]
NN_LEARNING_RATE = APP_CONFIG["nn"]["learning_rate"]
NN_EPOCHS = APP_CONFIG["nn"]["epochs"]
NN_ACTIVATION = APP_CONFIG["nn"]["activation"]
NN_TRAINING_SET_SIZE = APP_CONFIG["nn"]["training_set_size"]
NN_DEFAULT_DIGIT = APP_CONFIG["nn"]["default_digit"]

NN_JSON_MODEL = os.path.join(DYNAMIC_PATH, APP_CONFIG["nn"]["json_model"])
