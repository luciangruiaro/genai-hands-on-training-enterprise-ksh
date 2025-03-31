## 🧠 Neural Networks Playground (Python Refactor)

This project is a **modular educational platform** for experimenting with neural networks and simple neurons.  
Originally written in Java and now fully refactored to Python using Flask, this codebase supports:

- 📈 Training and testing simple neurons & multilayer NNs
- 🧪 Visualizing weights, biases, and predictions
- 📊 Plotting neuron evolution and network confidence
- 🧠 Live demos with REST endpoints

---

## 📁 Project Structure

```
neural_networks/
│   .env                       ← Environment variables
│   app.py                    ← Flask app entry point
│
├── api/                      ← REST Controllers
│   ├── nn_controller.py
│   ├── sn_controller.py
│   └── python_api.py
│
├── config/                   ← Central configuration
│   ├── global_config.py
│   └── __init__.py           ← YAML loader, APP_CONFIG available
│
├── helpers/                  ← Utility math/data functions
│   ├── data_helper.py
│   └── math_helpers.py
│
├── my_models/                ← Domain models
│   ├── training_data.py
│   ├── image_data.py
│   └── pair.py
│
├── nn/                       ← Multilayer neural network
│   ├── nn.py
│   ├── neuron.py
│   ├── layer.py
│   ├── nn_service.py
│   ├── train_nn.py
│   └── test_nn.py
│
├── simple_neuron/           ← Single neuron logic
│   ├── s_neuron.py
│   ├── train.py
│   ├── test.py
│   ├── sn_printer.py
│   ├── sn_state.py
│   ├── neuron_visualiser.py
│   └── simple_neuron_service.py
│
├── training_data/           ← Python-generated MNIST data
│   └── generate_training_data.py
│
├── visualizations/          ← Matplotlib visual tools
│   ├── neuron_evolution.py
│   ├── network_evolution.py
│   ├── gradient.py
│   ├── sigmoid.py
│   └── learning_rate.py
│
└── resources/               ← Configuration
    ├── application.yaml
    └── log_config.yaml
```

---

## 🛠 Setup Instructions

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your `.env` file:

```env
APP_CONFIG=resources/application.yaml
FLASK_RUN_HOST=127.0.0.1
FLASK_RUN_PORT=5000
FLASK_DEBUG=true
```

4. Run the app:

```bash
python run.py
```

---

## 🐳 Docker Support

This project supports Docker & Docker Compose out of the box:

### 📦 Build & Run

```bash
# Build the container
docker-compose build

# Run in foreground
docker-compose up

# Run in detached mode
docker-compose up -d

# Stop the container
docker-compose down
```

### 📁 Files

- `Dockerfile` – Builds the Flask app container with Gunicorn
- `docker-compose.yml` – Simplified runner for dev or deployment
- `.dockerignore` – Keeps the image slim and clean

---

## 🧪 Optional Makefile Commands

> If using a Makefile (recommended for DX/CI automation)

```makefile
run:                # Run locally
	python run.py

serve:              # Run with Gunicorn (production)
	gunicorn -w 4 -b 0.0.0.0:5000 wsgi:app

dev:                # Run with Flask debug
	FLASK_ENV=development python run.py

docker-build:       # Build Docker container
	docker-compose build

docker-up:          # Start Docker app
	docker-compose up

docker-up-detached: # Start in background
	docker-compose up -d

docker-down:        # Stop container
	docker-compose down
```

---

## 🚀 REST API Overview

| Endpoint                         | Purpose                         |
|----------------------------------|---------------------------------|
| `/nn/train`                      | Train full NN                   |
| `/nn/test/submit` (POST)         | Submit input for classification |
| `/nn/network-d3`                 | Visualize NN structure          |
| `/sn/visualize/initial`          | View fresh simple neuron        |
| `/sn/visualize/after`            | View trained simple neuron      |
| `/python/generate-training-data` | Generate new MNIST samples      |

---

## 🧰 Configuration Tips

- Use `from config import global_config as cfg` for all constants
- Use `from config import APP_CONFIG` for YAML-based settings
- `.env` and `resources/application.yaml` are both loaded at runtime
