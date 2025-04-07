# 🐍 MultiMode Python Web Application Template

A flexible, modular Python web app template that lets you switch between **Flask** and **FastAPI**, run locally or via
Docker, and apply clean architecture practices. Designed for scalability, teaching, and rapid prototyping.

---

## 🚀 Features

- ✅ Flask or FastAPI support (switch via config)
- ✅ Run locally or in Docker
- ✅ YAML-based central configuration
- ✅ Unified logging (console + file)
- ✅ Modular service-oriented architecture
- ✅ Autocreates `data/` and `logs/` folders

---

## 📆 Folder Structure

```
.
├── api/                # Flask and FastAPI route handlers
├── config/             # YAML and config loader
├── core/               # App factory and bootstrap logic
├── helpers/            # Logging and utilities
├── templates/          # HTML templates (Flask)
├── static/             # Static assets (Flask)
├── data/               # Input/output data (local mode)
├── logs/               # Log files (local or mounted in Docker)
├── Dockerfile
├── docker-compose.yml
├── run.py              # Main entry point
├── wsgi.py             # WSGI entry point for gunicorn
└── requirements.txt
```

---

## 🛠️ Requirements

- Python 3.10+
- Optional: Docker + Docker Compose
- Optional (WSL/Linux/macOS): `make`

---

## 🧪 Running the App

### ▶️ Local Mode (No Docker)

Use `run.py` directly — supports both Flask and FastAPI.

```bash
# Development mode (debug enabled)
python run.py --mode dev

# Production-like mode
python run.py --mode prod
```

> ⚠️ Make sure you're running from the project root folder.

---

### 🐳 Docker Mode

Use Docker to run the app with mounted volumes for `data/` and `logs/`.

```bash
# Build the image
docker compose build

# Run the container
docker compose up
```

This maps:

- `./data` → `/app/data` in container
- `./logs` → `/app/logs` in container

---

### 🔥 WSGI (for Gunicorn / Production Servers)

To serve via Gunicorn (inside Docker):

```bash
docker compose up --build
```

> This uses `create_app()` dynamically based on your YAML config.

---

## ⚙️ Configuration

Edit `config/config.yaml` to change framework, port, debug mode, messages, and logging level:

```yaml
framework: flask  # or fastapi

app:
  name: "MultiModeApp"
  version: "1.0.0"
  debug: true
  port: 5000

logging:
  level: "INFO"
```

---

## 🔧 Logging

Logs are:

- Printed to the console
- Saved to rotating log files in `logs/` (or `/app/logs` in Docker)

Each class can instantiate a logger using:

```python
from helpers.logger import setup_logger

logger = setup_logger("MyService")
```

---

## 💻 Endpoints & UI

- Web UI (Flask only): [http://localhost:5000](http://localhost:5000)
- `/hello` GET/POST endpoints
- Styled with consistent dark theme

---

## 🪮 Removed Makefile

For cross-platform compatibility, `make` was removed. Use native Python or Docker commands instead.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🧬 Suggested Use

Clone this repo to use as a boilerplate for:

- Rapid API prototyping
- Teaching Flask vs FastAPI
- Microservice-style backends
- Custom web apps and dashboards

