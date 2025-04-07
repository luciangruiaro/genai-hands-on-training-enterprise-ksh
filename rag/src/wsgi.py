from core.app_factory import create_app
from config.config_loader import ConfigLoader
import uvicorn
import sys

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = create_app(template_dir=TEMPLATE_DIR, static_dir=STATIC_DIR)
config = ConfigLoader().get_config()
framework = config.get("framework", "").lower()
port = config["app"].get("port", 5000)
debug = config["app"].get("debug", False)

if __name__ == "__main__":
    if framework == "fastapi":
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
    elif framework == "flask":
        app.run(debug=debug, port=port)
    else:
        print(f"[ERROR] Unsupported framework: '{framework}'. Use 'flask' or 'fastapi'.", file=sys.stderr)
