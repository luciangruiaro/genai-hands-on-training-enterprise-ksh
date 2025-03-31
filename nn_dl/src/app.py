import logging
import os
import yaml
from flask import Flask, render_template
from dotenv import load_dotenv

from api.nn_controller import nn_bp
from api.sn_controller import sn_bp

from helpers.logger_config import get_logger

logger = get_logger(__name__)


def create_app():
    load_dotenv()

    config_path = os.getenv("APP_CONFIG", "resources/application.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            app_config = yaml.safe_load(f)
    else:
        app_config = {}

    app = Flask(__name__)
    app.config["DEBUG"] = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.config["APP_CONFIG"] = app_config

    # Register routes
    app.register_blueprint(nn_bp, url_prefix="/nn")
    app.register_blueprint(sn_bp, url_prefix="/sn")

    # ✅ Add homepage route here
    @app.route("/")
    def index():
        return render_template("index.html")

    return app
