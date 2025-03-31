import os
import yaml
from dotenv import load_dotenv

load_dotenv()


def load_app_config():
    config_path = os.getenv("APP_CONFIG", "resources/application.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


APP_CONFIG = load_app_config()
