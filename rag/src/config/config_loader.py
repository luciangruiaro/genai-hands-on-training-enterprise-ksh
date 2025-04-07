import yaml
import os
import logging

DEFAULT_CONFIG = {
    "framework": "flask",
    "app": {
        "name": "Py Multi Mode App Template - Lucian",
        "version": "0.1.0",
        "debug": False,
        "port": 5000
    },
    "logging": {
        "level": "INFO"
    }
}


class ConfigLoader:
    _config = None

    @classmethod
    def get_config(cls, path: str = "config/config.yaml") -> dict:
        """
        Loads the configuration file and merges it with defaults.
        Falls back to DEFAULT_CONFIG if loading fails or keys are missing.
        """
        if cls._config is None:
            config_data = {}
            try:
                if os.path.exists(path):
                    with open(path, "r") as file:
                        config_data = yaml.safe_load(file) or {}
                else:
                    logging.warning(f"[ConfigLoader] Config file not found at {path}. Using defaults.")
            except Exception as e:
                logging.error(f"[ConfigLoader] Failed to load config from {path}: {e}")

            # Merge with defaults (deep merge)
            cls._config = cls._deep_merge(DEFAULT_CONFIG.copy(), config_data)
        return cls._config

    @staticmethod
    def _deep_merge(default: dict, override: dict) -> dict:
        for key, value in override.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                default[key] = ConfigLoader._deep_merge(default[key], value)
            else:
                default[key] = value
        return default
