import argparse, uvicorn, os, sys

from core.app_factory import create_app, LOGS_DIR
from config.config_loader import ConfigLoader
from helpers.logger import setup_logger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['dev', 'prod'], default='dev')
    args = parser.parse_args()

    config = ConfigLoader().get_config()
    framework = config.get("framework", "").lower()
    port = config["app"].get("port", 5000)
    debug = config["app"].get("debug", False)

    setup_logger("app", log_dir=LOGS_DIR)
    app = create_app(template_dir=TEMPLATE_DIR, static_dir=STATIC_DIR)

    if framework == "fastapi":
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    elif framework == "flask":
        app.run(debug=(args.mode == 'dev' and debug), port=port)
    else:
        print(f"[ERROR] Unsupported framework: '{framework}'. Use 'flask' or 'fastapi'.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
