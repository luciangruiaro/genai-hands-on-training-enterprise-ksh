import argparse
from factory.fastapi_app import start_fastapi
from factory.cli_app import start_cli


def main():
    parser = argparse.ArgumentParser(description="LLM Helper App")
    parser.add_argument('--mode', choices=['rest', 'cli'], default='rest')
    args = parser.parse_args()

    if args.mode == 'rest':
        start_fastapi()
    else:
        start_cli()


if __name__ == "__main__":
    main()
