import os
import logging
from logging import Logger
from logging.handlers import RotatingFileHandler

# Constants
DEFAULT_LOG_DIR = "./logs"
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
MAX_LOG_FILE_SIZE = 5_000_000  # 5 MB
BACKUP_COUNT = 3


def setup_logger(name: str = "app", log_dir: str = DEFAULT_LOG_DIR, level: str = "INFO") -> Logger:
    """
    Sets up and returns a logger with console and rotating file handlers.

    Args:
        name (str): The name of the logger (also used as the log file name).
        log_dir (str): Directory where log files will be stored.
        level (str): Logging level (e.g., "DEBUG", "INFO", "ERROR").

    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger  # Avoid duplicate handlers if already set up

    log_level = getattr(logging, level.upper(), DEFAULT_LOG_LEVEL)

    logger.setLevel(log_level)

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file_path = os.path.join(log_dir, f"{name}.log")
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=MAX_LOG_FILE_SIZE, backupCount=BACKUP_COUNT
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
