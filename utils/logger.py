import logging
import os
import sys


def setup_logger(name, log_dir, level=logging.INFO):
    """Create a logger that writes to both console and a log file."""
    os.makedirs(log_dir, exist_ok=True)
    log_file  = os.path.join(log_dir, f"{name}.log")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler    = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger
