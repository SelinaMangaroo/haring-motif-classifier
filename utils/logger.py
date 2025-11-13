import os
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def get_logger(script_name: str) -> logging.Logger:
    """
    Creates and configures a logger for a given script.

    Args:
        script_name (str): Short name of the script (e.g. 'create_dataset', 'train', 'predict').

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    log_filename = f"{script_name}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

        logger.addHandler(fh)
        logger.addHandler(ch)

    logger.info(f"Logger initialized for {script_name}. Log file: {log_path}")
    return logger
