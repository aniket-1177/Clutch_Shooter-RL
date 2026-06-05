"""
utils/logger.py
---------------
Structured logging for training and evaluation runs.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def get_logger(name: str, log_dir: str = "results/logs") -> logging.Logger:
    """
    Returns a configured logger that writes to both console and a log file.

    Args:
        name:    Logger name (usually __name__ of the calling module).
        log_dir: Directory where log files are saved.

    Returns:
        logging.Logger instance.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:          # Avoid duplicate handlers on re-import
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(f"{log_dir}/{name}_{timestamp}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
