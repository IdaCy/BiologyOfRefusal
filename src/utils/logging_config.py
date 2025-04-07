import logging
import os

def setup_logger(
    log_file_path="logs/run.log",
    console_level="INFO",
    file_level="DEBUG"
):
    """
    Sets up a Python logger that logs to both console and file.
    """
    logger = logging.getLogger("gemma_logger")
    logger.setLevel(logging.DEBUG)  # master level

    # If no handlers, add them. (Prevents adding multiple times in certain environments.)
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, console_level.upper()))
        logger.addHandler(ch)

        # File handler
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        fh = logging.FileHandler(log_file_path, mode='a')
        fh.setLevel(getattr(logging, file_level.upper()))
        logger.addHandler(fh)

    return logger
