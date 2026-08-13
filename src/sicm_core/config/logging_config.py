import logging
from .paths import ensure_log_directory

def configure_logging(level="INFO", log_file="logs/sicm.log"):
    logger = logging.getLogger("sicm")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
        handler = logging.FileHandler(ensure_log_directory(log_file), encoding="utf-8")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
