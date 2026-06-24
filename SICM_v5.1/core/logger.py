import logging
from pathlib import Path


def setup_logger():

    Path("logs").mkdir(
        exist_ok=True
    )

    logging.basicConfig(

        filename="logs/sicm.log",

        level=logging.INFO,

        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        )
    )

    return logging.getLogger("SICM")
