from __future__ import annotations

import logging
import sys
from pathlib import Path


LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=getattr(logging, level), format=LOG_FORMAT, handlers=handlers, force=True)
