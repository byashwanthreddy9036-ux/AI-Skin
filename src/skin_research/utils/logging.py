"""Logging setup: concise console output plus an optional detailed log file."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure the root ``skin_research`` logger (idempotent)."""
    logger = logging.getLogger("skin_research")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``skin_research`` namespace."""
    if not name.startswith("skin_research"):
        name = f"skin_research.{name}"
    return logging.getLogger(name)
