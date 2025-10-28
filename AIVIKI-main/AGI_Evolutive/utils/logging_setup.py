"""Utility helpers to configure consistent logging for the CLI and runtime."""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

_DEFAULT_LOG_PATH = Path(os.environ.get("AGI_LOG_PATH", "runtime/logs/agi_evolutive.log"))
_CONFIGURED_PATH: Optional[Path] = None


def _resolve_level(level: Optional[str]) -> int:
    candidate = level or os.environ.get("AGI_LOG_LEVEL") or "INFO"
    numeric = getattr(logging, str(candidate).upper(), None)
    if isinstance(numeric, int):
        return numeric
    return logging.INFO


def configure_logging(log_path: Optional[str] = None, level: Optional[str] = None) -> Path:
    """Configure root logging handlers and return the active log file path.

    The configuration installs a rotating file handler and a console handler so the
    agent remains observable both from the terminal and from persisted log files.
    Subsequent calls are cheap and simply return the previously configured path.
    """

    global _CONFIGURED_PATH

    if _CONFIGURED_PATH is not None:
        return _CONFIGURED_PATH

    target = Path(log_path) if log_path else _DEFAULT_LOG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    # Remove any pre-existing handlers so we can start from a clean state.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    resolved_level = _resolve_level(level)
    root_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = RotatingFileHandler(
        target,
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(resolved_level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(resolved_level)
    console_handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.captureWarnings(True)

    def _handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logging.getLogger("agi_evolutive").error(
            "Unhandled exception: %s", exc_value, exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = _handle_exception

    _CONFIGURED_PATH = target
    return target


__all__ = ["configure_logging"]
