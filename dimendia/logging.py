"""Tiny logging helper so the whole package shares one configured logger."""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def get_logger(name: str = "dimendia") -> logging.Logger:
    """Return a package logger, configuring a stream handler exactly once."""
    global _CONFIGURED
    if not _CONFIGURED:
        level_name = os.environ.get("DIMENDIA_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root = logging.getLogger("dimendia")
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name)
