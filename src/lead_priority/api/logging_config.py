"""Structured-ish logging setup for the API and training scripts."""

from __future__ import annotations

import logging
import os

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(level: str | None = None) -> None:
    """Configure root logging once, honouring the ``LOG_LEVEL`` env var."""
    resolved = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, resolved, logging.INFO),
        format=_DEFAULT_FORMAT,
        force=False,
    )
