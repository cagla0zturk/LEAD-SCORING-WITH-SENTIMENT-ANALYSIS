"""Ensure the raw X Education ``Leads.csv`` is available locally.

The CSV ships with the repository under ``data/raw/``. This helper exists so that a
fresh clone (or CI environment) can self-heal by downloading from a public mirror if
the file is missing, instead of failing deep inside the training pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import requests

from lead_priority.config import LEADS_CSV_URL, RAW_LEADS_CSV

logger = logging.getLogger(__name__)

_DOWNLOAD_TIMEOUT_SECONDS = 60


def ensure_leads_csv(path: Path = RAW_LEADS_CSV, url: str = LEADS_CSV_URL) -> Path:
    """Return the path to ``Leads.csv``, downloading it from ``url`` if absent.

    Parameters
    ----------
    path:
        Local destination for the raw CSV.
    url:
        Public mirror to fetch from when ``path`` does not yet exist.

    Returns
    -------
    Path
        The path to the (now guaranteed to exist) CSV file.
    """
    if path.exists() and path.stat().st_size > 0:
        logger.debug("Leads.csv already present at %s", path)
        return path

    logger.info("Leads.csv not found locally; downloading from %s", url)
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS)
    response.raise_for_status()
    path.write_bytes(response.content)
    logger.info("Saved Leads.csv (%d bytes) to %s", len(response.content), path)
    return path
