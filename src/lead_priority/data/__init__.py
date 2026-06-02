"""Data access: raw tabular loading + synthetic interaction-note generation."""

from __future__ import annotations

from lead_priority.data.download import ensure_leads_csv
from lead_priority.data.loaders import load_raw_leads
from lead_priority.data.synthetic_interactions import (
    InteractionDataset,
    generate_interaction_dataset,
)

__all__ = [
    "ensure_leads_csv",
    "load_raw_leads",
    "InteractionDataset",
    "generate_interaction_dataset",
]
