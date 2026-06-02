"""Lead Priority: lead scoring + engagement sentiment + combined priority score.

The package is organised into focused sub-packages:

* ``lead_priority.data``      - data download + synthetic interaction generation
* ``lead_priority.features``  - leakage-aware feature engineering for the tabular model
* ``lead_priority.scoring``   - conversion-probability models (LogReg baseline + LightGBM)
* ``lead_priority.sentiment`` - engagement intent/sentiment classifier (TR + EN)
* ``lead_priority.priority``  - combine conversion probability and sentiment into one score
* ``lead_priority.api``       - FastAPI service exposing ``/score`` and ``/leads/top``
"""

from __future__ import annotations

__version__ = "0.1.0"
