"""Make ``src/`` importable even when the package is not pip-installed.

Importing this module (``import _bootstrap``) prepends the project's ``src`` directory
to ``sys.path``. It is a no-op when the package is already installed (e.g. via
``pip install -e .`` or inside the Docker image).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
