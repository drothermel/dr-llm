"""Make ``scripts/`` importable for tests in ``tests/scripts/``.

The demo scripts are not part of the installed package, so this conftest puts
the project's ``scripts/`` directory on ``sys.path`` to let tests import them
as plain modules instead of executing them via ``runpy``.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
