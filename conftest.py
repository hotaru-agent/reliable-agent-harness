"""Pytest bootstrap for Project 2.

Adds the project root to ``sys.path`` so the ``harness`` and ``storage``
packages import cleanly when running pytest from any directory.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
