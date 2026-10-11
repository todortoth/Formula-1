"""
main.py · Formula 1 Strategic Decisions
=========================================================
Convenience launcher so ``python main.py`` works straight from a clone, without installing the package.
Equivalent to ``python -m f1strategy`` / the ``f1-graph`` command; see ``python main.py --help``.

Defaults reproduce the original example: 2025 Belgian Grand Prix, race, lap 44.
"""

import sys
from pathlib import Path

try:
    from f1strategy.cli import main
except ImportError:  # not installed: fall back to the src/ layout of this checkout
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from f1strategy.cli import main

if __name__ == "__main__":
    sys.exit(main())
