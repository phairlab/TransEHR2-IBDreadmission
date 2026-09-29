"""Put the repository root on ``sys.path``.

``python scripts/run_experiment.py`` makes ``scripts/`` the first entry on
``sys.path``, not the repository root, so ``import TransEHR2`` fails. The
package is not installed -- there is no ``pyproject.toml`` and the cluster
runs it straight from a checkout -- so something has to add the root, and
doing it here keeps it to one line per script instead of three.

Import it before any ``TransEHR2`` import, and leave it first:

    import _path  # noqa: F401  (repository root on sys.path)

    from TransEHR2.survival import TimeGrid

A formatter that sorts imports will move it below the package imports and
break them, which is why it is a plain module rather than a ``from`` import
that could be folded in with the rest.
"""

import sys

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
