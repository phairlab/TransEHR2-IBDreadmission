#!/usr/bin/env python
"""Print the trial name of one arm's winning cell.

``select_tuned_cell.py --output`` writes the winner's *config*, and renames
``EXPERIMENT_NAME`` to the output file's stem -- which is what a downstream
phase wants in a base, and exactly what a caller cannot use to find the
directory the winner's weights were written to. This prints that name.

Ranked by ``hp_tuning.results.rank_cells``, the same call
``select_tuned_cell.py`` makes, so the two cannot disagree about which cell
won.

    python scripts/winning_cell_name.py <manifest> --arm additive

Exits 1 when no cell produced a usable metric, and 1 when the grid is
incomplete unless ``--allow_incomplete`` is given: a winner drawn from a
partial grid is the best of whatever finished, which is not the same claim.
"""

from __future__ import annotations

import argparse
import sys

import _path  # noqa: F401  (repository root on sys.path)

from TransEHR2.hp_tuning.results import rank_cells
from TransEHR2.hp_tuning.spec import load_manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('manifest')
    parser.add_argument('--arm', required=True)
    parser.add_argument('--criterion', default=None,
                        help='Defaults to the one the grid shares.')
    parser.add_argument('--allow_incomplete', action='store_true')
    args = parser.parse_args(argv)

    ranking = rank_cells(load_manifest(args.manifest), args.arm,
                         args.criterion)
    if ranking['best'] is None:
        print(f"No cell of arm {args.arm!r} produced a usable "
              f"{ranking['metric']}.", file=sys.stderr)
        return 1
    if not ranking['complete'] and not args.allow_incomplete:
        print(f"Arm {args.arm!r} is incomplete: the winner would be the best "
              f"of the cells that finished. Pass --allow_incomplete to take "
              f"it anyway.", file=sys.stderr)
        return 1

    print(ranking['best'].trial['name'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
