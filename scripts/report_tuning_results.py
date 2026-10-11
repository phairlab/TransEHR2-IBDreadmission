"""Read a sweep's results back through its manifest and rank them.

Reports; it does not decide. ``select_tuned_cell.py`` writes the decision
record, and exits non-zero on an incomplete grid unless told otherwise --
this prints the same ranking without that gate, so a sweep can be watched
while it runs.

The MIMIC fork rendered these as manuscript tables through a JMIR
formatting package. That package is not carried here, and a sweep needs to
print a ranking rather than typeset one, so the output is plain text.

Usage:
    python scripts/report_tuning_results.py <manifest>
    python scripts/report_tuning_results.py <manifest> --csv out.csv
"""

import argparse
import csv
import os
import sys

import _path  # noqa: F401  (repository root on sys.path)

from TransEHR2.hp_tuning.results import (progress, rank_cells,
                                         rank_hyperparameter, read_result)
from TransEHR2.hp_tuning.spec import SELECTION_CRITERIA, load_manifest


STATUS_NOTE = {
    'pending': 'not finished',
    'unreadable': 'file present, metric missing',
    'nan': 'non-finite, excluded from the ranking',
}


def format_value(value) -> str:
    """A grid value as the config carries it, not as Python repr."""
    if value is None:
        return 'flat'
    if isinstance(value, float):
        return f'{value:g}'
    return str(value)


def cell_label(trial: dict) -> str:
    """One cell's coordinates, or its name for a trial without any."""
    cell = trial.get('cell') or {}
    if not cell:
        return trial['name']
    return ', '.join(f'{key}={format_value(value)}'
                     for key, value in sorted(cell.items()))


def report_arm(manifest: dict, arm: str, rows: list) -> None:
    """Print one arm's cells, best first, and accumulate ``rows`` for CSV."""
    ranked = rank_cells(manifest, arm)
    results = ranked['results']
    criterion = ranked['criterion']
    spec = SELECTION_CRITERIA[criterion]

    print(f"\narm {arm}  --  {spec['metric']} on {criterion}, "
          f"{spec['direction']} is best")
    usable = sorted((r for r in results if r.value is not None),
                    key=lambda r: r.value,
                    reverse=spec['direction'] == 'max')

    width = max((len(cell_label(r.trial)) for r in results), default=10)
    for result in usable:
        marker = '*' if ranked['best'] and result is ranked['best'] else ' '
        print(f'  {marker} {cell_label(result.trial):<{width}}  '
              f'{result.value:.6f}')
    for result in results:
        if result.value is None:
            print(f'    {cell_label(result.trial):<{width}}  '
                  f'-- {STATUS_NOTE.get(result.status, result.status)}')

    for result in results:
        rows.append({
            'arm': arm, 'trial': result.trial['name'],
            'cell': cell_label(result.trial), 'criterion': criterion,
            'metric': spec['metric'], 'status': result.status,
            'value': result.value,
        })

    if ranked['best'] is None:
        print('  no usable result yet')
    elif not ranked['complete']:
        # Worth saying plainly: the best of what finished is a weaker claim
        # than the best of the grid, and the two look identical in a table.
        print(f"  best so far: {cell_label(ranked['best'].trial)} -- but the "
              f"grid is incomplete, so this is the best of the cells that "
              f"finished, not of the grid")
    else:
        print(f"  winner: {cell_label(ranked['best'].trial)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Rank a sweep\'s cells from its manifest.')
    parser.add_argument('manifest', help='The manifest the sweep wrote')
    parser.add_argument('--csv', help='Also write the rows to this file')
    parser.add_argument('--no_tables', action='store_true',
                        help='Accepted and ignored; the output is plain text '
                             'either way. Kept so the command the generator '
                             'prints runs as printed.')
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    print(f"sweep:  {manifest['spec_name']}  ({manifest['design']})")
    print(f"fold:   {manifest['fold']}")
    print(f"models: {manifest['model_dir']}")

    for stage, counts in sorted(progress(manifest).items()):
        if not counts['total']:
            continue
        print(f"  {stage}: {counts['done']}/{counts['total']} usable")
        # A pending trial and one whose metric went non-finite both land
        # here, and they want different responses -- requeue against
        # investigate -- so name them rather than only counting them.
        if counts['pending']:
            shown = ', '.join(counts['pending'][:4])
            more = len(counts['pending']) - 4
            print(f"    outstanding: {shown}"
                  f"{f' and {more} more' if more > 0 else ''}")

    rows = []
    arms = sorted({trial['arm'] for trial in manifest['trials']})
    if manifest['design'] == 'factorial':
        for arm in arms:
            report_arm(manifest, arm, rows)
    else:
        for name in manifest['grid']:
            ranked = rank_hyperparameter(manifest, name)
            print(f'\n{name}')
            for result in ranked['results']:
                value = format_value(result.trial.get('value'))
                shown = ('--' if result.value is None
                         else f'{result.value:.6f}')
                print(f'  {value:<12} {shown}')
            if ranked['best'] is not None:
                print(f"  winner: "
                      f"{format_value(ranked['best'].trial.get('value'))}")

    if args.csv:
        with open(args.csv, 'w', newline='') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nWrote {len(rows)} rows to {args.csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
