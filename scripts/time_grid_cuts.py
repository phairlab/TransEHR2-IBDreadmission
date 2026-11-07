"""The horizon of ``TIME_GRID_CUTS_DAYS``: the largest observed readmission time.

Every cut but the last is fixed by hand in the experiment configs (the recut2
grid). The last cut is the horizon, and it is a property of the data: the
largest ``TIME_TO_EVENT`` over the episodes that were actually drawn and
whose outcome was an unplanned readmission. This reads ``labels.csv`` and
``extracted_rows.npy`` and prints it. The number goes into every experiment
config by hand; nothing here writes one.

What the maximum is taken over
------------------------------

* **The sampled episodes, not the eligible pool.** ``split.py`` draws one
  episode per patient and ``extracted_rows.npy`` names them; those are the
  episodes a model ever sees.
* **Unplanned readmission only** -- ``EVENT_TYPE == 1``.
* **Cohort-wide, so every fold shares one grid.** The draw is fold-
  independent by construction, so there is one answer rather than five.

``TimeGrid.discretize`` takes ``t < horizon_days`` strictly, so the episode
sitting exactly on the last cut is censored there rather than counted as an
event; the run says how many do.

Usage:
    python scripts/time_grid_cuts.py /path/to/data
"""

import argparse
import numpy as np
import os
import pandas as pd
import sys

import _path  # noqa: F401  (repository root on sys.path)

from TransEHR2.survival import format_days

MINUTES_PER_DAY = 1440.0

# build_labels.py's coding.
READMISSION = 1

LABELS_FILE = 'labels.csv'
SELECTION_FILE = 'extracted_rows.npy'


def sampled_labels(data_dir: str) -> pd.DataFrame:
    """The drawn episodes, in the extractor's row order."""

    labels_path = os.path.join(data_dir, LABELS_FILE)
    selection_path = os.path.join(data_dir, SELECTION_FILE)
    for path in (labels_path, selection_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'{path} is missing; run IBDdataprep\'s build_labels.py and '
                f'split.py first -- the horizon is a property of the draw, so '
                f'it cannot be taken before it exists'
            )

    labels = pd.read_csv(labels_path)
    rows = np.load(selection_path)
    if rows.size and (rows.min() < 0 or rows.max() >= len(labels)):
        raise ValueError(
            f'{SELECTION_FILE} spans [{rows.min()}, {rows.max()}] but '
            f'{LABELS_FILE} has {len(labels)} row(s); they are from '
            f'different runs'
        )
    return labels.iloc[rows].reset_index(drop=True)


def readmission_days(labels: pd.DataFrame) -> np.ndarray:
    """Days to unplanned readmission, for the episodes that had one."""

    readmitted = labels[labels['EVENT_TYPE'] == READMISSION]
    return (readmitted['TIME_TO_EVENT'].to_numpy(dtype=np.float64)
            / MINUTES_PER_DAY)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Largest observed readmission time, as the last '
                    'TIME_GRID_CUTS_DAYS cut.'
    )
    parser.add_argument('data_dir',
                        help='Directory holding labels.csv and '
                             'extracted_rows.npy')
    args = parser.parse_args(argv)

    labels = sampled_labels(args.data_dir)
    days = readmission_days(labels)
    if days.size == 0:
        print(f'No episode in the draw has EVENT_TYPE == {READMISSION}; '
              f'there is no readmission time to take the maximum of.')
        return 1

    horizon = round(float(days.max()), 2)
    print(f'data:      {args.data_dir}')
    print(f'episodes:  {len(labels)} drawn, one per patient')
    share = len(days) / len(labels) if len(labels) else 0.0
    print(f'           {len(days)} ended in unplanned readmission '
          f'({share:.2%})')
    print(f'readmission time: {format_days(days.min())} to '
          f'{format_days(days.max())}, median {format_days(np.median(days))}')

    at_horizon = int((days >= horizon).sum())
    print(f'\n  {at_horizon} episode(s) sit at or past {horizon:.2f} days. '
          f'discretize() takes\n  t < horizon strictly, so those are '
          f'censored at the horizon rather than\n  counted as events.')
    print(f'  BRIER_INTEGRATION_DAYS must be <= {horizon:.2f}.')

    print(f'\nSet the last TIME_GRID_CUTS_DAYS value in every experiment '
          f'config to:\n\n{horizon:.2f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
