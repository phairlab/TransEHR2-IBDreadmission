"""``TIME_GRID_CUTS_DAYS`` from the quantiles of observed readmission time.

Reads ``labels.csv`` and ``extracted_rows.npy`` and prints the decile breaks
of ``TIME_TO_EVENT`` over the episodes that were actually drawn and whose
outcome was an unplanned readmission. The numbers go into an experiment
config by hand; nothing here writes one.

Why this rather than the fixed 30/60/90-day grid
------------------------------------------------

Equal-width-in-intuition cuts put most of the cohort in one or two bins and
leave others nearly empty, and a DeepHit bin with no events in it contributes
nothing but a softmax slot. Quantile cuts spend the bins where the events are:
each of the ten holds a tenth of the observed readmissions by construction.

What the quantiles are taken over
---------------------------------

* **The sampled episodes, not the eligible pool.** ``split.py`` draws one
  episode per patient and ``extracted_rows.npy`` names them; those are the
  episodes a model ever sees, so they are the ones whose time distribution
  the grid should match. ``development/scripts/event_time_distribution.py``
  reports deciles over the whole eligible pool instead, which is a different
  and larger set -- read its decile table as a description of the cohort, not
  as the cuts in use.
* **Unplanned readmission only** -- ``EVENT_TYPE == 1``. An episode that was
  censored, or that ended in death or out-migration, has no readmission time
  to contribute. This is the observed-event distribution, which is what
  discretization is conventionally fitted to.
* **Cohort-wide, so every fold shares one grid.** The draw is fold-
  independent by construction (one episode per patient, used by every
  partition of every fold), so there is one answer rather than five. Per-fold
  grids fitted on training events alone would avoid letting test episodes
  inform the bin edges, at the cost of making per-bin metrics incomparable
  across folds; the edges are a coarse summary of the marginal time
  distribution and carry no covariate information, which is why the usual
  reading is that this is not the leakage worth paying for.

Two things the run checks rather than assumes
---------------------------------------------

* **Ties.** Two deciles landing on one value would give a zero-width bin,
  which ``TimeGrid`` refuses. Duplicates are collapsed and reported, so the
  grid comes back with fewer than ten bins rather than failing.
* **The horizon.** ``TimeGrid.discretize`` takes ``t < horizon_days``
  strictly, so an episode sitting exactly on the last cut is censored there
  rather than counted as an event. The last cut *is* the maximum observed
  readmission time, so that is at least one episode; the run says how many.

Usage:
    python scripts/time_grid_cuts.py /path/to/data
    python scripts/time_grid_cuts.py /path/to/data --step 0.2
"""

import argparse
import numpy as np
import os
import pandas as pd
import sys
import torch

import _path  # noqa: F401  (repository root on sys.path)

from TransEHR2.survival import TimeGrid, format_days

MINUTES_PER_DAY = 1440.0

# build_labels.py's coding.
READMISSION = 1

# One cut every tenth of the distribution, the last at the maximum. Ten cuts
# is ten bins: TimeGrid's last cut is the horizon, not the start of an open
# bin, so no eleventh is needed.
DEFAULT_STEP = 0.1

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
                f'split.py first -- the cuts are a property of the draw, so '
                f'they cannot be taken before it exists'
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


def quantile_cuts(days: np.ndarray, step: float):
    """``(cuts, quantiles, n_collapsed)`` -- the bin edges, ascending.

    The quantiles run from ``step`` to 1.0 inclusive: the lower end of the
    first bin is 0, which needs no cut, and the upper end of the last is the
    maximum, which does.
    """

    quantiles = np.arange(step, 1.0 + step / 2, step)
    quantiles = np.minimum(quantiles, 1.0)
    cuts = np.quantile(days, quantiles)

    kept, kept_q = [], []
    for cut, q in zip(cuts, quantiles):
        if not kept or cut > kept[-1]:
            kept.append(float(cut))
            kept_q.append(float(q))
    return kept, kept_q, len(cuts) - len(kept)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Decile breaks of observed readmission time, as '
                    'TIME_GRID_CUTS_DAYS.'
    )
    parser.add_argument('data_dir',
                        help='Directory holding labels.csv and '
                             'extracted_rows.npy')
    parser.add_argument('--step', type=float, default=DEFAULT_STEP,
                        help='Quantile step [default: %(default)s]')
    args = parser.parse_args(argv)

    if not 0.0 < args.step <= 1.0:
        print(f'--step must be in (0, 1], got {args.step}')
        return 1

    labels = sampled_labels(args.data_dir)
    days = readmission_days(labels)
    if days.size == 0:
        print(f'No episode in the draw has EVENT_TYPE == {READMISSION}; '
              f'there is no readmission time distribution to take quantiles '
              f'of.')
        return 1

    cuts, quantiles, collapsed = quantile_cuts(days, args.step)
    grid = TimeGrid(cuts_days=cuts)
    bins = grid.bin_of(torch.from_numpy(days))

    print(f'data:      {args.data_dir}')
    print(f'episodes:  {len(labels)} drawn, one per patient')
    share = len(days) / len(labels) if len(labels) else 0.0
    print(f'           {len(days)} ended in unplanned readmission '
          f'({share:.2%})')
    print(f'readmission time: {format_days(days.min())} to '
          f'{format_days(days.max())}, median {format_days(np.median(days))}')

    print(f'\n{"quantile":>9}  {"cut (days)":>12}  {"":>10}  '
          f'{"readmissions":>13}')
    counts = np.bincount(bins.numpy(), minlength=grid.n_bins)
    for i, (q, cut) in enumerate(zip(quantiles, cuts)):
        print(f'{q:>9.2f}  {cut:>12.2f}  {format_days(cut):>10}  '
              f'{counts[i]:>13}')

    if collapsed:
        print(f'\n  {collapsed} quantile(s) collapsed onto a value already '
              f'taken, leaving\n  {len(cuts)} bins rather than '
              f'{len(quantiles) + collapsed}. A zero-width bin is one '
              f'TimeGrid\n  refuses, and a repeated break means that much of '
              f'the distribution sits\n  on one value.')

    at_horizon = int((days >= grid.horizon_days).sum())
    print(f'\n  {at_horizon} episode(s) sit at or past the last cut. '
          f'discretize() takes\n  t < horizon strictly, so those are '
          f'censored at the horizon rather than\n  counted as events -- the '
          f'last cut being the maximum, at least one is.')
    print(f'  BRIER_INTEGRATION_DAYS must be <= {grid.horizon_days:.2f}.')

    print('\nPaste into the experiment config:\n')
    print('TIME_GRID_CUTS_DAYS: ['
          + ', '.join(f'{cut:.2f}' for cut in cuts) + ']')
    return 0


if __name__ == '__main__':
    sys.exit(main())
