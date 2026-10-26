"""CR D-calibration of the CSC baseline, read the way the DeepHit models are.

The evaluation reads the CSC on its exact curve: F_k(T_i) at each patient's
own outcome time, recalibrated by dense offsets. The DeepHit models have
their curve only at the grid's cuts, so their F_k(T_i) is Hyman-interpolated
between cuts, and their recalibration acts on the cut values and is
re-interpolated. A model that is calibrated at the cuts can still fail
D-calibration if the interpolation puts each F_k(T_i) in the wrong decile.

This scores the CSC through the DeepHit path, to separate the two causes:

* ``raw``: the CSC's test dump (its exact curve at the cuts) goes through
  ``cr_d_calibration`` and ``cr_dcal_chi2``, which interpolate as they do for
  DeepHit.
* ``recal``: the CSC's val and test dumps are AJ-k recalibrated at the cuts by
  ``recalibrate_predictions.recalibrate``, the DeepHit recalibration, and
  scored the same way.

If the CSC fails here where it passes on its exact curve, the failure belongs
to the grid reading, not to DeepHit.

Each row is one fold, cause and variant, with ``read`` set to ``cuts`` for
the values computed here and ``exact`` for the ones the evaluation already
wrote (``test_evaluation.yaml``, ``test_recal_aj_evaluation.yaml``), so the
two sit side by side. Nothing under the model directory is written.

PRINTS AND WRITES AGGREGATES ONLY: per-fold statistics, no episode.

Usage:
    python scripts/csc_dcal_at_cuts.py --out <csv> \\
        [--experiment_name part1_csc] [--model_dir ./models] \\
        [--experiment_config <yaml>]
"""

import argparse
import os

from typing import Dict, List

import pandas as pd
import yaml

import _path  # noqa: F401  (repository root on sys.path)

from TransEHR2.survival import (
    DEFAULT_CR_DCAL_ALPHA, DEFAULT_CR_DCAL_CHI2_BINS, DEFAULT_CR_DCAL_N_RHO,
    cr_d_calibration, cr_dcal_chi2, metric_label
)

from evaluate_finetuned_predictions import (
    TASK, get_fold_names, load_config, outcomes, read_predictions
)
from recalibrate_predictions import recalibrate


COLUMNS = ['fold', 'cause', 'variant', 'read', 'CR_DCal', 'chi2', 'p']
# The evaluation YAML each variant's exact-curve values are in.
EXACT_SPLITS = {'raw': 'test', 'recal': 'test_recal_aj'}


def at_cuts(data: Dict, grid, dcal: Dict) -> List[Dict]:
    """Each cause's D-calibration of ``data`` on the interpolated grid curve."""
    _, ci, ev, days = outcomes(data, grid)
    rows = []
    for k, cause in enumerate(grid.cause_names):
        prediction = (data['cif'], data['pmf'], data['p_survive'], days, ci,
                      ev, k, grid)
        statistic, p_value = cr_dcal_chi2(*prediction,
                                          n_bins=dcal['chi2_bins'])
        rows.append({'cause': cause, 'CR_DCal': cr_d_calibration(
                         *prediction, alpha=dcal['alpha'],
                         n_rho=dcal['n_rho']),
                     'chi2': statistic, 'p': p_value})
    return rows


def exact(experiment_dir: str, grid) -> List[Dict]:
    """The exact-curve values the evaluation wrote, as rows."""
    rows = []
    for variant, split in EXACT_SPLITS.items():
        path = os.path.join(experiment_dir, f'{split}_evaluation.yaml')
        if not os.path.exists(path):
            print(f"{path} not found; no exact '{variant}' rows")
            continue
        with open(path) as f_in:
            evaluation = yaml.safe_load(f_in)
        for i, fold in enumerate(evaluation['folds']):
            for cause in grid.cause_names:
                key = metric_label(cause)
                rows.append({
                    'fold': fold, 'cause': cause, 'variant': variant,
                    'read': 'exact',
                    'CR_DCal': evaluation[f'{key}_CR_DCal'][i],
                    'chi2': evaluation[f'{key}_CR_DCal_Chi2'][i],
                    'p': evaluation[f'{key}_CR_DCal_Chi2_P'][i]})
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="CR D-calibration of the CSC read through the DeepHit "
                    "models' interpolated grid curve")
    parser.add_argument('--out', required=True,
                        help='CSV to write the per-fold rows to')
    parser.add_argument('--experiment_name', default='part1_csc')
    parser.add_argument('--model_dir', default='./models')
    parser.add_argument('--experiment_config', default=None,
                        help='Experiment YAML, for the grid and the '
                             'CR_DCAL_* settings')
    args = parser.parse_args(argv)

    grid, cfg = load_config(args.experiment_config)
    dcal = {'alpha': float(cfg.get('CR_DCAL_ALPHA', DEFAULT_CR_DCAL_ALPHA)),
            'n_rho': int(cfg.get('CR_DCAL_N_RHO', DEFAULT_CR_DCAL_N_RHO)),
            'chi2_bins': int(cfg.get('CR_DCAL_CHI2_BINS',
                                     DEFAULT_CR_DCAL_CHI2_BINS))}
    experiment_dir = os.path.join(args.model_dir, args.experiment_name)
    if not os.path.isdir(experiment_dir):
        raise FileNotFoundError(f'{experiment_dir} does not exist')

    def dump(fold_name, split):
        return read_predictions(os.path.join(
            experiment_dir, fold_name, TASK,
            f'{TASK}_{split}_finetuned_output.csv'), grid)

    rows = []
    for fold_name in get_fold_names(experiment_dir):
        val, test = dump(fold_name, 'val'), dump(fold_name, 'test')
        if val is None or test is None:
            print(f"{fold_name}: needs both val and test dumps, skipping")
            continue
        for variant, data in (('raw', test),
                              ('recal', recalibrate(val, test, grid))):
            rows.extend({'fold': fold_name, 'variant': variant,
                         'read': 'cuts', **row}
                        for row in at_cuts(data, grid, dcal))
        print(f"{fold_name}: scored at the cuts")
    if not rows:
        print("No fold has both val and test dumps; nothing written")
        return 1

    frame = pd.DataFrame(rows + exact(experiment_dir, grid), columns=COLUMNS)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(frame.pivot_table(index=['cause', 'variant'], columns='read',
                            values='chi2', aggfunc='median').round(1))
    print(f"-> {args.out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
