"""Write the cause-specific Cox baseline's grid dumps from its parameters.

``fit_baseline_csc.R`` leaves, per fold, each patient's linear predictors
and each cause's baseline cumulative hazard (``TransEHR2.csc``). This
rebuilds the CSC's curve from them and writes it at the cuts, in the
columns ``dump_transehr2_deephit_predictions.py`` writes, as
``competing_risks_{split}_finetuned_output.csv`` for train, val and test:
the pmf is the differences of the cut values, and ``p_survive`` is one
minus their sum at the last cut. Every script that reads a dump then reads
the CSC's; the ones that need its curve between the cuts rebuild it
themselves.

Usage:
    python scripts/dump_baseline_csc_predictions.py <experiment_name> \\
        [--model_dir ./models] [--experiment_config <yaml>]
"""

import argparse
import numpy as np
import os

import _path  # noqa: F401  (repository root on sys.path)

from TransEHR2.csc import (
    BASELINE_FILE, LP_FILE, CSCCurve, read_csc_baseline, read_csc_lp
)

from dump_transehr2_deephit_predictions import save_predictions_csv
from evaluate_finetuned_predictions import TASK, get_fold_names, load_config


SPLITS = ('train', 'val', 'test')


def main():
    parser = argparse.ArgumentParser(
        description='Write CSC grid dumps from its exported parameters')
    parser.add_argument('experiment_name', type=str,
                        help='Name of the experiment under model_dir')
    parser.add_argument('--model_dir', type=str, default='./models',
                        help='Root directory containing the experiment')
    parser.add_argument(
        '--experiment_config', type=str, default=None,
        help='Experiment YAML, read for TIME_GRID_CUTS_DAYS and '
             'MODELLED_CAUSES, as evaluate_finetuned_predictions.py reads it.')
    args = parser.parse_args()

    grid, _ = load_config(args.experiment_config)
    experiment_dir = os.path.join(args.model_dir, args.experiment_name)
    if not os.path.isdir(experiment_dir):
        raise FileNotFoundError(f'{experiment_dir} does not exist')

    for fold_name in get_fold_names(experiment_dir):
        folder = os.path.join(experiment_dir, fold_name, TASK)
        baseline = read_csc_baseline(
            os.path.join(folder, BASELINE_FILE.format(task=TASK)), grid)
        if baseline is None:
            print(f"{fold_name}: no CSC baseline hazard, skipping")
            continue
        for split in SPLITS:
            lp = read_csc_lp(os.path.join(
                folder, LP_FILE.format(task=TASK, split=split)), grid)
            if lp is None:
                print(f"{fold_name}: no {split} linear predictors, skipping")
                continue
            cif, _ = CSCCurve(lp['lp'], baseline).read(grid.cuts)
            save_predictions_csv(
                lp['idx'], lp['time_to_event'], lp['event_type'], cif,
                np.diff(cif, axis=2, prepend=0.0),
                1.0 - cif[:, :, -1].sum(axis=1), grid,
                os.path.join(folder, f'{TASK}_{split}_finetuned_output.csv'))


if __name__ == '__main__':
    main()
