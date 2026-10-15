"""Recalibrate each fold's test predictions, and dump them beside the originals.

AJ-k recalibration (Alberge et al., AISTATS 2026, appendix F.1): per cause
and cut, the model's average incidence on a fold's validation predictions
is moved onto the validation outcomes' Aalen-Johansen curve, and that shift
is applied to the same fold's test predictions, then repaired back into a
distribution where it left one (``survival.apply_aj_recalibration``).

The validation split is the calibration set: it is held out of the weights,
and the recalibration is a handful of numbers per cause, one per cut, so
reusing the split that also chose the epoch costs little.

Opt-in, and it never overwrites: per fold it writes
``competing_risks_test_recal_aj_finetuned_output.csv`` beside the dump it
was made from, in the dump's own columns, which
``evaluate_finetuned_predictions.py`` then scores as the ``test_recal_aj``
split with the same routines as every other. What the corrected dump cannot
say about itself -- the fitted offsets, and how much of the shifted surface
had to be repaired -- goes in ``{model_dir}/{experiment}/test_recal_aj_record.yaml``.

A fold without both a validation and a test dump is skipped.

Usage:
    python scripts/recalibrate_predictions.py <experiment_name> \\
        [--model_dir ./models] [--experiment_config <yaml>]
"""

import argparse
import os
import yaml

from typing import Dict

import _path  # noqa: F401  (repository root on sys.path)

from TransEHR2.survival import (
    TimeGrid, aj_recalibration_offsets, apply_aj_recalibration
)

from dump_finetuned_predictions import save_predictions_csv
from evaluate_finetuned_predictions import (
    TASK, get_fold_names, load_config, outcomes, read_predictions
)


SPLIT = 'test_recal_aj'


def recalibrate(val: Dict, test: Dict, grid: TimeGrid) -> Dict:
    """``test``'s predictions after AJ-k recalibration fitted on ``val``.

    Returns the test dict with ``cif``, ``pmf`` and ``p_survive`` replaced,
    plus the fitted ``offsets`` (n_causes, n_bins) and the
    ``repaired_fraction`` of (patient, cause, cut) cells.
    """
    _, ci, ev, days = outcomes(val, grid)
    offsets = aj_recalibration_offsets(val['cif'], days, ci, ev, grid)
    cif, pmf, p_survive, repaired = apply_aj_recalibration(
        test['cif'], test['pmf'], test['p_survive'], offsets)
    return {**test, 'cif': cif, 'pmf': pmf, 'p_survive': p_survive,
            'offsets': offsets, 'repaired_fraction': repaired}


def main():
    parser = argparse.ArgumentParser(
        description='AJ-k recalibrate finetuned test predictions, fitted on '
                    'the validation predictions'
    )
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

    def path(fold_name, split):
        return os.path.join(experiment_dir, fold_name, TASK,
                            f'{TASK}_{split}_finetuned_output.csv')

    folds = {}
    for fold_name in get_fold_names(experiment_dir):
        val = read_predictions(path(fold_name, 'val'), grid)
        test = read_predictions(path(fold_name, 'test'), grid)
        if val is None or test is None:
            print(f"{fold_name}: needs both val and test dumps, skipping")
            continue
        recalibrated = recalibrate(val, test, grid)
        # Beside the dump it was made from, never over it: the plain
        # predictions stay what the model said.
        save_predictions_csv(
            recalibrated['idx'], recalibrated['time_to_event'],
            recalibrated['event_type'], recalibrated['cif'],
            recalibrated['pmf'], recalibrated['p_survive'], grid,
            path(fold_name, SPLIT))
        folds[fold_name] = {
            'repaired_fraction': float(recalibrated['repaired_fraction']),
            'offsets': {
                cause: [float(v) for v in recalibrated['offsets'][k]]
                for k, cause in enumerate(grid.cause_names)},
        }
        print(f"{fold_name}: repaired "
              f"{recalibrated['repaired_fraction']:.2%} of cells")

    if not folds:
        print("No fold has both val and test dumps; nothing written")
        return
    out_path = os.path.join(experiment_dir, f'{SPLIT}_record.yaml')
    with open(out_path, 'w') as f_out:
        yaml.dump({
            'experiment': args.experiment_name,
            'recalibration': 'aalen_johansen',
            'calibration_split': 'val',
            'applied_to': 'test',
            'time_grid_cuts_days': list(map(float, grid.cuts)),
            'causes': list(grid.cause_names),
            # Offsets are the model's mean excess over Aalen-Johansen at
            # each cut, subtracted from every test prediction.
            'folds': folds,
        }, f_out, default_flow_style=False, indent=2, sort_keys=False)
    print(f"Recalibrated {len(folds)} folds -> {out_path}")


if __name__ == '__main__':
    main()
