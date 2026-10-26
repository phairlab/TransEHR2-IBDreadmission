"""The CR D-calibration curves behind D_k, for P-P plots.

For cause k, u = F_k(T)/F_k(tau) is uniform among the patients who fail of k
when the model is calibrated, and b_k(rho) is the censoring-adjusted share of
the cause's predicted incidence whose u is at most rho
(``cr_dcal_numerator_from_values``). Plotted against rho, b_k is a P-P plot,
which is the diagonal for a calibrated model. D_k is its root-mean-square
distance from the diagonal (at alpha = 2).

Each curve is read exactly as the evaluation read it: the CSC on its exact
curve, recalibrated by its dense offsets, and the DeepHit models on their grid
curve through Hyman (``evaluate_finetuned_predictions.dcal_values``). Before
anything is written, every curve's D_k is recomputed from it and checked
against the D_k the evaluation wrote: a mismatch stops the run.

Writes one CSV: experiment, variant (raw or recal), fold, cause, rho, b. Rows
are on the config's CR_DCAL_N_RHO grid, the one D_k is computed on.

PRINTS AND WRITES AGGREGATES ONLY: curves over every test episode of a fold,
no episode.

Usage:
    python scripts/cr_dcal_curves.py --out <csv> \\
        --experiments part1_transehr2 part1_tabdh part1_csc \\
        [--model_dir ./models] [--config_dir TransEHR2/configs/experiments]
"""

import argparse
import os

from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

import _path  # noqa: F401  (repository root on sys.path)

from TransEHR2.survival import (
    DEFAULT_CR_DCAL_ALPHA, DEFAULT_CR_DCAL_N_RHO,
    cr_dcal_numerator_from_values, metric_label
)

from evaluate_finetuned_predictions import (
    TASK, csc_reads, dcal_values, get_fold_names, load_config,
    load_csc_curve, outcomes, read_predictions
)


COLUMNS = ['experiment', 'variant', 'fold', 'cause', 'rho', 'b']
SPLITS = {'raw': 'test', 'recal': 'test_recal_aj'}
# The curve reproduces D_k through the same arithmetic; anything past
# rounding means it is not the curve the evaluation scored.
TOLERANCE = 1e-9


def fold_curves(experiment_dir: str, fold_name: str, split: str, grid,
                rhos: np.ndarray) -> Dict[str, np.ndarray]:
    """``{cause: b(rho)}`` for one fold and split, or {} if no dump."""
    data = read_predictions(os.path.join(
        experiment_dir, fold_name, TASK,
        f'{TASK}_{split}_finetuned_output.csv'), grid)
    if data is None:
        return {}
    _, ci, ev, days = outcomes(data, grid)
    curve = load_csc_curve(experiment_dir, fold_name, split, data['idx'],
                           grid)
    csc = None if curve is None else csc_reads(curve, days, grid, [])
    curves = {}
    for k, cause in enumerate(grid.cause_names):
        numerator, denominator = cr_dcal_numerator_from_values(
            *dcal_values(data, grid, k, days, csc), ci, ev, k, rhos)
        curves[cause] = numerator / denominator
    return curves


def check_against_evaluation(experiment_dir: str, split: str, folds: List,
                             curves: Dict, grid, rhos, alpha) -> float:
    """The largest gap between each curve's D_k and the evaluation's."""
    path = os.path.join(experiment_dir, f'{split}_evaluation.yaml')
    with open(path) as f_in:
        evaluation = yaml.safe_load(f_in)
    worst = 0.0
    for fold_name in folds:
        i = evaluation['folds'].index(fold_name)
        for cause in grid.cause_names:
            b = curves[(fold_name, cause)]
            d_k = float(np.mean(np.abs(b - rhos) ** alpha) ** (1.0 / alpha))
            reported = evaluation[f'{metric_label(cause)}_CR_DCal'][i]
            worst = max(worst, abs(d_k - reported))
    return worst


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='CR D-calibration curves b_k(rho) for P-P plots')
    parser.add_argument('--out', required=True)
    parser.add_argument('--experiments', nargs='+', required=True)
    parser.add_argument('--model_dir', default='./models')
    parser.add_argument('--config_dir',
                        default=os.path.join('TransEHR2', 'configs',
                                             'experiments'),
                        help='Where <experiment>.yaml is, for the grid and '
                             'the CR_DCAL_* settings')
    args = parser.parse_args(argv)

    frames = []
    for experiment in args.experiments:
        grid, cfg = load_config(os.path.join(args.config_dir,
                                             f'{experiment}.yaml'))
        alpha = float(cfg.get('CR_DCAL_ALPHA', DEFAULT_CR_DCAL_ALPHA))
        n_rho = int(cfg.get('CR_DCAL_N_RHO', DEFAULT_CR_DCAL_N_RHO))
        rhos = np.arange(1, n_rho + 1, dtype=np.float64) / n_rho
        experiment_dir = os.path.join(args.model_dir, experiment)
        if not os.path.isdir(experiment_dir):
            raise FileNotFoundError(f'{experiment_dir} does not exist')

        for variant, split in SPLITS.items():
            curves, folds = {}, []
            for fold_name in get_fold_names(experiment_dir):
                found = fold_curves(experiment_dir, fold_name, split, grid,
                                    rhos)
                if not found:
                    continue
                folds.append(fold_name)
                for cause, b in found.items():
                    curves[(fold_name, cause)] = b
            if not folds:
                print(f"{experiment} {variant}: no {split} dumps, skipping")
                continue
            worst = check_against_evaluation(experiment_dir, split, folds,
                                              curves, grid, rhos, alpha)
            if worst > TOLERANCE:
                raise ValueError(
                    f'{experiment} {variant}: a curve gives a D_k {worst:.3g} '
                    f'away from {split}_evaluation.yaml, so it is not the '
                    f'curve that was scored; rerun the evaluation first')
            print(f"{experiment} {variant}: {len(folds)} folds, D_k matches "
                  f"the evaluation to {worst:.1e}")
            frames.extend(pd.DataFrame({
                'experiment': experiment, 'variant': variant,
                'fold': fold_name, 'cause': cause, 'rho': rhos, 'b': b})
                for (fold_name, cause), b in curves.items())

    if not frames:
        print("No curves; nothing written")
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pd.concat(frames, ignore_index=True)[COLUMNS].to_csv(args.out,
                                                         index=False)
    print(f"-> {args.out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
