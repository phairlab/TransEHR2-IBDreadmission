"""Score the CSVs ``dump_finetuned_predictions.py`` writes, fold by fold.

Separate from the training run on purpose: scoring a saved prediction is
cheap, repeatable and needs no GPU, so a change of metric or of horizon
costs a re-read rather than a re-fit. The training run reports its own
scores too, but those are computed inside a loop that also selects on them;
these are computed once, from files, and are the ones to quote.

Writes one YAML per split at ``{model_dir}/{experiment}/{split}_evaluation.yaml``,
each metric a list of per-fold values in fold order.

Usage:
    python scripts/evaluate_finetuned_predictions.py <experiment_name> \\
        [--model_dir ./models] [--experiment_config <yaml>]
"""

import argparse
import numpy as np
import os
import pandas as pd
import re
import yaml

from collections import defaultdict
from typing import Dict, List, Optional

import _path  # noqa: F401  (repository root on sys.path)

from TransEHR2.survival import (
    DEFAULT_CAUSES, DEFAULT_CUTS_DAYS, TimeGrid, cause_specific_brier,
    cause_specific_concordance, format_days, integrated_brier,
    metric_label
)


SPLITS = ('train', 'val', 'test')
TASK = 'competing_risks'


def get_fold_names(experiment_dir: str) -> List[str]:
    """Fold subdirectories of an experiment's model directory, sorted."""
    folds = [d for d in os.listdir(experiment_dir)
             if re.match(r'fold\d+$', d)
             and os.path.isdir(os.path.join(experiment_dir, d))]
    folds.sort(key=lambda d: int(d[4:]))
    return folds


def read_predictions(path: str, grid: TimeGrid) -> Optional[Dict]:
    """Read one split's CSV back into the arrays the metrics want.

    The cumulative incidence columns are matched by name rather than by
    position, so a file written against a different grid is caught here
    instead of being silently reshaped into one.
    """
    if not os.path.exists(path):
        return None

    frame = pd.read_csv(path)
    expected = [
        [f"cif_{cause}_{format_days(float(cut)).replace(' ', '')}"
         for cut in grid.cuts]
        for cause in grid.cause_names
    ]
    missing = [c for cols in expected for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} is missing {missing}. It was written against a "
            f"different TIME_GRID_CUTS_DAYS than the one being scored "
            f"({list(grid.cuts)}); re-dump, or pass the config that "
            f"produced it."
        )

    cif = np.stack([frame[cols].to_numpy() for cols in expected], axis=1)
    return {
        'cif': cif,                                   # (n, n_causes, n_bins)
        'time_to_event': frame['time_to_event'].to_numpy(),
        'event_type': frame['event_type'].to_numpy(),
        'p_survive': frame['p_survive'].to_numpy(),
    }


def score(data: Dict, grid: TimeGrid) -> Dict[str, float]:
    """Every metric one split of one fold contributes.

    Discrimination at each cut and over the grid, calibration by the
    IPCW Brier score, and the counts that say how much of either to
    believe -- a concordance over forty deaths is a number, but not one to
    put a decimal place on.
    """
    import torch

    bin_index, cause_index, is_event = grid.discretize(
        torch.as_tensor(data['time_to_event'], dtype=torch.float64),
        torch.as_tensor(data['event_type']))
    bi = bin_index.numpy()
    ci = cause_index.numpy()
    ev = is_event.numpy().astype(bool)
    cif = data['cif']

    out: Dict[str, float] = {}
    for k, cause in enumerate(grid.cause_names):
        key = metric_label(cause)
        out[f'{key}_Cindex'] = cause_specific_concordance(cif, bi, ci, ev, k)
        for b, cut in enumerate(grid.cuts):
            label = format_days(float(cut)).replace(' ', '')
            out[f'{key}_Cindex_at_{label}'] = cause_specific_concordance(
                cif, bi, ci, ev, k, horizon_bin=b)
        brier = cause_specific_brier(cif, bi, ci, ev, k)
        for b, cut in enumerate(grid.cuts):
            label = format_days(float(cut)).replace(' ', '')
            out[f'{key}_Brier_at_{label}'] = float(brier[b])
        out[f'{key}_Integrated_Brier'] = integrated_brier(brier, grid)
        out[f'{key}_Events'] = int(np.sum(ev & (ci == k)))

    finite = [out[f'{metric_label(c)}_Cindex'] for c in grid.cause_names]
    finite = [v for v in finite if np.isfinite(v)]
    out['Mean_Cindex'] = float(np.mean(finite)) if finite else float('nan')
    out['Censored'] = int(np.sum(~ev))
    out['Episodes'] = int(bi.size)
    return out


def main():
    parser = argparse.ArgumentParser(
        description='Compute evaluation metrics from finetuned prediction CSVs'
    )
    parser.add_argument('experiment_name', type=str,
                        help='Name of the experiment under model_dir')
    parser.add_argument('--model_dir', type=str, default='./models',
                        help='Root directory containing the experiment')
    parser.add_argument(
        '--experiment_config', type=str, default=None,
        help='Experiment YAML, read only for TIME_GRID_CUTS_DAYS. Without '
             'it the default grid is assumed, and a mismatch with the '
             'dumped columns is an error rather than a silent rescoring.')
    args = parser.parse_args()

    cuts = list(DEFAULT_CUTS_DAYS)
    causes = list(DEFAULT_CAUSES)
    if args.experiment_config:
        with open(args.experiment_config, 'r') as f_in:
            cfg = yaml.safe_load(f_in)
        cuts = cfg.get('TIME_GRID_CUTS_DAYS', cuts)
        causes = cfg.get('MODELLED_CAUSES', causes)
    grid = TimeGrid(cuts, causes)

    experiment_dir = os.path.join(args.model_dir, args.experiment_name)
    if not os.path.isdir(experiment_dir):
        raise FileNotFoundError(f'{experiment_dir} does not exist')

    fold_names = get_fold_names(experiment_dir)
    if not fold_names:
        raise FileNotFoundError(f'No fold directories under {experiment_dir}')
    print(f"Scoring {len(fold_names)} folds on a {grid.n_bins}-bin grid: "
          f"{', '.join(grid.labels())}\n")

    for split in SPLITS:
        per_metric: Dict[str, List] = defaultdict(list)
        folds_present = []

        for fold_name in fold_names:
            path = os.path.join(
                experiment_dir, fold_name, TASK,
                f'{TASK}_{split}_finetuned_output.csv')
            data = read_predictions(path, grid)
            if data is None:
                continue
            folds_present.append(fold_name)
            for name, value in score(data, grid).items():
                per_metric[name].append(value)

        if not folds_present:
            print(f"{split}: no prediction files found, skipping")
            continue

        payload = {
            'experiment': args.experiment_name,
            'split': split,
            'task': TASK,
            'folds': folds_present,
            'time_grid_cuts_days': list(map(float, grid.cuts)),
            **{name: [float(v) for v in values]
               for name, values in per_metric.items()},
        }
        out_path = os.path.join(experiment_dir, f'{split}_evaluation.yaml')
        with open(out_path, 'w') as f_out:
            yaml.dump(payload, f_out, default_flow_style=False, indent=2,
                      sort_keys=False)

        mean_c = np.nanmean(per_metric['Mean_Cindex'])
        print(f"{split}: {len(folds_present)} folds, "
              f"mean C-index {mean_c:.4f} -> {out_path}")


if __name__ == '__main__':
    main()
