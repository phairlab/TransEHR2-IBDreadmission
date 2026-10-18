"""Score the CSVs ``dump_transehr2_deephit_predictions.py`` writes, fold by fold.

Separate from the training run on purpose: scoring a saved prediction is
cheap, repeatable and needs no GPU, so a change of metric or of horizon
costs a re-read rather than a re-fit. The training run reports its own
scores too, but those are computed inside a loop that also selects on them;
these are computed once, from files, and are the ones to quote.

Writes one YAML per split at ``{model_dir}/{experiment}/{split}_evaluation.yaml``,
each metric a list of per-fold values in fold order.

``test_recal_aj`` is scored as one more split when its dump exists: the
corrected test predictions ``recalibrate_predictions.py`` writes beside each
fold's dump, in the dump's own columns. It goes through the same routines
under the same keys, so ``test_recal_aj_evaluation.yaml`` reads against
``test_evaluation.yaml`` key for key; how each corrected dump was made is in
that script's ``test_recal_aj_record.yaml``.

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
from typing import Dict, List, Optional, Tuple

import _path  # noqa: F401  (repository root on sys.path)

from TransEHR2.cli import TUNING_FOLD
from TransEHR2.survival import (
    DEFAULT_BRIER_INTEGRATION_DAYS, DEFAULT_CAUSES, DEFAULT_CR_DCAL_ALPHA,
    DEFAULT_CR_DCAL_CHI2_BINS, DEFAULT_CR_DCAL_N_RHO,
    DEFAULT_CR_DCAL_TEST_LEVEL, DEFAULT_CUTS_DAYS, TimeGrid, brier_times,
    cause_specific_brier, cause_specific_concordance, cr_d_calibration,
    cr_dcal_chi2, event_times_days, format_days, integrated_brier,
    interpolate_cif, metric_label
)


# The last is the recalibrated test split, when recalibrate_predictions.py
# has written it; a split with no dump in any fold is skipped.
SPLITS = ('train', 'val', 'test', 'test_recal_aj')
TASK = 'competing_risks'


def get_fold_names(experiment_dir: str,
                   exclude: Optional[List[str]] = None) -> List[str]:
    """Fold subdirectories of an experiment's model directory, sorted.

    ``TUNING_FOLD`` is excluded by default. These are the numbers that get
    quoted, and the fold the hyperparameters were chosen on is not one of
    them. It only appears here at all if it was dumped deliberately.
    """
    exclude = [TUNING_FOLD] if exclude is None else exclude
    folds = [d for d in os.listdir(experiment_dir)
             if d not in exclude and re.match(r'fold\d+$', d)
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

    # pandas writes the shortest repr that round-trips, but its default
    # parser can land an ulp off it; the dump is only exact read this way.
    frame = pd.read_csv(path, float_precision='round_trip')
    expected = {
        prefix: [
            [f"{prefix}_{cause}_{format_days(float(cut)).replace(' ', '')}"
             for cut in grid.cuts]
            for cause in grid.cause_names
        ]
        for prefix in ('cif', 'pmf')
    }
    missing = [c for cols in expected['cif'] for c in cols
               if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} is missing {missing}. It was written against a "
            f"different TIME_GRID_CUTS_DAYS than the one being scored "
            f"({list(grid.cuts)}); re-dump, or pass the config that "
            f"produced it."
        )
    missing = [c for cols in expected['pmf'] for c in cols
               if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} has no pmf columns ({missing[0]}, ...). It predates "
            f"their addition to dump_transehr2_deephit_predictions.py, which the "
            f"survival in the D-calibration weights is summed from; "
            f"re-dump it."
        )

    def stack(prefix):
        return np.stack([frame[cols].to_numpy() for cols in expected[prefix]],
                        axis=1)

    return {
        'cif': stack('cif'),                          # (n, n_causes, n_bins)
        'pmf': stack('pmf'),                          # (n, n_causes, n_bins)
        'idx': frame['idx'].to_numpy(),
        'time_to_event': frame['time_to_event'].to_numpy(),
        'event_type': frame['event_type'].to_numpy(),
        'p_survive': frame['p_survive'].to_numpy(),
    }


def outcomes(data: Dict, grid: TimeGrid):
    """``(bin_index, cause_index, is_event, days)`` of one split, as numpy."""
    import torch

    bin_index, cause_index, is_event = grid.discretize(
        torch.as_tensor(data['time_to_event'], dtype=torch.float64),
        torch.as_tensor(data['event_type']))
    return (bin_index.numpy(), cause_index.numpy(),
            is_event.numpy().astype(bool),
            event_times_days(data['time_to_event'], grid))


def load_config(path: Optional[str]) -> Tuple[TimeGrid, Dict]:
    """The grid and the experiment config, or the defaults without a path."""
    cfg = {}
    if path:
        with open(path, 'r') as f_in:
            cfg = yaml.safe_load(f_in)
    grid = TimeGrid(cfg.get('TIME_GRID_CUTS_DAYS', list(DEFAULT_CUTS_DAYS)),
                    cfg.get('MODELLED_CAUSES', list(DEFAULT_CAUSES)),
                    cfg.get('BRIER_INTEGRATION_DAYS',
                            DEFAULT_BRIER_INTEGRATION_DAYS))
    return grid, cfg


def write_evaluation(out_path: str, header: Dict,
                     per_metric: Dict[str, List]) -> None:
    payload = {**header,
               **{name: [float(v) for v in values]
                  for name, values in per_metric.items()}}
    with open(out_path, 'w') as f_out:
        yaml.dump(payload, f_out, default_flow_style=False, indent=2,
                  sort_keys=False)


def score(data: Dict, grid: TimeGrid, dcal: Dict) -> Dict[str, float]:
    """Every metric one split of one fold contributes.

    Discrimination at each cut and over the grid, calibration by the
    IPCW Brier score and by CR D-calibration with its chi-squared test, and
    the counts that say how much of any of it to believe -- a concordance
    over forty deaths is a number, but not one to put a decimal place on.

    ``dcal`` holds the CR_DCAL_* settings: ``alpha``, ``n_rho``,
    ``chi2_bins`` and ``test_level``. ``CR_DCal`` is the paper's sum over
    causes; ``CR_DCal_Chi2_Pass`` is 1.0 when every cause's p-value clears
    the Bonferroni level ``test_level / n_causes``.
    """
    bi, ci, ev, days = outcomes(data, grid)
    cif = data['cif']
    taus = brier_times(grid)

    out: Dict[str, float] = {}
    for k, cause in enumerate(grid.cause_names):
        key = metric_label(cause)
        out[f'{key}_Cindex'] = cause_specific_concordance(cif, bi, ci, ev, k)
        for b, cut in enumerate(grid.cuts):
            label = format_days(float(cut)).replace(' ', '')
            out[f'{key}_Cindex_at_{label}'] = cause_specific_concordance(
                cif, bi, ci, ev, k, horizon_bin=b)

        # Two readings of the same curve: at the cuts, where the head
        # predicted and no interpolation is involved, and over the dense
        # axis the integral runs on. The per-cut scores stay at every cut,
        # including those past the integration bound -- they are what the
        # model said there, and nothing is integrated over them.
        at_cuts = cause_specific_brier(cif[:, k, :], days, ci, ev, k,
                                       grid.cuts)
        for b, cut in enumerate(grid.cuts):
            label = format_days(float(cut)).replace(' ', '')
            out[f'{key}_Brier_at_{label}'] = float(at_cuts[b])
        out[f'{key}_Integrated_Brier'] = integrated_brier(
            cause_specific_brier(interpolate_cif(cif[:, k, :], grid, taus),
                                 days, ci, ev, k, taus), taus)
        out[f'{key}_Events'] = int(np.sum(ev & (ci == k)))

        prediction = (data['cif'], data['pmf'], data['p_survive'], days, ci,
                      ev, k, grid)
        out[f'{key}_CR_DCal'] = cr_d_calibration(
            *prediction, alpha=dcal['alpha'], n_rho=dcal['n_rho'])
        statistic, p_value = cr_dcal_chi2(*prediction,
                                          n_bins=dcal['chi2_bins'])
        out[f'{key}_CR_DCal_Chi2'] = statistic
        out[f'{key}_CR_DCal_Chi2_P'] = p_value

    labels = [metric_label(c) for c in grid.cause_names]
    out['CR_DCal'] = float(sum(out[f'{key}_CR_DCal'] for key in labels))
    level = dcal['test_level'] / grid.n_causes
    out['CR_DCal_Chi2_Pass'] = float(
        all(out[f'{key}_CR_DCal_Chi2_P'] > level for key in labels))

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
        help='Experiment YAML, read for TIME_GRID_CUTS_DAYS, '
             'MODELLED_CAUSES, BRIER_INTEGRATION_DAYS and the CR_DCAL_* '
             'keys. Without it the defaults are assumed, and a mismatch with '
             'the dumped columns is an error rather than a silent rescoring.')
    args = parser.parse_args()

    grid, cfg = load_config(args.experiment_config)
    dcal = {
        'alpha': float(cfg.get('CR_DCAL_ALPHA', DEFAULT_CR_DCAL_ALPHA)),
        'n_rho': int(cfg.get('CR_DCAL_N_RHO', DEFAULT_CR_DCAL_N_RHO)),
        'chi2_bins': int(cfg.get('CR_DCAL_CHI2_BINS',
                                 DEFAULT_CR_DCAL_CHI2_BINS)),
        'test_level': float(cfg.get('CR_DCAL_TEST_LEVEL',
                                    DEFAULT_CR_DCAL_TEST_LEVEL)),
    }

    experiment_dir = os.path.join(args.model_dir, args.experiment_name)
    if not os.path.isdir(experiment_dir):
        raise FileNotFoundError(f'{experiment_dir} does not exist')

    fold_names = get_fold_names(experiment_dir)
    if not fold_names:
        raise FileNotFoundError(f'No fold directories under {experiment_dir}')
    print(f"Scoring {len(fold_names)} folds on a {grid.n_bins}-bin grid: "
          f"{', '.join(grid.labels())}")
    print(f"Integrated Brier score: discharge to "
          f"{format_days(grid.brier_integration_days)}\n")

    def predictions(fold_name, split):
        return read_predictions(os.path.join(
            experiment_dir, fold_name, TASK,
            f'{TASK}_{split}_finetuned_output.csv'), grid)

    def header(split, folds_present):
        return {
            'experiment': args.experiment_name,
            'split': split,
            'task': TASK,
            'folds': folds_present,
            'time_grid_cuts_days': list(map(float, grid.cuts)),
            'brier_integration_days': grid.brier_integration_days,
            'cr_dcal': dcal,
        }

    for split in SPLITS:
        per_metric: Dict[str, List] = defaultdict(list)
        folds_present = []

        for fold_name in fold_names:
            data = predictions(fold_name, split)
            if data is None:
                continue
            folds_present.append(fold_name)
            for name, value in score(data, grid, dcal).items():
                per_metric[name].append(value)

        if not folds_present:
            print(f"{split}: no prediction files found, skipping")
            continue

        out_path = os.path.join(experiment_dir, f'{split}_evaluation.yaml')
        write_evaluation(out_path, header(split, folds_present), per_metric)

        mean_c = np.nanmean(per_metric['Mean_Cindex'])
        print(f"{split}: {len(folds_present)} folds, "
              f"mean C-index {mean_c:.4f}, CR D-cal "
              f"{np.nanmean(per_metric['CR_DCal']):.4f} -> {out_path}")


if __name__ == '__main__':
    main()
