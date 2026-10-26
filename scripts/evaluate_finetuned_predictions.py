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

The fixed-horizon concordance weights by a censoring curve fitted on the
fold's train dump, which therefore has to exist.

The cause-specific Cox baseline is scored on its exact curve, rebuilt from
the parameters it leaves beside its dumps (``TransEHR2.csc``): the
fixed-horizon metrics, the integrated Brier score and CR D-calibration all
read it, and the recalibrated split reads it shifted by the fold's dense
offsets and repaired once. The metrics read at the cuts read the dump,
which is the same curve there.

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
from TransEHR2.csc import (
    BASELINE_FILE, LP_FILE, OFFSETS_FILE, CSCCurve, read_csc_baseline,
    read_csc_lp, read_csc_offsets
)
from TransEHR2.survival import (
    DEFAULT_BRIER_INTEGRATION_DAYS, DEFAULT_CAUSES, DEFAULT_CR_DCAL_ALPHA,
    DEFAULT_CR_DCAL_CHI2_BINS, DEFAULT_CR_DCAL_N_RHO,
    DEFAULT_CR_DCAL_TEST_LEVEL, DEFAULT_CUTS_DAYS,
    DEFAULT_EVAL_HORIZONS_DAYS, RECAL_FLOOR, TimeGrid, _cr_dcal_values,
    aalen_johansen, brier_times, cause_specific_brier,
    cause_specific_concordance, cr_d_calibration_from_values,
    cr_dcal_chi2_from_values, event_times_days, format_days,
    integrated_brier, integrated_brier_to, interpolate_cif,
    lee_weighted_concordance,
    metric_label
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


def eval_horizons(cfg: Dict, grid: TimeGrid) -> List[float]:
    """``EVAL_HORIZONS_DAYS``: whole days, ascending, within the grid."""
    horizons = [float(t) for t in cfg.get('EVAL_HORIZONS_DAYS',
                                          list(DEFAULT_EVAL_HORIZONS_DAYS))]
    bad = [t for t in horizons if t <= 0 or t != round(t)
           or t > grid.horizon_days]
    if bad or horizons != sorted(set(horizons)):
        raise ValueError(
            f'EVAL_HORIZONS_DAYS must be distinct whole days, ascending, in '
            f'(0, {grid.horizon_days}]; got {horizons}')
    return horizons


def chi2_pass(p_values: List[float], test_level: float) -> float:
    """1.0 when every cause's chi-squared p clears ``test_level * K``.

    H0 is that the cause is calibrated, so a cause passes when H0 is *not*
    rejected. Alberge et al. (AISTATS 2026, appendix E) decide their
    per-cause test this way, with the level multiplied by the number of
    causes ``K`` rather than divided by it: the conservative direction for
    a pass, since dividing would make passing easier as causes are added.
    """
    level = test_level * len(p_values)
    return float(all(p > level for p in p_values))


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


def csc_reads(curve: CSCCurve, days: np.ndarray, grid: TimeGrid,
              horizons: List[float]) -> Dict:
    """Every read :func:`score` makes of a CSC curve, in one pass.

    ``F`` at each point of every Brier axis and at the last cut, as
    ``times`` and ``shared`` (n, n_causes, p), and at each patient's own
    ``days`` as ``rows`` (n, n_causes).
    """
    times = np.unique(np.concatenate(
        [brier_times(grid), [grid.horizon_days]]
        + [brier_times(grid, tau) for tau in horizons]))
    shared, rows = curve.read(times, days)
    return {'times': times, 'shared': shared, 'rows': rows}


def dcal_values(data: Dict, grid: TimeGrid, k: int, days: np.ndarray,
                csc: Optional[Dict] = None
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(F_k(t_i), F_k(tau), S(t_i))``, the three values CR D-cal reads.

    From the grid curve through Hyman, or, with ``csc`` (:func:`csc_reads`),
    from the CSC's exact curve: S(C) = 1 - sum_k F_k(C), floored as a
    recalibrated survival is, since it divides and a repaired curve can sum
    to one.
    """
    if csc is None:
        return _cr_dcal_values(data['cif'], data['pmf'], data['p_survive'],
                               days, k, grid)
    at = np.searchsorted(csc['times'], grid.horizon_days)
    assert csc['times'][at] == grid.horizon_days
    return (csc['rows'][:, k], csc['shared'][:, k, at],
            np.maximum(1.0 - csc['rows'].sum(axis=1), RECAL_FLOOR))


def load_csc_curve(experiment_dir: str, fold_name: str, split: str,
                   idx: np.ndarray, grid: TimeGrid) -> Optional[CSCCurve]:
    """The fold's CSC curve for the dump rows ``idx``, or None.

    None when no CSC parameters sit beside the dumps, so the model is read
    on its grid. ``test_recal_aj`` reads the test linear predictors shifted
    by the fold's dense offsets.
    """
    def path(template, part=None):
        return os.path.join(experiment_dir, fold_name, TASK,
                            template.format(task=TASK, split=part))

    baseline = read_csc_baseline(path(BASELINE_FILE), grid)
    if baseline is None:
        return None
    recal = split == 'test_recal_aj'
    lp_path = path(LP_FILE, 'test' if recal else split)
    lp = read_csc_lp(lp_path, grid)
    if lp is None:
        raise FileNotFoundError(f'{fold_name} has a CSC baseline hazard '
                                f'but no {lp_path}')
    at = pd.Index(lp['idx']).get_indexer(idx)
    if np.any(at < 0):
        raise ValueError(f'{lp_path} lacks {int(np.sum(at < 0))} '
                         f'episodes of the {split} dump')
    offsets = None
    if recal:
        offsets_path = path(OFFSETS_FILE)
        offsets = read_csc_offsets(offsets_path, grid)
        if offsets is None:
            raise FileNotFoundError(
                f'{fold_name} has a recalibrated CSC dump but no '
                f'{offsets_path}; rerun recalibrate_predictions.py')
    return CSCCurve(lp['lp'][at], baseline, offsets=offsets)


def score(data: Dict, grid: TimeGrid, dcal: Dict, horizons: List[float],
          censoring: Optional[Tuple[np.ndarray, np.ndarray]],
          csc: Optional[Dict] = None) -> Dict[str, float]:
    """Every metric one split of one fold contributes.

    Discrimination at each cut and over the grid, calibration by the
    IPCW Brier score and by CR D-calibration with its chi-squared test, and
    the counts that say how much of any of it to believe -- a concordance
    over forty deaths is a number, but not one to put a decimal place on.

    At each of ``horizons``, Lee's weighted concordance of ``F_k(tau)``,
    with ``censoring`` -- the training partition's ``(days, is_event)`` --
    for its weights; the integrated Brier score from discharge to ``tau``;
    the same score of the split's own Aalen-Johansen curve, fitted and
    scored on these episodes; and IPA, one minus their ratio.

    ``csc`` is :func:`csc_reads` of the CSC's exact curve. With it, those
    four, the integrated Brier score and CR D-calibration read that curve;
    without it, they read the grid curve through ``interpolate_cif``.

    ``dcal`` holds the CR_DCAL_* settings: ``alpha``, ``n_rho``,
    ``chi2_bins`` and ``test_level``. ``CR_DCal`` is the paper's sum over
    causes; ``CR_DCal_Chi2_Pass`` is 1.0 when every cause's p-value
    exceeds ``test_level * n_causes``, so that no cause's calibration is
    rejected (:func:`chi2_pass`).
    """
    bi, ci, ev, days = outcomes(data, grid)
    cif = data['cif']
    taus = brier_times(grid)

    def exact_at(times):
        """The CSC curve at ``times``, all among those csc_reads read."""
        at = np.searchsorted(csc['times'], times)
        assert np.array_equal(csc['times'][at], times)
        return csc['shared'][:, :, at]

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
        if csc is None:
            out[f'{key}_Integrated_Brier'] = integrated_brier(
                cause_specific_brier(interpolate_cif(cif[:, k, :], grid,
                                                     taus),
                                     days, ci, ev, k, taus), taus)
        else:
            out[f'{key}_Integrated_Brier'] = integrated_brier_to(
                exact_at(taus)[:, k, :], days, ci, ev, k, grid,
                grid.brier_integration_days, daily=True)
        out[f'{key}_Events'] = int(np.sum(ev & (ci == k)))

        for tau in horizons:
            label = format_days(tau).replace(' ', '')
            axis = brier_times(grid, tau)
            if csc is None:
                risk = interpolate_cif(cif[:, k, :], grid,
                                       np.array([tau]))[:, 0]
                ibs = integrated_brier_to(cif[:, k, :], days, ci, ev, k,
                                          grid, tau)
            else:
                risk = exact_at(np.array([tau]))[:, k, 0]
                ibs = integrated_brier_to(exact_at(axis)[:, k, :], days, ci,
                                          ev, k, grid, tau, daily=True)
            null = integrated_brier_to(
                aalen_johansen(days, ci, ev, k, axis)[None, :], days, ci, ev,
                k, grid, tau, daily=True)
            out[f'{key}_Cindex_Lee_at_{label}'] = lee_weighted_concordance(
                risk, days, ci, ev, k, tau, *censoring)
            out[f'{key}_IBS_to_{label}'] = ibs
            out[f'{key}_IBS_Null_to_{label}'] = null
            out[f'{key}_IPA_to_{label}'] = (1.0 - ibs / null if null > 0
                                            else float('nan'))

        values = (*dcal_values(data, grid, k, days, csc), ci, ev, k)
        out[f'{key}_CR_DCal'] = cr_d_calibration_from_values(
            *values, alpha=dcal['alpha'], n_rho=dcal['n_rho'])
        statistic, p_value = cr_dcal_chi2_from_values(
            *values, n_bins=dcal['chi2_bins'])
        out[f'{key}_CR_DCal_Chi2'] = statistic
        out[f'{key}_CR_DCal_Chi2_P'] = p_value

    labels = [metric_label(c) for c in grid.cause_names]
    out['CR_DCal'] = float(sum(out[f'{key}_CR_DCal'] for key in labels))
    out['CR_DCal_Chi2_Pass'] = chi2_pass(
        [out[f'{key}_CR_DCal_Chi2_P'] for key in labels], dcal['test_level'])

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
             'MODELLED_CAUSES, BRIER_INTEGRATION_DAYS, EVAL_HORIZONS_DAYS '
             'and the CR_DCAL_* keys. Without it the defaults are assumed, '
             'and a mismatch with the dumped columns is an error rather than '
             'a silent rescoring.')
    args = parser.parse_args()

    grid, cfg = load_config(args.experiment_config)
    horizons = eval_horizons(cfg, grid)
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
          f"{format_days(grid.brier_integration_days)}")
    print(f"Fixed horizons: "
          f"{', '.join(format_days(t) for t in horizons)}\n")

    def stem(fold_name, split):
        return os.path.join(experiment_dir, fold_name, TASK,
                            f'{TASK}_{split}')

    def predictions(fold_name, split):
        return read_predictions(
            f'{stem(fold_name, split)}_finetuned_output.csv', grid)

    def csc_curve(fold_name, split, idx):
        return load_csc_curve(experiment_dir, fold_name, split, idx, grid)

    censoring = {}

    def train_censoring(fold_name):
        """The train partition's ``(days, is_event)``, for Lee's G."""
        if fold_name not in censoring:
            train = predictions(fold_name, 'train')
            if train is None:
                raise FileNotFoundError(
                    f'{fold_name} has no train dump, which the censoring '
                    f'weights of Lee\'s concordance are fitted on')
            _, _, train_ev, train_days = outcomes(train, grid)
            censoring[fold_name] = (train_days, train_ev)
        return censoring[fold_name]

    def header(split, folds_present, is_csc):
        return {
            'experiment': args.experiment_name,
            'split': split,
            'task': TASK,
            'folds': folds_present,
            'time_grid_cuts_days': list(map(float, grid.cuts)),
            'modelled_causes': list(grid.cause_names),
            'brier_integration_days': grid.brier_integration_days,
            'eval_horizons_days': horizons,
            'csc_curve': is_csc,
            'cr_dcal': dcal,
        }

    for split in SPLITS:
        per_metric: Dict[str, List] = defaultdict(list)
        folds_present, has_csc = [], []

        for fold_name in fold_names:
            data = predictions(fold_name, split)
            if data is None:
                continue
            folds_present.append(fold_name)
            curve = csc_curve(fold_name, split, data['idx'])
            has_csc.append(curve is not None)
            reads = (None if curve is None else
                     csc_reads(curve, outcomes(data, grid)[3], grid,
                               horizons))
            fold_censoring = train_censoring(fold_name) if horizons else None
            for name, value in score(data, grid, dcal, horizons,
                                     fold_censoring, reads).items():
                per_metric[name].append(value)

        if not folds_present:
            print(f"{split}: no prediction files found, skipping")
            continue
        # One curve per split: a mean over folds scored on different
        # readings of the model would describe neither.
        if len(set(has_csc)) > 1:
            raise ValueError(
                f'{split}: CSC parameters sit beside the dump in some folds '
                f'and not others ({dict(zip(folds_present, has_csc))})')

        out_path = os.path.join(experiment_dir, f'{split}_evaluation.yaml')
        write_evaluation(out_path,
                         header(split, folds_present, has_csc[0]),
                         per_metric)

        mean_c = np.nanmean(per_metric['Mean_Cindex'])
        print(f"{split}: {len(folds_present)} folds, "
              f"mean C-index {mean_c:.4f}, CR D-cal "
              f"{np.nanmean(per_metric['CR_DCal']):.4f} -> {out_path}")


if __name__ == '__main__':
    main()
