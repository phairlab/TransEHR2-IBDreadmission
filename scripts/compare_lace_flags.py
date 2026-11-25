"""Compare flagging patients by LACE with flagging them by a model's CIF
(plan 2026-11-18, Task C).

For one experiment, fold by fold, at each horizon ``H`` of
``configs/reports/part1_lace_comparison.yaml``:

* **Outcome**: the first event is readmission or death by ``H``
  (``TransEHR2.flagging.composite_outcome``), IPCW-weighted with ``G``
  fitted on the scored patients.
* **Scores**: LACE as the AMA sheet scores it (IBDdataprep's ``lace.py``,
  ``lace_ama.csv``), and the model's composite risk ``F_readmission(H) +
  F_death(H)``, read on the Hyman-interpolated curve (or the CSC's exact
  curve) of the raw, unrecalibrated predictions.
* **Policies**: LACE >= each cutoff (7 and 14), and for each cutoff the
  model thresholded two ways on the fold's ``val`` patients, then frozen
  and applied to its ``test`` patients: (a) flag the same fraction of
  ``val`` as LACE does; (b) the largest threshold whose IPCW sensitivity
  on ``val`` reaches LACE's.
* **Metrics on test**: each policy's IPCW sensitivity, specificity, PPV and
  flag rate; each score's IPCW ROC curve and AUC; the decision curves of
  the model flagging ``F >= p_t``; the model flagging ``q(F) >= level``
  as a fixed policy for each level of ``MODEL_QUANTILE_FLAGS`` (``q`` the
  percentile rank of F among the fold's ``val`` patients, so ``q(F) >=
  0.8`` flags about the top 20%); LACE >= each cutoff as a fixed policy;
  treat-all and treat-none.

A full-cohort experiment's one fold, ``full``, has fold0's test patients as
its val and the held-out set as its test, so the same run scores the
external evaluation.

Each population of the config (every index episode, and the DAD index
episodes) is scored on its own, thresholds included. Writes into
``--out_dir``:

* ``{experiment}_lace_comparison_{population}.yaml``: a header with each
  fold's thresholds, then every metric as a per-fold list, keyed
  ``{H}_{policy}_{metric}`` (``30d_LACE_ge7_Sensitivity``,
  ``30d_Model_rate_LACE7_Threshold``, ``30d_Model_AUC``, ...);
* ``{experiment}_lace_roc.csv``: the ROC curves, LACE at each of its
  integer values, the model on a fixed grid of false-positive rates;
* ``{experiment}_lace_decision.csv``: the decision curves.

Everything written is an aggregate: no patient's score or outcome. The
dumps' outcomes are checked against ``labels.csv`` at their ``idx``, which
refuses dumps keyed by extracted position (before the 2026-11-16 fix).

Usage:
    python scripts/compare_lace_flags.py <experiment_name> \\
        --experiment_config <yaml> --lace data/lace_ama.csv \\
        --labels data/labels.csv --out_dir tables/<run>/lace \\
        [--model_dir ./models] [--config <lace comparison yaml>]
"""

import argparse
import os

from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

import _path  # noqa: F401  (repository root on sys.path)

from TransEHR2 import flagging
from TransEHR2.survival import format_days, interpolate_cif

from evaluate_finetuned_predictions import (
    TASK, get_fold_names, load_config, load_csc_curve, outcomes,
    read_predictions
)


DEFAULT_CONFIG = os.path.join(
    _path.ROOT, 'TransEHR2', 'configs', 'reports',
    'part1_lace_comparison.yaml')


def horizon_label(h: float) -> str:
    return format_days(float(h)).replace(' ', '')


def quantile_policy(level: float) -> str:
    """``Model_q80`` for the policy flagging ``q(F) >= 0.8``."""
    return f'Model_q{round(100 * float(level)):g}'


def composite_risk(data: Dict, grid, causes: List[int], horizon: float,
                   curve=None) -> np.ndarray:
    """``sum_k F_k(H)`` over ``causes``: Hyman for a grid model, the exact
    curve for a CSC."""
    if curve is not None:
        at, _ = curve.read(np.array([horizon]))
        return at[:, causes, 0].sum(axis=1)
    return sum(interpolate_cif(data['cif'][:, k, :], grid,
                               np.array([horizon]))[:, 0] for k in causes)


def check_outcomes(data: Dict, labels: pd.DataFrame, where: str) -> None:
    """Refuse a dump whose outcomes are not ``labels.csv``'s at its idx."""
    idx = np.asarray(data['idx'], dtype=np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= len(labels)):
        raise ValueError(f'{where}: idx outside labels.csv')
    rows = labels.iloc[idx]
    if not (np.allclose(rows['TIME_TO_EVENT'].to_numpy(dtype=float),
                        np.asarray(data['time_to_event'], dtype=float))
            and np.array_equal(rows['EVENT_TYPE'].to_numpy(dtype=np.int64),
                               np.asarray(data['event_type'],
                                          dtype=np.int64))):
        raise ValueError(
            f'{where}: the outcomes differ from labels.csv at the dump\'s '
            f'idx. The dump is keyed by extracted position (before the '
            f'2026-11-16 fix) or is of another build; re-dump it.')


def lace_at(lace: pd.DataFrame, idx: np.ndarray, where: str) -> pd.DataFrame:
    """``lace_ama.csv``'s rows for the dump rows ``idx``, in order."""
    missing = ~np.isin(idx, lace.index)
    if missing.any():
        raise ValueError(f'{where}: {int(missing.sum())} episode(s) with no '
                         f'row in the LACE file')
    return lace.loc[idx]


def split_inputs(experiment_dir, fold, split, grid, labels, lace, causes,
                 horizons):
    """One split's outcomes, LACE and composite risks at each horizon."""
    path = os.path.join(experiment_dir, fold, TASK,
                        f'{TASK}_{split}_finetuned_output.csv')
    data = read_predictions(path, grid)
    if data is None:
        raise FileNotFoundError(f'{path} does not exist')
    where = f'{fold} {split}'
    check_outcomes(data, labels, where)
    rows = lace_at(lace, np.asarray(data['idx'], dtype=np.int64), where)
    curve = load_csc_curve(experiment_dir, fold, split, data['idx'], grid)
    _, ci, ev, days = outcomes(data, grid)
    return {
        'days': days, 'cause_index': ci, 'is_event': ev,
        'lace': rows['LACE'].to_numpy(dtype=np.float64),
        'index_type': rows['INDEX_TYPE'].to_numpy(),
        'risk': {h: composite_risk(data, grid, causes, h, curve)
                 for h in horizons},
        'csc': curve is not None,
    }


def subset(split: Dict, keep: np.ndarray, horizons) -> Dict:
    out = {key: split[key][keep] for key in
           ('days', 'cause_index', 'is_event', 'lace', 'index_type')}
    out['risk'] = {h: split['risk'][h][keep] for h in horizons}
    return out


def score_fold(val: Dict, test: Dict, causes: List[int], cfg: Dict,
               p_t: np.ndarray, fpr_grid: np.ndarray):
    """``(metrics, thresholds, roc rows, decision rows)`` of one fold and
    population."""
    metrics, thresholds, roc, decision = {}, {}, [], []
    for h in cfg['HORIZONS_DAYS']:
        label = horizon_label(h)
        v_pos, v_w = flagging.composite_outcome(
            val['days'], val['cause_index'], val['is_event'], causes, h)
        t_pos, t_w = flagging.composite_outcome(
            test['days'], test['cause_index'], test['is_event'], causes, h)
        v_risk, t_risk = val['risk'][h], test['risk'][h]
        metrics[f'{label}_Positives_N'] = int(t_pos.sum())
        metrics[f'{label}_Scored_N'] = int(t_pos.size)

        policies = {}
        for cut in cfg['LACE_CUTOFFS']:
            policies[f'LACE_ge{cut}'] = test['lace'] >= cut
            rate = float(np.mean(val['lace'] >= cut))
            target = flagging.rates(val['lace'] >= cut, v_pos,
                                    v_w)['Sensitivity']
            for rule, threshold in (
                    ('rate', flagging.matched_rate_threshold(v_risk, rate)),
                    ('sens', flagging.matched_sensitivity_threshold(
                        v_risk, v_pos, v_w, target))):
                name = f'Model_{rule}_LACE{cut}'
                thresholds[f'{label}_{name}'] = threshold
                metrics[f'{label}_{name}_Threshold'] = (
                    float('nan') if threshold is None else threshold)
                policies[name] = (np.zeros(t_risk.size, dtype=bool)
                                  if threshold is None
                                  else t_risk >= threshold)
        for name, flag in policies.items():
            for metric, value in flagging.rates(flag, t_pos, t_w).items():
                metrics[f'{label}_{name}_{metric}'] = value

        for name, score in (('LACE', test['lace']), ('Model', t_risk)):
            curve = flagging.roc_curve(score, t_pos, t_w)
            metrics[f'{label}_{name}_AUC'] = curve['auc']
            if name == 'LACE':
                for c, x, y in zip(curve['threshold'], curve['fpr'],
                                   curve['tpr']):
                    roc.append({'horizon': label, 'score': name,
                                'threshold': c, 'fpr': x, 'tpr': y})
            else:
                # The step curve's highest tpr at each grid fpr.
                at = np.searchsorted(curve['fpr'], fpr_grid, side='right') - 1
                for x, y in zip(fpr_grid, curve['tpr'][np.maximum(at, 0)]):
                    roc.append({'horizon': label, 'score': name,
                                'threshold': np.nan, 'fpr': x, 'tpr': y})

        curves = {'Model': flagging.decision_curve(t_risk, t_pos, t_w, p_t),
                  'Treat_all': flagging.net_benefit(
                      np.ones(t_risk.size, dtype=bool), t_pos, t_w, p_t),
                  'Treat_none': np.zeros(p_t.size)}
        for cut in cfg['LACE_CUTOFFS']:
            curves[f'LACE_ge{cut}'] = flagging.net_benefit(
                test['lace'] >= cut, t_pos, t_w, p_t)
        rank = flagging.percentile_rank(v_risk, t_risk)
        for level in cfg['MODEL_QUANTILE_FLAGS']:
            curves[quantile_policy(level)] = flagging.net_benefit(
                rank >= level, t_pos, t_w, p_t)
        for policy, values in curves.items():
            for p, value in zip(p_t, values):
                decision.append({'horizon': label, 'policy': policy,
                                 'p_t': p, 'net_benefit': value})
    return metrics, thresholds, roc, decision


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('experiment_name')
    parser.add_argument('--model_dir', default='./models')
    parser.add_argument('--experiment_config', required=True,
                        help='Experiment YAML, for the grid and the causes')
    parser.add_argument('--config', default=DEFAULT_CONFIG,
                        help='LACE comparison config [default: %(default)s]')
    parser.add_argument('--lace', required=True,
                        help='lace_ama.csv from IBDdataprep\'s lace.py')
    parser.add_argument('--labels', required=True,
                        help='labels.csv, whose row is a dump\'s idx')
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args(argv)

    grid, _ = load_config(args.experiment_config)
    with open(args.config) as f_in:
        cfg = yaml.safe_load(f_in)
    missing = [c for c in cfg['COMPOSITE_CAUSES']
               if c not in grid.cause_names]
    if missing:
        raise SystemExit(f'{args.experiment_name} does not model {missing}, '
                         f'which the composite outcome needs')
    causes = [grid.cause_names.index(c) for c in cfg['COMPOSITE_CAUSES']]
    horizons = [float(h) for h in cfg['HORIZONS_DAYS']]
    if max(horizons) > grid.horizon_days:
        raise SystemExit('a horizon lies past the grid')
    nb = cfg['NET_BENEFIT_THRESHOLDS']
    p_t = np.round(np.arange(nb['START'], nb['STOP'] + nb['STEP'] / 2,
                             nb['STEP']), 10)
    fpr_grid = np.linspace(0.0, 1.0, int(cfg['ROC_GRID_POINTS']))

    labels = pd.read_csv(args.labels, usecols=['TIME_TO_EVENT', 'EVENT_TYPE'])
    lace = pd.read_csv(args.lace, dtype={'INDEX_TYPE': str})
    lace.index = lace['idx'].astype(np.int64)

    experiment_dir = os.path.join(args.model_dir, args.experiment_name)
    folds = get_fold_names(experiment_dir)
    if not folds:
        raise SystemExit(f'no folds under {experiment_dir}')

    per_population = {pop: (defaultdict(list), {}) for pop in
                      cfg['POPULATIONS']}
    roc_rows, decision_rows = [], []
    is_csc = set()
    for fold in folds:
        splits = {s: split_inputs(experiment_dir, fold, s, grid, labels,
                                  lace, causes, horizons)
                  for s in ('val', 'test')}
        is_csc.add(splits['test']['csc'])
        for pop, index_type in cfg['POPULATIONS'].items():
            parts = [s if index_type is None else
                     subset(s, s['index_type'] == index_type, horizons)
                     for s in (splits['val'], splits['test'])]
            metrics, thresholds, roc, decision = score_fold(
                *parts, causes, cfg, p_t, fpr_grid)
            per_metric, fold_thresholds = per_population[pop]
            for key, value in metrics.items():
                per_metric[key].append(value)
            fold_thresholds[fold] = thresholds
            for row in roc:
                roc_rows.append({'fold': fold, 'population': pop, **row})
            for row in decision:
                decision_rows.append({'fold': fold, 'population': pop,
                                      **row})
        print(f'{fold}: scored')
    if len(is_csc) > 1:
        raise ValueError('CSC parameters sit beside some folds\' dumps and '
                         'not others')
    csc = is_csc.pop()

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, args.experiment_name)
    for pop, (per_metric, fold_thresholds) in per_population.items():
        header = {
            'experiment': args.experiment_name, 'population': pop,
            'index_type': cfg['POPULATIONS'][pop], 'folds': folds,
            'horizons_days': horizons,
            'composite_causes': list(cfg['COMPOSITE_CAUSES']),
            'lace_cutoffs': list(cfg['LACE_CUTOFFS']),
            'model_quantile_flags': [float(q) for q in
                                     cfg['MODEL_QUANTILE_FLAGS']],
            'lace': 'AMA sheet (lace.py lace_ama.csv)',
            'model_score': 'sum of F_k(H) over the composite causes, raw '
                           'predictions',
            'csc_curve': csc,
            'thresholds': fold_thresholds,
        }
        payload = {**header, **{k: [float(v) for v in vals]
                                for k, vals in per_metric.items()}}
        path = f'{stem}_lace_comparison_{pop}.yaml'
        with open(path, 'w') as f_out:
            yaml.dump(payload, f_out, default_flow_style=False,
                      sort_keys=False)
        print(f'{pop}: -> {path}')
    pd.DataFrame(roc_rows).to_csv(f'{stem}_lace_roc.csv', index=False)
    pd.DataFrame(decision_rows).to_csv(f'{stem}_lace_decision.csv',
                                       index=False)
    print(f'ROC and decision curves -> {stem}_lace_{{roc,decision}}.csv')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
