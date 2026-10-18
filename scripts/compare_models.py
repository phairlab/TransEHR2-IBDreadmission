"""Compare the Part 1 models fold by fold, one table per cause.

Reads the per-fold metric lists ``evaluate_finetuned_predictions.py`` writes
and the comparisons named in ``configs/reports/part1_comparisons.yaml``. Each
comparison is a paired, two-sided Nadeau-Bengio corrected resampled t test
over the folds (``TransEHR2.stats``), on n_test / n_train from the config and
k - 1 degrees of freedom. Means are reported with the same corrected SE.

The P values of one cause's table form one Benjamini-Hochberg family: every
tested metric crossed with every comparison in it. Metrics the config lists as
reported only (IPA) get means and differences but no test, and stay out of the
family.

The CR D-calibration chi-squared is a goodness-of-fit test of one model, not a
comparison, so it is reported per model in a table of its own and kept out of
the family too. H0 is uniformity of the cause's probability-integral values, so
a cause passes a fold when H0 is *not* rejected, at p > level * K with K the
number of modelled causes; the pass is worked out here from the per-fold p
values. A fold passes overall when every cause does. A NaN P is reported as it
is, with a note, and the fold counts as neither passing nor failing.

Evaluations are only compared when their headers agree on the time grid, the
horizons, the causes, the folds, and the D-calibration alpha and chi-squared
bin count. Anything else is refused rather than compared: a paired test pairs
folds by position, and a difference between two models scored on different
grids or D-calibration settings is not one either model's quality explains.

Writes ``{cause}_comparisons.csv`` and ``dcal_chi2.csv`` into ``--out_dir``.

Usage:
    python scripts/compare_models.py --out_dir <dir> \\
        [--model_dir ./models] \\
        [--config TransEHR2/configs/reports/part1_comparisons.yaml]
"""

import argparse
import os
import yaml

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import _path  # noqa: F401  (repository root on sys.path)

from TransEHR2.stats import (
    benjamini_hochberg, corrected_resampled_ttest, corrected_standard_error
)
from TransEHR2.survival import format_days, metric_label


DEFAULT_CONFIG = os.path.join(
    _path.ROOT, 'TransEHR2', 'configs', 'reports', 'part1_comparisons.yaml')

# Header fields two evaluations must share to be compared.
HEADER_KEYS = ('time_grid_cuts_days', 'eval_horizons_days',
               'modelled_causes', 'folds')
# CR D-calibration settings under the header's ``cr_dcal``: D_k depends on
# alpha and the chi-squared on its bin count, so neither compares across them.
CR_DCAL_KEYS = ('alpha', 'chi2_bins')


def horizon_label(tau: float) -> str:
    """A horizon as the evaluation keys spell it: ``30d``, ``1y``."""
    return format_days(float(tau)).replace(' ', '')


def parse_arm(arm: str, models: Dict[str, str],
              variants: Dict[str, str]) -> Tuple[str, str]:
    """``'TransEHR2 raw'`` -> ``('TransEHR2', 'raw')``, checked."""
    model, _, variant = arm.rpartition(' ')
    if model not in models or variant not in variants:
        raise SystemExit(
            f'comparison arm {arm!r} is not "<model> <variant>" with a model '
            f'in {sorted(models)} and a variant in {sorted(variants)}')
    return model, variant


def load_evaluations(cfg: Dict,
                     model_dir: str) -> Dict[Tuple[str, str], Dict]:
    """Every (model, variant) evaluation the tables draw on."""
    evaluations = {}
    for model, experiment in cfg['MODELS'].items():
        for variant, split in cfg['VARIANTS'].items():
            path = os.path.join(model_dir, experiment,
                                cfg['EVALUATION_FILE'].format(split=split))
            if not os.path.isfile(path):
                raise SystemExit(f'{model} {variant}: {path} does not exist')
            with open(path, 'r') as f_in:
                evaluations[(model, variant)] = yaml.safe_load(f_in)
    return evaluations


def check_headers(evaluations: Dict[Tuple[str, str], Dict]) -> Dict:
    """The shared header, or SystemExit naming every disagreement."""
    problems = []
    reference = None
    for (model, variant), evaluation in evaluations.items():
        dcal = evaluation.get('cr_dcal') or {}
        missing = ([key for key in HEADER_KEYS if key not in evaluation]
                   + [f'cr_dcal.{key}' for key in CR_DCAL_KEYS
                      if key not in dcal])
        if missing:
            problems.append(f'{model} {variant}: header lacks {missing}')
            continue
        header = {key: evaluation[key] for key in HEADER_KEYS}
        header.update({f'cr_dcal.{key}': dcal[key] for key in CR_DCAL_KEYS})
        if reference is None:
            reference = (model, variant, header)
            continue
        ref_model, ref_variant, ref_header = reference
        for key in header:
            if header[key] != ref_header[key]:
                problems.append(
                    f'{model} {variant} and {ref_model} {ref_variant} '
                    f'disagree on {key}: {header[key]!r} vs '
                    f'{ref_header[key]!r}')
    if problems:
        raise SystemExit('Refusing to compare:\n  ' + '\n  '.join(problems))
    return reference[2]


def fold_values(evaluation: Dict, key: str, label: str) -> np.ndarray:
    if key not in evaluation:
        raise SystemExit(f'{label}: no metric {key!r} in its evaluation')
    return np.asarray(evaluation[key], dtype=np.float64)


def comparison_row(metric: str, arm_a: str, arm_b: str, a: np.ndarray,
                   b: np.ndarray, ratio: float, tested: bool) -> Dict:
    row = {'metric': metric, 'comparison': f'{arm_a} vs {arm_b}',
           'mean_a': np.nan, 'se_a': corrected_standard_error(a, ratio),
           'mean_b': np.nan, 'se_b': corrected_standard_error(b, ratio),
           'mean_difference': np.nan, 't': np.nan, 'df': np.nan,
           'p': np.nan, 'p_bh': np.nan, 'note': ''}
    for side, values in (('a', a), ('b', b)):
        if np.any(~np.isnan(values)):
            row[f'mean_{side}'] = float(np.nanmean(values))
    try:
        test = corrected_resampled_ttest(a, b, ratio)
    except ValueError as err:
        row['note'] = str(err)
        return row
    row['mean_difference'] = test.mean_difference
    if tested:
        row.update(t=test.statistic, df=test.df, p=test.p_value,
                   note=test.note or '')
    else:
        row['note'] = 'reported, not tested'
    return row


def comparison_table(cfg: Dict, evaluations: Dict, models: Dict,
                     cause: str, horizons: List[float]) -> pd.DataFrame:
    """One cause's table, BH-adjusted as one family."""
    key = metric_label(cause)
    taus = [horizon_label(t) for t in horizons]

    def expand(templates):
        out = []
        for template in templates:
            if '{tau}' in template:
                out += [template.format(cause=key, tau=tau) for tau in taus]
            else:
                out.append(template.format(cause=key))
        return out

    ratio = float(cfg['N_TRAIN_TEST_RATIO'])
    rows = []
    for metrics, tested in ((expand(cfg['TESTED_METRICS']), True),
                            (expand(cfg.get('REPORTED_METRICS', [])), False)):
        for metric in metrics:
            for arm_a, arm_b in cfg['COMPARISONS']:
                a = parse_arm(arm_a, models, cfg['VARIANTS'])
                b = parse_arm(arm_b, models, cfg['VARIANTS'])
                rows.append(comparison_row(
                    metric, arm_a, arm_b,
                    fold_values(evaluations[a], metric, arm_a),
                    fold_values(evaluations[b], metric, arm_b),
                    ratio, tested))
    table = pd.DataFrame(rows)
    # Untested rows carry NaN p, which benjamini_hochberg leaves out of m.
    table['p_bh'] = benjamini_hochberg(table['p'].to_numpy())
    return table


def dcal_chi2_table(evaluations: Dict, causes: List[str], folds: List[str],
                    level: float) -> pd.DataFrame:
    """The per-fold chi-squared and P of every model, and the folds passing.

    A fold whose P is NaN is untested: it is counted in ``folds_untested``,
    not as passing or failing, and the note names it. Overall, a fold fails
    when any cause fails, and is otherwise untested when any cause is.
    """
    threshold = level * len(causes)
    rows = []
    for (model, variant), evaluation in evaluations.items():
        label = f'{model} {variant}'
        any_fails = np.zeros(len(folds), dtype=bool)
        any_untested = np.zeros(len(folds), dtype=bool)
        for cause in causes:
            key = metric_label(cause)
            statistic = fold_values(evaluation, f'{key}_CR_DCal_Chi2', label)
            p = fold_values(evaluation, f'{key}_CR_DCal_Chi2_P', label)
            untested = np.isnan(p)
            passing = ~untested & (np.nan_to_num(p) > threshold)
            any_fails |= ~untested & ~passing
            any_untested |= untested
            row = {'model': model, 'variant': variant, 'cause': cause}
            row.update({f'chi2_{f}': v for f, v in zip(folds, statistic)})
            row.update({f'p_{f}': v for f, v in zip(folds, p)})
            row['folds_passing'] = int(passing.sum())
            row['folds_untested'] = int(untested.sum())
            row['note'] = ''
            if untested.any():
                names = ', '.join(f for f, u in zip(folds, untested) if u)
                row['note'] = (
                    f'P is NaN in {names}: the chi-squared is undefined when '
                    f'its expected bucket counts are zero, i.e. the model '
                    f'predicts no {cause} incidence by the last cut')
            rows.append(row)
        overall_untested = any_untested & ~any_fails
        rows.append({'model': model, 'variant': variant, 'cause': 'all',
                     'folds_passing': int((~any_fails
                                           & ~any_untested).sum()),
                     'folds_untested': int(overall_untested.sum()),
                     'note': ''})
    table = pd.DataFrame(rows)
    table['n_folds'] = len(folds)
    table['pass_threshold_p'] = threshold
    return table


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0])
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Directory the CSVs are written into')
    parser.add_argument('--model_dir', type=str, default='./models',
                        help='Root directory holding the experiments')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG,
                        help='Comparison config [default: %(default)s]')
    args = parser.parse_args()

    with open(args.config, 'r') as f_in:
        cfg = yaml.safe_load(f_in)
    if cfg['BH_FAMILY'] != 'cause':
        raise SystemExit(f'BH_FAMILY {cfg["BH_FAMILY"]!r}: only cause is '
                         f'implemented')
    os.makedirs(args.out_dir, exist_ok=True)

    evaluations = load_evaluations(cfg, args.model_dir)
    header = check_headers(evaluations)
    causes = list(header['modelled_causes'])

    for cause in causes:
        table = comparison_table(cfg, evaluations, cfg['MODELS'], cause,
                                 header['eval_horizons_days'])
        path = os.path.join(args.out_dir, f'{cause}_comparisons.csv')
        table.to_csv(path, index=False)
        tested = table['p'].notna()
        print(f'{cause}: {int(tested.sum())} tests, '
              f'{int((table["p_bh"] < 0.05).sum())} at p_BH < 0.05 '
              f'-> {path}')

    dcal = dcal_chi2_table(evaluations, causes, list(header['folds']),
                           float(cfg['DCAL_CHI2_LEVEL']))
    path = os.path.join(args.out_dir, 'dcal_chi2.csv')
    dcal.to_csv(path, index=False)
    print(f'D-cal chi-squared -> {path}')


if __name__ == '__main__':
    main()
