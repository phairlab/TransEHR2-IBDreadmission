"""The full-cohort configs are their CV configs with only the planned changes.

Each part1_full_cohort_<stem>.yaml is part1_<stem>.yaml retrained on the
whole 2-point cohort and scored on the held-out set (user, 2026-11-15). A
setting changed in one and not the other would make the held-out numbers
describe a different model from the cross-validated one.
"""

import os

import pytest
import yaml

CONFIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), 'configs')
EXPERIMENTS = os.path.join(CONFIGS, 'experiments')

# stem -> the keys allowed to differ, and what the full-cohort config holds.
CHANGES = {
    'transehr2': {},
    'tabdh': {},
    'csc': {'FOLDS': ['full']},
    'transehr2_norank': {'PRETRAINED_FROM': 'part1_full_cohort_transehr2'},
    'transehr2_csc': {'PRETRAINED_FROM': 'part1_full_cohort_transehr2'},
    'tabdh_norank': {},
}


def load(path):
    with open(path) as f_in:
        return yaml.safe_load(f_in)


@pytest.mark.parametrize('stem', sorted(CHANGES))
def test_only_the_planned_keys_differ(stem):
    cv = load(os.path.join(EXPERIMENTS, f'part1_{stem}.yaml'))
    full = load(os.path.join(EXPERIMENTS, f'part1_full_cohort_{stem}.yaml'))
    expected = {'EXPERIMENT_NAME': f'part1_full_cohort_{stem}',
                **CHANGES[stem]}
    differ = {k for k in set(cv) | set(full) if cv.get(k) != full.get(k)}
    assert differ == set(expected)
    assert {k: full[k] for k in expected} == expected


@pytest.mark.parametrize('stem', ['transehr2_norank', 'transehr2_csc'])
def test_the_ablations_can_adopt_the_full_cohort_encoder(stem):
    from fit_transehr2_deephit import check_pretraining_source
    path = os.path.join(EXPERIMENTS, f'part1_full_cohort_{stem}.yaml')
    check_pretraining_source(path, load(path), 'part1_full_cohort_transehr2')


def test_the_report_config_differs_only_in_its_models():
    reports = os.path.join(CONFIGS, 'reports')
    cv = load(os.path.join(reports, 'part1_comparisons.yaml'))
    full = load(os.path.join(reports, 'part1_full_cohort_comparisons.yaml'))
    assert {k: v for k, v in cv.items() if k != 'MODELS'} == {
        k: v for k, v in full.items() if k != 'MODELS'}
    assert full['MODELS'] == {
        model: experiment.replace('part1_', 'part1_full_cohort_', 1)
        for model, experiment in cv['MODELS'].items()}
