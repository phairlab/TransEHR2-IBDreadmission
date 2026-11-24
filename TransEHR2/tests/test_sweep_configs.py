"""The ranking-term sweep's configs are their bases with only the planned
changes (plan 2026-11-18, Task A; user, 2026-11-20)."""

import os

import pytest
import yaml

CONFIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), 'configs')
EXPERIMENTS = os.path.join(CONFIGS, 'experiments')
REPORTS = os.path.join(CONFIGS, 'reports')
SIGMAS = {'1e-3': 0.001, '3e-3': 0.003, '1e-2': 0.01, '3e-2': 0.03,
          '1e-1': 0.1}
SWEPT = {'TransEHR2': 'transehr2', 'TabDeepHit': 'tabdh'}
ANTOLINI = '{cause}_Cindex_Antolini_to_30d'


def load(path):
    with open(path) as f_in:
        return yaml.safe_load(f_in)


@pytest.mark.parametrize('base', sorted(SWEPT.values()))
@pytest.mark.parametrize('tag', sorted(SIGMAS))
def test_only_the_planned_keys_differ(base, tag):
    original = load(os.path.join(EXPERIMENTS, f'part1_{base}.yaml'))
    step = load(os.path.join(EXPERIMENTS, f'part1_{base}_sigma{tag}.yaml'))
    expected = {'EXPERIMENT_NAME': f'part1_{base}_sigma{tag}',
                'DEEPHIT_SIGMA': SIGMAS[tag],
                'DEEPHIT_RANK_SCALE_BY_SIGMA': True,
                'DEEPHIT_RANK_MAX_DAYS': 91}
    if base == 'transehr2':
        expected['PRETRAINED_FROM'] = 'part1_transehr2'
    if SIGMAS[tag] == original['DEEPHIT_SIGMA']:
        del expected['DEEPHIT_SIGMA']    # the base's own sigma
    differ = {k for k in set(original) | set(step)
              if original.get(k) != step.get(k)}
    assert differ == set(expected)
    assert {k: step[k] for k in expected} == expected


@pytest.mark.parametrize('tag', sorted(SIGMAS))
def test_the_transehr2_steps_can_adopt_its_encoder(tag):
    from fit_transehr2_deephit import check_pretraining_source
    path = os.path.join(EXPERIMENTS, f'part1_transehr2_sigma{tag}.yaml')
    check_pretraining_source(path, load(path), 'part1_transehr2')


@pytest.mark.parametrize('tag', sorted(SIGMAS))
def test_a_step_compares_its_deephits_with_the_other_four(tag):
    cv = load(os.path.join(REPORTS, 'part1_comparisons.yaml'))
    step = load(os.path.join(REPORTS, f'part1_sweep_sigma{tag}_comparisons'
                                      f'.yaml'))
    # The sweep's main measure is tested there and only reported in the CV
    # comparison, for the anchor (user, 2026-11-24).
    assert step['TESTED_METRICS'] == [ANTOLINI] + cv['TESTED_METRICS']
    # The time ratios are the CV and held-out figures' only (user,
    # 2026-11-24): the steps' evaluations need no re-scoring for them.
    assert [m for m in cv['REPORTED_METRICS']
            if m != ANTOLINI and '_Ratio_' not in m] == \
        step['REPORTED_METRICS']
    assert ANTOLINI in cv['REPORTED_METRICS']
    rest = {'MODELS', 'TESTED_METRICS', 'REPORTED_METRICS'}
    assert {k: v for k, v in cv.items() if k not in rest} == {
        k: v for k, v in step.items() if k not in rest}
    expected = dict(cv['MODELS'])
    expected.update({model: f'part1_{base}_sigma{tag}'
                     for model, base in SWEPT.items()})
    assert step['MODELS'] == expected


def test_the_cut_the_steps_name_is_a_grid_cut():
    from TransEHR2.losses import rank_max_bin_for
    from TransEHR2.survival import TimeGrid
    config = load(os.path.join(EXPERIMENTS, 'part1_transehr2_sigma1e-3.yaml'))
    grid = TimeGrid(config['TIME_GRID_CUTS_DAYS'], config['MODELLED_CAUSES'],
                    config['BRIER_INTEGRATION_DAYS'])
    assert rank_max_bin_for(grid, config['DEEPHIT_RANK_MAX_DAYS']) == 2
