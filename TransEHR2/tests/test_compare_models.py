"""compare_models.py end to end, on evaluation YAMLs the tests fabricate.

The expected P values are recomputed here from the per-fold lists by the
formula, not through ``TransEHR2.stats``, and the expected BH values by the
step-up definition taken literally: p_(i) * m / i, then the minimum over every
rank at or above i.
"""

import math
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from scipy import stats

from TransEHR2.survival import metric_label


CAUSES = ['readmission', 'death', 'out_migration']
HORIZONS = [30, 90, 365, 1095, 1825]
TAUS = ['30d', '90d', '1y', '3y', '5y']
FOLDS = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']
CUTS = [46.45, 178.67, 435.09, 898.92, 1736.54, 6454.83]
SPLITS = {'raw': 'test', 'recal': 'test_recal_aj'}
EXPERIMENTS = {'TransEHR2': 'part1_transehr2', 'CSC': 'part1_csc',
               'TabDeepHit': 'part1_tabdh',
               'TransEHR2_noRank': 'part1_transehr2_norank',
               'TabDeepHit_noRank': 'part1_tabdh_norank',
               'TransEHR2_CSC': 'part1_transehr2_csc'}
RATIO = 0.25


def metric_keys(cause):
    key = metric_label(cause)
    tested = ([f'{key}_Cindex_Lee_at_{t}' for t in TAUS]
              + [f'{key}_IBS_to_{t}' for t in TAUS] + [f'{key}_CR_DCal'])
    reported = [f'{key}_IPA_to_{t}' for t in TAUS] + [
        f'{key}_{side}estimation_{stat}' for side in ('Over', 'Under')
        for stat in ('MAE', 'N')]
    return tested, reported


def header(split, override):
    out = {'experiment': 'x', 'split': split, 'task': 'competing_risks',
           'folds': list(FOLDS), 'time_grid_cuts_days': list(CUTS),
           'eval_horizons_days': [float(h) for h in HORIZONS],
           'modelled_causes': list(CAUSES),
           'cr_dcal': {'alpha': 2.0, 'n_rho': 64, 'chi2_bins': 10,
                       'test_level': 0.05}}
    out.update(override)
    return out


def fabricate(model_dir, seed=0, override=None, chi2_p=None):
    """Every evaluation the shipped config names, with random per-fold values.

    ``override`` maps (model, variant) to header fields to change;
    ``chi2_p`` maps a cause to the per-fold D-cal P every evaluation carries.
    """
    rng = np.random.default_rng(seed)
    override = override or {}
    chi2_p = chi2_p or {}
    for model, experiment in EXPERIMENTS.items():
        for variant, split in SPLITS.items():
            payload = header(split, override.get((model, variant), {}))
            for cause in CAUSES:
                tested, reported = metric_keys(cause)
                for key in tested + reported:
                    payload[key] = [float(v) for v in
                                    rng.normal(0.7, 0.02, len(FOLDS))]
                key = metric_label(cause)
                payload[f'{key}_CR_DCal_Chi2'] = [
                    float(v) for v in rng.uniform(5, 20, len(FOLDS))]
                payload[f'{key}_CR_DCal_Chi2_P'] = chi2_p.get(
                    cause, [float(v) for v in rng.uniform(0, 1, 5)])
            path = model_dir / experiment / f'{split}_evaluation.yaml'
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f_out:
                yaml.dump(payload, f_out, sort_keys=False)


def run(tmp_path, monkeypatch):
    from compare_models import main
    out_dir = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', [
        'compare_models.py', '--model_dir', str(tmp_path / 'models'),
        '--out_dir', str(out_dir)])
    main()
    return out_dir


def read_evaluation(model_dir, model, variant):
    path = (model_dir / EXPERIMENTS[model]
            / f'{SPLITS[variant]}_evaluation.yaml')
    with open(path) as f_in:
        return yaml.safe_load(f_in)


def expected_p(a, b):
    d = np.asarray(a) - np.asarray(b)
    k = d.size
    t = d.mean() / math.sqrt((1 / k + RATIO) * d.var(ddof=1))
    return 2 * stats.t.sf(abs(t), k - 1)


def step_up(p):
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    out = [0.0] * m
    for rank, i in enumerate(order, start=1):
        out[i] = min(1.0, min(p[order[j - 1]] * m / j
                              for j in range(rank, m + 1)))
    return out


@pytest.fixture
def config():
    from compare_models import DEFAULT_CONFIG
    with open(DEFAULT_CONFIG) as f_in:
        return yaml.safe_load(f_in)


def test_the_shipped_config_names_the_planned_design(config):
    assert config['N_TRAIN_TEST_RATIO'] == RATIO
    assert config['BH_FAMILY'] == 'cause'
    assert config['VARIANTS'] == SPLITS
    assert config['MODELS'] == EXPERIMENTS
    pairs = [('TransEHR2', 'CSC'), ('TransEHR2', 'TabDeepHit'),
             ('TabDeepHit', 'CSC')]
    expected = [[f'{a} {v}', f'{b} {v}'] for v in ('raw', 'recal')
                for a, b in pairs]
    expected += [[f'{m} recal', f'{m} raw']
                 for m in ('TransEHR2', 'CSC', 'TabDeepHit')]
    # The ablations against their references (user, 2026-10-26).
    ablations = [('TransEHR2_noRank', 'TransEHR2'),
                 ('TabDeepHit_noRank', 'TabDeepHit'),
                 ('TransEHR2_CSC', 'TransEHR2'), ('TransEHR2_CSC', 'CSC')]
    expected += [[f'{a} {v}', f'{b} {v}'] for v in ('raw', 'recal')
                 for a, b in ablations]
    expected += [[f'{m} recal', f'{m} raw']
                 for m in ('TransEHR2_noRank', 'TabDeepHit_noRank',
                           'TransEHR2_CSC')]
    # Each Cox head against the no-rank DeepHit (user, 2026-10-27).
    expected += [[f'{a} {v}', f'{b} {v}'] for v in ('raw', 'recal')
                 for a, b in [('TransEHR2_CSC', 'TransEHR2_noRank'),
                              ('TabDeepHit_noRank', 'CSC')]]
    assert config['COMPARISONS'] == expected


def test_p_values_and_bh_match_the_definitions(tmp_path, monkeypatch,
                                               config):
    model_dir = tmp_path / 'models'
    fabricate(model_dir)
    out_dir = run(tmp_path, monkeypatch)

    for cause in CAUSES:
        table = pd.read_csv(out_dir / f'{cause}_comparisons.csv')
        tested, reported = metric_keys(cause)
        expected = []
        for metric in tested:
            for arm_a, arm_b in config['COMPARISONS']:
                a = read_evaluation(model_dir, *arm_a.split(' '))[metric]
                b = read_evaluation(model_dir, *arm_b.split(' '))[metric]
                expected.append(expected_p(a, b))
        rows = table[table['metric'].isin(tested)]
        # 11 tested metrics x 24 comparisons in one family.
        assert len(rows) == len(expected) == 264
        assert rows['p'].to_numpy() == pytest.approx(expected, rel=1e-9)
        assert rows['p_bh'].to_numpy() == pytest.approx(
            step_up(expected), rel=1e-9)
        assert (rows['df'] == 4).all()

        untested = table[table['metric'].isin(reported)]
        assert len(untested) == 9 * 24
        assert untested[['t', 'p', 'p_bh']].isna().all().all()


def test_the_means_and_ses_are_the_corrected_ones(tmp_path, monkeypatch):
    model_dir = tmp_path / 'models'
    fabricate(model_dir, seed=1)
    out_dir = run(tmp_path, monkeypatch)
    table = pd.read_csv(out_dir / 'death_comparisons.csv')
    row = table[(table['metric'] == 'Death_IBS_to_1y')
                & (table['comparison'] == 'TransEHR2 raw vs CSC raw')]
    a = np.array(read_evaluation(model_dir, 'TransEHR2', 'raw')
                 ['Death_IBS_to_1y'])
    b = np.array(read_evaluation(model_dir, 'CSC', 'raw')
                 ['Death_IBS_to_1y'])
    assert row['mean_a'].item() == pytest.approx(a.mean())
    assert row['se_a'].item() == pytest.approx(
        math.sqrt((1 / 5 + RATIO) * a.var(ddof=1)))
    assert row['se_b'].item() == pytest.approx(
        math.sqrt((1 / 5 + RATIO) * b.var(ddof=1)))
    assert row['mean_difference'].item() == pytest.approx((a - b).mean())


def test_the_families_are_per_table_not_pooled(tmp_path, monkeypatch):
    fabricate(tmp_path / 'models', seed=2)
    out_dir = run(tmp_path, monkeypatch)
    tables = [pd.read_csv(out_dir / f'{c}_comparisons.csv')
              for c in CAUSES]
    p = np.concatenate([t['p'].dropna().to_numpy() for t in tables])
    pooled = step_up(list(p))
    per_table = np.concatenate([t['p_bh'].dropna().to_numpy()
                                for t in tables])
    assert not np.allclose(pooled, per_table)


def test_the_dcal_pass_rule_is_level_times_k(tmp_path, monkeypatch):
    """K = 3: 0.10 fails and 0.16 passes; overall pass needs every cause."""
    fabricate(tmp_path / 'models', chi2_p={
        'readmission': [0.10, 0.16, 0.16, 0.9, 0.16],
        'death': [0.9, 0.9, 0.14, 0.9, 0.9],
        'out_migration': [0.9, 0.9, 0.9, 0.9, float('nan')]})
    out_dir = run(tmp_path, monkeypatch)
    table = pd.read_csv(out_dir / 'dcal_chi2.csv')
    rows = table[(table['model'] == 'CSC') & (table['variant'] == 'recal')]
    passing = dict(zip(rows['cause'], rows['folds_passing']))
    assert passing == {'readmission': 4, 'death': 4, 'out_migration': 4,
                       'all': 2}
    # fold5's NaN is neither a pass nor a fail, for the cause and overall.
    untested = dict(zip(rows['cause'], rows['folds_untested']))
    assert untested == {'readmission': 0, 'death': 0, 'out_migration': 1,
                        'all': 1}
    notes = dict(zip(rows['cause'], rows['note'].fillna('')))
    assert 'fold5' in notes['out_migration']
    assert notes['readmission'] == ''
    out_migration = rows[rows['cause'] == 'out_migration']
    assert math.isnan(out_migration['p_fold5'].item())
    assert np.allclose(table['pass_threshold_p'], 0.15)
    # The chi-squared stays out of the comparison tables.
    comparisons = pd.read_csv(out_dir / 'readmission_comparisons.csv')
    assert not comparisons['metric'].str.contains('Chi2').any()


@pytest.mark.parametrize('field,value', [
    ('time_grid_cuts_days', [30.0, 90.0, 365.0]),
    ('eval_horizons_days', [30, 365]),
    ('modelled_causes', ['readmission', 'death']),
    ('folds', ['fold1', 'fold2', 'fold3', 'fold4', 'fold6']),
    ('cr_dcal.alpha', 1.0),
    ('cr_dcal.chi2_bins', 20),
])
def test_disagreeing_headers_are_refused(tmp_path, monkeypatch, field,
                                         value):
    if field.startswith('cr_dcal.'):
        dcal = header('test', {})['cr_dcal']
        dcal[field.split('.')[1]] = value
        field_override = {'cr_dcal': dcal}
    else:
        field_override = {field: value}
    fabricate(tmp_path / 'models',
              override={('CSC', 'recal'): field_override})
    with pytest.raises(SystemExit, match=field.replace('.', r'\.')):
        run(tmp_path, monkeypatch)


def test_dcal_settings_the_guard_ignores_do_not_refuse(tmp_path,
                                                       monkeypatch):
    """Only alpha and the bin count change what D_k and the chi-squared
    mean."""
    dcal = header('test', {})['cr_dcal']
    dcal['test_level'] = 0.01
    fabricate(tmp_path / 'models',
              override={('CSC', 'recal'): {'cr_dcal': dcal}})
    run(tmp_path, monkeypatch)


def test_a_header_without_a_compared_field_is_refused(tmp_path, monkeypatch):
    model_dir = tmp_path / 'models'
    fabricate(model_dir)
    path = model_dir / 'part1_tabdh' / 'test_evaluation.yaml'
    with open(path) as f_in:
        payload = yaml.safe_load(f_in)
    del payload['eval_horizons_days']
    del payload['cr_dcal']['chi2_bins']
    with open(path, 'w') as f_out:
        yaml.dump(payload, f_out)
    with pytest.raises(SystemExit,
                       match=r"header lacks \['eval_horizons_days', "
                             r"'cr_dcal\.chi2_bins'\]"):
        run(tmp_path, monkeypatch)
