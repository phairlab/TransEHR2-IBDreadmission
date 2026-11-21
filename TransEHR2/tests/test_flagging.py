"""The LACE comparison's flagging metrics (plan 2026-11-18, Task C), on
hand-built and synthetic cohorts."""

import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from TransEHR2 import flagging
from TransEHR2.survival import TimeGrid

from .cr_cohort import simulate


CAUSES = ["readmission", "death", "out_migration"]
CUTS = [8.0, 31.0, 91.0, 181.0, 366.0]
READMIT, DEATH, OUT = 0, 1, 2


def uncensored(n=400, seed=0):
    rng = np.random.default_rng(seed)
    days = rng.uniform(0.5, 120.0, n)
    cause = rng.integers(0, 3, n)
    return days, cause, np.ones(n, dtype=bool)


# ------------------------------------------------------------- the outcome

def test_the_four_groups_and_their_weights():
    # Censored at 2 (1 of 6 at risk) and at 50 (past H).
    days = np.array([2.0, 5.0, 6.0, 8.0, 20.0, 50.0])
    cause = np.array([-1, READMIT, OUT, DEATH, READMIT, -1])
    event = np.array([False, True, True, True, True, False])
    positive, weight = flagging.composite_outcome(days, cause, event,
                                                  [READMIT, DEATH], 10.0)
    assert positive.tolist() == [False, True, False, True, False, False]
    g = 5 / 6
    # Censored before H: 0. Out-migration by H is a weighted negative,
    # not dropped. Past H: event-free at H, weighted by G(H).
    np.testing.assert_allclose(weight, [0.0, 1 / g, 1 / g, 1 / g, 1 / g,
                                        1 / g])


def test_without_censoring_the_rates_are_the_plain_ones():
    days, cause, event = uncensored()
    positive, weight = flagging.composite_outcome(days, cause, event,
                                                  [READMIT, DEATH], 30.0)
    assert np.all(weight == 1.0)
    flag = np.random.default_rng(1).random(days.size) < 0.3
    got = flagging.rates(flag, positive, weight)
    assert got['Sensitivity'] == pytest.approx(flag[positive].mean())
    assert got['Specificity'] == pytest.approx((~flag[~positive]).mean())
    assert got['PPV'] == pytest.approx(positive[flag].mean())
    assert got['Flag_Rate'] == pytest.approx(flag.mean())


# -------------------------------------------------------------- ROC and NB

def test_the_auc_without_censoring_is_mann_whitney_ties_halved():
    days, cause, event = uncensored(seed=2)
    positive, weight = flagging.composite_outcome(days, cause, event,
                                                  [READMIT, DEATH], 60.0)
    score = np.random.default_rng(3).integers(0, 10, days.size).astype(float)
    pos, neg = score[positive], score[~positive]
    expected = (np.mean(pos[:, None] > neg[None, :])
                + 0.5 * np.mean(pos[:, None] == neg[None, :]))
    curve = flagging.roc_curve(score, positive, weight)
    assert curve['auc'] == pytest.approx(expected, rel=1e-12)
    assert curve['fpr'][0] == curve['tpr'][0] == 0.0
    assert curve['fpr'][-1] == curve['tpr'][-1] == 1.0


def test_treat_all_and_treat_none():
    days, cause, event = uncensored(seed=4)
    positive, weight = flagging.composite_outcome(days, cause, event,
                                                  [READMIT, DEATH], 30.0)
    p_t = np.array([0.0, 0.05, 0.2])
    prevalence = positive.mean()
    np.testing.assert_allclose(
        flagging.net_benefit(np.ones(days.size, bool), positive, weight,
                             p_t),
        prevalence - (1 - prevalence) * p_t / (1 - p_t), rtol=1e-12)
    np.testing.assert_array_equal(
        flagging.net_benefit(np.zeros(days.size, bool), positive, weight,
                             p_t), 0.0)


def test_sensitivity_at_matches_rates():
    days = np.array([1.0, 3.0, 4.0, 9.0, 12.0, 40.0, 41.0])
    cause = np.array([READMIT, -1, DEATH, READMIT, OUT, -1, READMIT])
    event = cause >= 0
    positive, weight = flagging.composite_outcome(days, cause, event,
                                                  [READMIT, DEATH], 30.0)
    score = np.array([0.9, 0.1, 0.4, 0.4, 0.2, 0.3, 0.8])
    thresholds = np.array([0.0, 0.3, 0.4, 0.5, 0.95])
    got = flagging.sensitivity_at(score, positive, weight, thresholds)
    for c, value in zip(thresholds, got):
        assert value == pytest.approx(
            flagging.rates(score >= c, positive, weight)['Sensitivity'])


# --------------------------------------------------------------- the rules

def test_rule_a_flags_the_same_fraction():
    score = np.random.default_rng(5).random(1000)
    for fraction in (0.07, 0.25, 0.5):
        threshold = flagging.matched_rate_threshold(score, fraction)
        assert np.mean(score >= threshold) == pytest.approx(fraction)
    assert flagging.matched_rate_threshold(score, 0.0) is None


def test_rule_b_is_the_largest_threshold_meeting_the_sensitivity():
    days, cause, event = uncensored(seed=6)
    positive, weight = flagging.composite_outcome(days, cause, event,
                                                  [READMIT, DEATH], 30.0)
    score = np.random.default_rng(7).random(days.size) + positive * 0.3
    target = 0.6
    threshold = flagging.matched_sensitivity_threshold(score, positive,
                                                       weight, target)
    assert flagging.rates(score >= threshold, positive,
                          weight)['Sensitivity'] >= target
    higher = np.unique(score)[np.unique(score) > threshold].min()
    assert flagging.rates(score >= higher, positive,
                          weight)['Sensitivity'] < target


# ----------------------------------------------------------- the script

def write_dump(path, data, idx, grid):
    from dump_transehr2_deephit_predictions import save_predictions_csv
    path.parent.mkdir(parents=True, exist_ok=True)
    save_predictions_csv(idx, data['time_to_event'], data['event_type'],
                         data['cif'], data['pmf'], data['p_survive'], grid,
                         str(path))


@pytest.fixture
def experiment(tmp_path):
    grid = TimeGrid(CUTS, CAUSES)
    rng = np.random.default_rng(8)
    splits, start = {}, 0
    for split, seed in (('val', 10), ('test', 11)):
        data = simulate(grid, n=500, seed=seed)
        splits[split] = (data, np.arange(start, start + 500))
        start += 500
    labels = pd.DataFrame({
        'TIME_TO_EVENT': np.concatenate([d['time_to_event']
                                         for d, _ in splits.values()]),
        'EVENT_TYPE': np.concatenate([d['event_type']
                                      for d, _ in splits.values()])})
    labels.to_csv(tmp_path / 'labels.csv', index=False)
    lace = pd.DataFrame({'idx': np.arange(start), 'PATID': 'x',
                         'STAY_INDEX': 0,
                         'INDEX_TYPE': rng.choice(['DAD', 'AMB'], start),
                         'LACE': rng.integers(1, 20, start)})
    lace.to_csv(tmp_path / 'lace_ama.csv', index=False)
    for split, (data, idx) in splits.items():
        write_dump(tmp_path / 'models' / 'exp' / 'fold1' / 'competing_risks'
                   / f'competing_risks_{split}_finetuned_output.csv', data,
                   idx, grid)
    config = tmp_path / 'exp.yaml'
    config.write_text(yaml.dump({'TIME_GRID_CUTS_DAYS': CUTS,
                                 'MODELLED_CAUSES': CAUSES}))
    return tmp_path, config, splits, lace


def run(tmp_path, config, monkeypatch):
    import compare_lace_flags
    out = tmp_path / 'out'
    assert compare_lace_flags.main([
        'exp', '--model_dir', str(tmp_path / 'models'),
        '--experiment_config', str(config),
        '--lace', str(tmp_path / 'lace_ama.csv'),
        '--labels', str(tmp_path / 'labels.csv'),
        '--out_dir', str(out)]) == 0
    return out


def test_the_script_writes_both_populations(experiment, monkeypatch):
    tmp_path, config, splits, lace = experiment
    out = run(tmp_path, config, monkeypatch)
    for pop in ('all', 'dad'):
        with open(out / f'exp_lace_comparison_{pop}.yaml') as f_in:
            found = yaml.safe_load(f_in)
        assert found['folds'] == ['fold1']
        for h in ('7d', '30d', '90d'):
            for name in ('LACE_ge7', 'LACE_ge14', 'Model_rate_LACE7',
                         'Model_sens_LACE14'):
                for metric in ('Sensitivity', 'Specificity', 'PPV',
                               'Flag_Rate'):
                    assert len(found[f'{h}_{name}_{metric}']) == 1
            assert 0.0 <= found[f'{h}_Model_AUC'][0] <= 1.0
    roc = pd.read_csv(out / 'exp_lace_roc.csv')
    assert set(roc['population']) == {'all', 'dad'}
    decision = pd.read_csv(out / 'exp_lace_decision.csv')
    none = decision[decision['policy'] == 'Treat_none']
    assert (none['net_benefit'] == 0).all()


def test_rule_a_matches_lace_flag_rate_on_val(experiment, monkeypatch):
    from compare_lace_flags import composite_risk
    tmp_path, config, splits, lace = experiment
    out = run(tmp_path, config, monkeypatch)
    with open(out / 'exp_lace_comparison_all.yaml') as f_in:
        found = yaml.safe_load(f_in)
    grid = TimeGrid(CUTS, CAUSES)
    val, idx = splits['val']
    risk = composite_risk(val, grid, [0, 1], 30.0)
    threshold = found['thresholds']['fold1']['30d_Model_rate_LACE7']
    assert np.mean(risk >= threshold) == pytest.approx(
        np.mean(lace['LACE'].to_numpy()[idx] >= 7), abs=1e-9)


def test_dumps_keyed_by_another_numbering_are_refused(experiment,
                                                      monkeypatch):
    tmp_path, config, _, _ = experiment
    labels = pd.read_csv(tmp_path / 'labels.csv')
    labels.iloc[::-1].reset_index(drop=True).to_csv(
        tmp_path / 'labels.csv', index=False)
    with pytest.raises(ValueError, match='re-dump'):
        run(tmp_path, config, monkeypatch)
