"""The one time convention every horizon metric follows.

At a horizon of d days, an event is "by d" when its time is at most d, and a
model's incidence is read at d on its continuous curve (``interpolate_cif``
for a grid model), not at the nearest cut. The examples put events at
exactly 7.0 days and at 7.5 days. With d = 7, the first is a case and the
second is not. Between them, the grid's cut at 8 days is never what is read.
"""

import numpy as np
import pytest

from TransEHR2.survival import (
    TimeGrid, aalen_johansen, cause_specific_brier, integrated_brier_to,
    interpolate_cif, lee_weighted_concordance
)

from .cr_cohort import simulate


CAUSES = ["readmission", "death", "out_migration"]
# The first cut is past the horizon tested, as in the Part 1 grid.
CUTS = [8.0, 31.0, 91.0, 181.0]

# Three patients, outcomes in days: cause 0 at exactly 7.0, cause 0 at 7.5,
# and censored at 30. Nobody is censored before 30, so G = 1 up to it.
DAYS = np.array([7.0, 7.5, 30.0])
CAUSE = np.array([0, 0, -1])
EVENT = np.array([True, True, False])


@pytest.mark.parametrize('tau, expected', [
    (7.0, 1.0),        # case: 7.0; controls 7.5 (0.1) and 30 (0.5)
    (7.5, 2.0 / 3.0),  # cases: 7.0 as above, and 7.5, below its control
])
def test_lee_takes_an_event_at_the_horizon_and_not_after_it(tau, expected):
    risk = np.array([0.9, 0.1, 0.5])
    got = lee_weighted_concordance(risk, DAYS, CAUSE, EVENT, 0, tau,
                                   DAYS, EVENT)
    assert got == pytest.approx(expected, rel=1e-12)


def test_the_brier_target_at_7_days_counts_7_0_and_not_7_5():
    p = np.array([[0.6], [0.2], [0.1]])
    at_7 = cause_specific_brier(p, DAYS, CAUSE, EVENT, 0, np.array([7.0]))
    assert at_7[0] == pytest.approx(((0.6 - 1) ** 2 + 0.2 ** 2 + 0.1 ** 2)
                                    / 3, rel=1e-12)
    at_7_5 = cause_specific_brier(p, DAYS, CAUSE, EVENT, 0, np.array([7.5]))
    assert at_7_5[0] == pytest.approx(((0.6 - 1) ** 2 + (0.2 - 1) ** 2
                                       + 0.1 ** 2) / 3, rel=1e-12)


def test_the_aalen_johansen_reference_includes_an_event_at_the_horizon():
    got = aalen_johansen(DAYS, CAUSE, EVENT, 0, np.array([6.99, 7.0, 7.4]))
    np.testing.assert_allclose(got, [0.0, 1.0 / 3.0, 1.0 / 3.0], rtol=1e-12)


def test_the_scorer_reads_the_curve_at_the_horizon_not_at_a_cut():
    from evaluate_finetuned_predictions import outcomes, score

    grid = TimeGrid(CUTS, CAUSES, brier_integration_days=30.0)
    data = simulate(grid, n=800, seed=21)
    _, ci, ev, days = outcomes(data, grid)
    dcal = {'alpha': 2.0, 'n_rho': 200, 'chi2_bins': 10, 'test_level': 0.05}
    out = score(data, grid, dcal, [7.0, 30.0])

    for k, key in enumerate(('Readmission', 'Death', 'Out_Migration')):
        at_7 = interpolate_cif(data['cif'][:, k, :], grid,
                               np.array([7.0]))[:, 0]
        # Seven days is inside the first bin, so the curve there is not the
        # value at the cut.
        assert np.all(at_7 < data['cif'][:, k, 0])
        assert out[f'{key}_Cindex_Lee_at_7d'] == lee_weighted_concordance(
            at_7, days, ci, ev, k, 7.0, days, ev)
        assert out[f'{key}_IBS_to_7d'] == integrated_brier_to(
            data['cif'][:, k, :], days, ci, ev, k, grid, 7.0)


def test_every_evaluation_records_the_convention(tmp_path, monkeypatch):
    import sys
    import yaml
    import evaluate_finetuned_predictions
    from evaluate_finetuned_predictions import HORIZON_CONVENTION

    from .test_evaluate_finetuned_predictions import write_split

    grid = TimeGrid(CUTS, CAUSES, brier_integration_days=30.0)
    for s, split in enumerate(('train', 'test')):
        write_split(str(tmp_path / 'exp' / 'fold1' / 'competing_risks'
                        / f'competing_risks_{split}_finetuned_output.csv'),
                    simulate(grid, n=400, seed=30 + s), grid)
    config = tmp_path / 'exp.yaml'
    config.write_text(yaml.dump({
        'TIME_GRID_CUTS_DAYS': CUTS, 'MODELLED_CAUSES': CAUSES,
        'BRIER_INTEGRATION_DAYS': 30, 'EVAL_HORIZONS_DAYS': [7, 30]}))
    monkeypatch.setattr(sys, 'argv', [
        'evaluate_finetuned_predictions.py', 'exp', '--model_dir',
        str(tmp_path), '--experiment_config', str(config)])
    evaluate_finetuned_predictions.main()
    for split in ('train', 'test'):
        with open(tmp_path / 'exp' / f'{split}_evaluation.yaml') as f_in:
            header = yaml.safe_load(f_in)
        assert header['horizon_convention'] == HORIZON_CONVENTION
        assert header['eval_horizons_days'] == [7.0, 30.0]


def test_the_pinball_keys_score_the_quantile_time_to_each_horizon():
    from evaluate_finetuned_predictions import outcomes, score

    from TransEHR2.survival import incidence_quantile_time, pinball_loss

    grid = TimeGrid(CUTS, CAUSES, brier_integration_days=30.0)
    data = simulate(grid, n=600, seed=23)
    _, ci, ev, days = outcomes(data, grid)
    dcal = {'alpha': 2.0, 'n_rho': 200, 'chi2_bins': 10, 'test_level': 0.05}
    out = score(data, grid, dcal, [7.0, 30.0], pinball_q=0.25)
    # The evaluation horizons, then the grid's.
    for h, label in ((7.0, '7d'), (30.0, '30d'), (181.0, '181d')):
        for k, key in enumerate(('Readmission', 'Death', 'Out_Migration')):
            estimated = incidence_quantile_time(data['cif'][:, k, :], grid,
                                                0.25, h)
            assert out[f'{key}_Pinball_q25_to_{label}'] == pinball_loss(
                estimated, days, ci, ev, k, h, 0.25)
            assert out[f'{key}_Pinball_N_to_{label}'] == int(np.sum(
                ev & (ci == k) & (days <= h) & np.isfinite(estimated)))


def test_g_comes_from_the_scored_split_not_the_train_dump(tmp_path,
                                                          monkeypatch):
    import sys
    import yaml
    import evaluate_finetuned_predictions

    from .test_evaluate_finetuned_predictions import write_split

    grid = TimeGrid(CUTS, CAUSES, brier_integration_days=30.0)
    config = tmp_path / 'exp.yaml'
    config.write_text(yaml.dump({
        'TIME_GRID_CUTS_DAYS': CUTS, 'MODELLED_CAUSES': CAUSES,
        'BRIER_INTEGRATION_DAYS': 30, 'EVAL_HORIZONS_DAYS': [7, 30],
        'PINBALL_Q': 0.3}))
    test = simulate(grid, n=400, seed=31)
    found = []
    # Two train dumps with different censoring: the test scores must agree.
    for train_seed in (32, 33):
        root = tmp_path / str(train_seed)
        fold = root / 'exp' / 'fold1' / 'competing_risks'
        write_split(str(fold / 'competing_risks_train_finetuned_output.csv'),
                    simulate(grid, n=400, seed=train_seed), grid)
        write_split(str(fold / 'competing_risks_test_finetuned_output.csv'),
                    test, grid)
        monkeypatch.setattr(sys, 'argv', [
            'evaluate_finetuned_predictions.py', 'exp', '--model_dir',
            str(root), '--experiment_config', str(config)])
        evaluate_finetuned_predictions.main()
        with open(root / 'exp' / 'test_evaluation.yaml') as f_in:
            found.append(yaml.safe_load(f_in))
    assert found[0] == found[1]
    assert found[0]['pinball_q'] == 0.3
    assert found[0]['censoring_weights'] == (
        evaluate_finetuned_predictions.CENSORING_SOURCE)
    assert 'Readmission_Pinball_q30_to_7d' in found[0]
    assert 'Readmission_Ratio_After_to_7d' in found[0]


def test_the_ratio_keys_read_the_time_error_estimate_to_each_horizon():
    """One estimate per patient, the one the time errors read, grouped by
    each horizon (user, 2026-11-24)."""
    from evaluate_finetuned_predictions import outcomes, score

    from TransEHR2.survival import half_incidence_time, time_ratios

    grid = TimeGrid(CUTS, CAUSES, brier_integration_days=30.0)
    data = simulate(grid, n=600, seed=29)
    _, ci, ev, days = outcomes(data, grid)
    dcal = {'alpha': 2.0, 'n_rho': 200, 'chi2_bins': 10, 'test_level': 0.05}
    out = score(data, grid, dcal, [7.0, 30.0], pinball_q=0.25)
    for h, label in ((7.0, '7d'), (30.0, '30d'), (181.0, '181d')):
        for k, key in enumerate(('Readmission', 'Death', 'Out_Migration')):
            estimated = half_incidence_time(data['cif'][:, k, :], grid)
            ratios = time_ratios(estimated, days, ci, ev, k, h)
            for side in ('Before', 'After'):
                np.testing.assert_equal(out[f'{key}_Ratio_{side}_to_{label}'],
                                        ratios[f'{side}_Ratio'])
                assert (out[f'{key}_Ratio_{side}_N_to_{label}']
                        == ratios[f'{side}_N'])
    # Not the pinball loss's q: no key spells it.
    assert not any(k.startswith('Readmission_Ratio_') and '_q' in k
                   for k in out)
