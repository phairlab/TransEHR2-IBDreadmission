"""The CSC reconstruction, its dense recalibration, and the scripts that use
them, on synthetic parameters."""

import sys

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from TransEHR2.csc import (
    CSCCurve, csc_dense_offsets, read_csc_baseline, read_csc_lp,
    recalibration_axis
)
from TransEHR2.survival import (
    MINUTES_PER_DAY, RECAL_FLOOR, TimeGrid, _repair_shifted_cif,
    aj_recalibration_offsets, cr_d_calibration, cr_d_calibration_from_values,
    cr_dcal_chi2, cr_dcal_chi2_from_values
)

from .cr_cohort import simulate


CAUSES = ["readmission", "death", "out_migration"]
# Off the whole days, as the real cuts are.
CUTS = [46.45, 178.67, 435.09]
REFERENCE = Path(__file__).parent / 'fixtures' / 'csc_reference'


def baseline_hazard(n_times, seed, horizon=435.09):
    """Training event times past the horizon too, each cause stepping at
    about half of them."""
    rng = np.random.default_rng(seed)
    times = np.sort(rng.choice(np.arange(1, 50 * n_times), n_times,
                               replace=False) * (1.2 * horizon)
                    / (50 * n_times))
    steps = (rng.uniform(size=(3, n_times)) < 0.5) * rng.exponential(
        0.6 / n_times, size=(3, n_times))
    return times, np.cumsum(steps, axis=1)


def linear_predictors(n, seed):
    return np.random.default_rng(seed).normal(0.0, 0.7, size=(n, 3))


def late_readmissions(n, seed):
    """Readmissions only after day 200, nothing else: AJ sits at zero early
    and climbs past every patient's incidence late."""
    rng = np.random.default_rng(seed + 1000)
    readmitted = rng.uniform(size=n) < 0.8
    days = np.where(readmitted, rng.uniform(200.0, 430.0, size=n), 435.09)
    return days * MINUTES_PER_DAY, np.where(readmitted, 1, 0)


def brute_force_cif(lp, times, cumhaz, t):
    """riskRegression's predictCIF_cpp loop, one patient and time at a time."""
    hazard = np.diff(cumhaz, axis=1, prepend=0.0)
    out = np.zeros((lp.shape[0], 3))
    for i in range(lp.shape[0]):
        risk = np.exp(lp[i])
        surv, cif = 1.0, np.zeros(3)
        for j in range(times.size):
            if times[j] > t:
                break
            cif += surv * risk * hazard[:, j]
            surv *= 1.0 - np.sum(risk * hazard[:, j])
        out[i] = cif
    return out


# ------------------------------------------------------- the reconstruction

def test_the_curve_is_riskregressions_loop():
    times, cumhaz = baseline_hazard(120, seed=1)
    lp = linear_predictors(40, seed=2)
    # Before the first time, on a training time, between two, past the last.
    at = np.array([0.0, times[0] / 2, times[3], (times[10] + times[11]) / 2,
                   times[-1], times[-1] + 50.0])
    shared, rows = CSCCurve(lp, (times, cumhaz), chunk=7).read(
        at, row_times=np.resize(at, 40))
    for p, t in enumerate(at):
        expected = brute_force_cif(lp, times, cumhaz, t)
        np.testing.assert_allclose(shared[:, :, p], expected, rtol=1e-12,
                                   atol=1e-15)
    np.testing.assert_allclose(
        rows, np.stack([shared[i, :, i % at.size] for i in range(40)]),
        rtol=0, atol=0)


def test_a_step_past_one_is_normalised():
    # One patient at lp = 0. The second step's summed hazard is 2.5, which
    # would turn the product-limit factor to -1.5.
    times = np.array([1.0, 2.0, 3.0])
    cumhaz = np.cumsum([[0.1, 1.0, 0.2], [0.1, 1.0, 0.2], [0.0, 0.5, 0.2]],
                       axis=1)
    shared, _ = CSCCurve(np.zeros((1, 3)), (times, cumhaz)).read(times)
    # Step 1: F = (0.1, 0.1, 0), S = 0.8. Step 2 shares the 0.8 left as
    # 1 : 1 : 0.5, and S = 0. Step 3 has nothing left to move.
    np.testing.assert_allclose(shared[0], [[0.1, 0.42, 0.42],
                                           [0.1, 0.42, 0.42],
                                           [0.0, 0.16, 0.16]], rtol=1e-14)


def test_the_curve_stays_a_distribution_past_an_oversized_step():
    times, cumhaz = baseline_hazard(60, seed=5)
    # The last steps outgrow a high-risk patient, as a tiny late risk set
    # makes them do.
    cumhaz[:, -5:] += np.cumsum(np.full((3, 5), 0.4), axis=1)
    lp = linear_predictors(50, seed=6)
    shared, _ = CSCCurve(lp, (times, cumhaz), chunk=7).read(times)
    total = shared.sum(axis=1)
    assert np.all(total <= 1.0 + 1e-12)
    assert np.all(np.diff(shared, axis=2) >= 0.0)
    step = (np.exp(lp)[:, :, None]
            * np.diff(cumhaz, axis=1, prepend=0.0)[None]).sum(axis=1)
    oversized = step > 1.0
    assert oversized.any() and not oversized.all(axis=1).any()
    # riskRegression's own loop leaves the simplex here ...
    loop = np.stack([brute_force_cif(lp, times, cumhaz, t) for t in times],
                    axis=2)
    assert np.any(loop.sum(axis=1) > 1.0)
    # ... and agrees with the curve up to each patient's first such step.
    first = np.where(oversized.any(axis=1), oversized.argmax(axis=1),
                     times.size)
    before = np.arange(times.size)[None, :] < first[:, None]
    np.testing.assert_allclose(shared.transpose(0, 2, 1)[before],
                               loop.transpose(0, 2, 1)[before], rtol=1e-12,
                               atol=1e-15)


@pytest.mark.skipif(not (REFERENCE / 'risk.csv').exists(),
                    reason='no riskRegression reference values: run '
                           'Rscript TransEHR2/tests/fixtures/'
                           'make_csc_reference.R with riskRegression '
                           'installed')
def test_the_curve_matches_riskregression():
    grid = TimeGrid([1e6], CAUSES)
    baseline = read_csc_baseline(str(REFERENCE / 'baseline_hazard.csv'), grid)
    patients = read_csc_lp(str(REFERENCE / 'lp.csv'), grid)
    risk = pd.read_csv(REFERENCE / 'risk.csv', float_precision='round_trip')
    query = np.sort(risk['time'].unique())
    shared, _ = CSCCurve(patients['lp'], baseline).read(query)
    position = pd.Index(patients['idx']).get_indexer(risk['idx'])
    got = shared[position, risk['cause'].to_numpy() - 1,
                 np.searchsorted(query, risk['time'])]
    product_limit = risk['product_limit'].to_numpy(dtype=bool)
    np.testing.assert_allclose(got[product_limit],
                               risk['risk'][product_limit], rtol=1e-10,
                               atol=1e-12)
    # The exponential form is a different curve, so the check above is the
    # one that pins which riskRegression uses.
    assert not np.allclose(got[~product_limit], risk['risk'][~product_limit],
                           rtol=1e-10, atol=1e-12)


# ------------------------------------------------ the dense recalibration

def val_and_test(grid, seed=3):
    times, cumhaz = baseline_hazard(300, seed=seed)
    val = {'lp': linear_predictors(600, seed + 1)}
    val['time_to_event'], val['event_type'] = late_readmissions(600, seed)
    test = {'lp': linear_predictors(400, seed + 2)}
    test['time_to_event'], test['event_type'] = late_readmissions(400,
                                                                  seed + 7)
    for split in (val, test):
        split['idx'] = np.arange(split['lp'].shape[0])
    return (times, cumhaz), val, test


def test_the_dense_offsets_at_the_cuts_are_the_grid_offsets():
    from evaluate_finetuned_predictions import outcomes

    grid = TimeGrid(CUTS, CAUSES)
    baseline, val, _ = val_and_test(grid)
    _, ci, ev, days = outcomes(val, grid)
    curve = CSCCurve(val['lp'], baseline)
    axis, dense = csc_dense_offsets(curve, baseline[0], days, ci, ev, grid)

    assert axis[0] == 0 and axis[-1] == grid.horizon_days
    cif, _ = curve.read(grid.cuts)
    np.testing.assert_allclose(
        dense[:, np.searchsorted(axis, grid.cuts)],
        aj_recalibration_offsets(cif, days, ci, ev, grid),
        rtol=0, atol=1e-15)


def test_every_recalibrated_read_comes_from_one_curve():
    from evaluate_finetuned_predictions import outcomes
    from recalibrate_predictions import recalibrate_csc

    grid = TimeGrid(CUTS, CAUSES)
    baseline, val, test = val_and_test(grid)
    _, ci, ev, days = outcomes(val, grid)
    axis, dense = csc_dense_offsets(CSCCurve(val['lp'], baseline),
                                    baseline[0], days, ci, ev, grid)

    # Both repairs have to act: the shifted curve dips, so the running
    # maximum lifts it, and some patients' causes sum past one at the
    # horizon.
    raw_on_axis, _ = CSCCurve(test['lp'], baseline).read(axis)
    shifted = raw_on_axis - dense[None]
    clipped = np.clip(shifted, 0.0, 1.0)
    assert np.any(np.diff(clipped[:, :, :-1], axis=2) < 0)
    lifted = np.maximum.accumulate(clipped, axis=2)
    assert np.sum(lifted[:, :, -1].sum(axis=1) > 1.0) > 10
    one_curve = _repair_shifted_cif(shifted)

    curve = CSCCurve(test['lp'], baseline, offsets=(axis, dense), chunk=33)
    # Every kind of read: the axis itself, a val event time and the float
    # after it, whole days, the cuts, and each patient's own time.
    val_event = days[ev][0]
    times = np.unique(np.concatenate([
        axis, [val_event, np.nextafter(val_event, np.inf)],
        np.arange(0.0, 436.0), grid.cuts]))
    _, _, _, test_days = outcomes(test, grid)
    shared, rows = curve.read(times, test_days)
    last = np.searchsorted(axis, times, side='right') - 1
    assert np.array_equal(shared, one_curve[:, :, last])
    own = np.searchsorted(axis, test_days, side='right') - 1
    assert np.array_equal(rows, one_curve[np.arange(own.size), :, own])
    assert curve.repaired_fraction == np.mean(one_curve != shifted)

    # The recalibrated dump is that curve at the cuts.
    recal = recalibrate_csc(val, test, baseline, grid)
    assert np.array_equal(
        recal['cif'], one_curve[:, :, np.searchsorted(axis, grid.cuts)])
    assert np.array_equal(recal['pmf'],
                          np.diff(recal['cif'], axis=2, prepend=0.0))
    assert np.all(recal['p_survive'] >= RECAL_FLOOR)
    assert np.all(np.diff(shared, axis=2) >= 0)


def test_the_axis_holds_both_sides_of_a_val_event():
    grid = TimeGrid(CUTS, CAUSES)
    axis = recalibration_axis(np.array([10.0, 500.0]), np.array([20.0, 30.0]),
                              np.array([True, False]), grid)
    assert np.array_equal(axis, [0.0, 10.0, 20.0, np.nextafter(20.0, np.inf),
                                 46.45, 178.67, 435.09])


# ------------------------------------------------------------- D-cal

def test_d_cal_on_a_step_curve_is_the_grid_d_cal_at_the_cuts():
    """With every time on a cut, Hyman and a step curve read the same."""
    grid = TimeGrid([30.0, 90.0, 365.0], CAUSES)
    data = simulate(grid, n=500, seed=16)
    rng = np.random.default_rng(17)
    cif, pmf, p_survive = data['cif'], data['pmf'], data['p_survive']
    ev = rng.uniform(size=500) < 0.6
    at = np.where(ev, rng.integers(0, 2, size=500),
                  rng.integers(0, 3, size=500))
    days = grid.cuts[at]
    ci = np.where(ev, rng.integers(0, 3, size=500), -1)

    rows = cif[np.arange(500), :, at]
    for k in range(grid.n_causes):
        values = (rows[:, k], cif[:, k, -1], 1.0 - rows.sum(axis=1), ci, ev,
                  k)
        prediction = (cif, pmf, p_survive, days, ci, ev, k, grid)
        assert cr_d_calibration_from_values(*values) == pytest.approx(
            cr_d_calibration(*prediction), rel=1e-9)
        assert cr_dcal_chi2_from_values(*values) == pytest.approx(
            cr_dcal_chi2(*prediction), rel=1e-9)


# ------------------------------------------------------------ end to end

def mixed_outcomes(n, seed):
    """Every cause, and censoring, spread over the whole grid."""
    rng = np.random.default_rng(seed + 2000)
    return (rng.uniform(0.5, 435.0, size=n) * MINUTES_PER_DAY,
            rng.integers(0, 4, size=n))


def write_csc_fold(folder, grid, seed, splits=('train', 'val', 'test')):
    """What fit_csc_baseline.R leaves in a fold."""
    folder.mkdir(parents=True, exist_ok=True)
    times, cumhaz = baseline_hazard(300, seed=seed)
    frame = {'time_days': times}
    for k, cause in enumerate(CAUSES):
        frame[f'cumhaz_{cause}'] = cumhaz[k]
    pd.DataFrame(frame).to_csv(
        folder / 'competing_risks_csc_baseline_hazard.csv', index=False)
    start = 0
    for s, split in enumerate(splits):
        n = 500 if split == 'train' else 300
        idx = np.arange(start, start + n)
        start += n
        tte, event = mixed_outcomes(n, seed + s)
        frame = {'idx': idx, 'time_to_event': tte, 'event_type': event}
        lp = linear_predictors(n, seed + 10 + s)
        for k, cause in enumerate(CAUSES):
            frame[f'lp_{cause}'] = lp[:, k]
        pd.DataFrame(frame).to_csv(
            folder / f'competing_risks_{split}_csc_lp.csv', index=False)


def test_the_csc_scripts_end_to_end(tmp_path, monkeypatch):
    import dump_baseline_csc_predictions
    import evaluate_finetuned_predictions
    import recalibrate_predictions
    from evaluate_finetuned_predictions import outcomes, read_predictions
    from TransEHR2.csc import read_csc_offsets
    from TransEHR2.survival import lee_weighted_concordance

    grid = TimeGrid(CUTS, CAUSES)
    config = tmp_path / 'experiment.yaml'
    config.write_text(yaml.dump({
        'TIME_GRID_CUTS_DAYS': CUTS, 'MODELLED_CAUSES': CAUSES,
        'BRIER_INTEGRATION_DAYS': 365, 'CR_DCAL_N_RHO': 200,
        'EVAL_HORIZONS_DAYS': [30, 90, 365]}))
    folder = tmp_path / 'csc' / 'fold1' / 'competing_risks'
    write_csc_fold(folder, grid, seed=30)

    def run(module):
        monkeypatch.setattr(sys, 'argv', [
            'script.py', 'csc', '--model_dir', str(tmp_path),
            '--experiment_config', str(config)])
        module.main()

    for module in (dump_baseline_csc_predictions, recalibrate_predictions,
                   evaluate_finetuned_predictions):
        run(module)

    baseline = read_csc_baseline(
        str(folder / 'competing_risks_csc_baseline_hazard.csv'), grid)
    lp = {s: read_csc_lp(str(folder / f'competing_risks_{s}_csc_lp.csv'),
                         grid) for s in ('train', 'val', 'test')}
    dump = read_predictions(
        str(folder / 'competing_risks_test_finetuned_output.csv'), grid)
    cif, _ = CSCCurve(lp['test']['lp'], baseline).read(grid.cuts)
    assert np.array_equal(dump['cif'], cif)
    np.testing.assert_allclose(dump['p_survive'],
                               1.0 - cif[:, :, -1].sum(axis=1), atol=1e-15)

    offsets = read_csc_offsets(
        str(folder / 'competing_risks_csc_recal_offsets.csv'), grid)
    recal_dump = read_predictions(
        str(folder / 'competing_risks_test_recal_aj_finetuned_output.csv'),
        grid)
    recal_cif, _ = CSCCurve(lp['test']['lp'], baseline,
                            offsets=offsets).read(grid.cuts)
    assert np.array_equal(recal_dump['cif'], recal_cif)
    with open(tmp_path / 'csc' / 'test_recal_aj_record.yaml') as f_in:
        assert yaml.safe_load(f_in)['folds']['fold1']['csc_curve']

    results = {}
    for name in ('train', 'val', 'test', 'test_recal_aj'):
        with open(tmp_path / 'csc' / f'{name}_evaluation.yaml') as f_in:
            results[name] = yaml.safe_load(f_in)
        assert results[name]['csc_curve']
    assert set(results['test']) == set(results['test_recal_aj'])
    for key in ('Readmission_Cindex_Lee_at_30d', 'Death_IBS_to_1y',
                'Out_Migration_IPA_to_90d', 'Readmission_CR_DCal',
                'Death_CR_DCal_Chi2_P', 'Readmission_Integrated_Brier'):
        for name in ('test', 'test_recal_aj'):
            assert np.isfinite(results[name][key][0]), (name, key)

    # The concordance read the exact curve at the horizon, not the grid.
    _, ci, ev, days = outcomes(dump, grid)
    _, _, train_ev, train_days = outcomes(lp['train'], grid)
    at_30, _ = CSCCurve(lp['test']['lp'], baseline).read(np.array([30.0]))
    assert results['test']['Readmission_Cindex_Lee_at_30d'][0] == \
        pytest.approx(lee_weighted_concordance(
            at_30[:, 0, 0], days, ci, ev, 0, 30.0, train_days, train_ev),
            rel=1e-12)


def test_csc_parameters_in_only_some_folds_are_refused(tmp_path,
                                                       monkeypatch):
    import dump_baseline_csc_predictions
    import evaluate_finetuned_predictions
    from dump_transehr2_deephit_predictions import save_predictions_csv

    grid = TimeGrid(CUTS, CAUSES)
    config = tmp_path / 'experiment.yaml'
    config.write_text(yaml.dump({
        'TIME_GRID_CUTS_DAYS': CUTS, 'MODELLED_CAUSES': CAUSES,
        'CR_DCAL_N_RHO': 50, 'EVAL_HORIZONS_DAYS': [30]}))
    write_csc_fold(tmp_path / 'exp' / 'fold1' / 'competing_risks', grid,
                   seed=40, splits=('train', 'test'))
    for s, split in enumerate(('train', 'test')):
        data = simulate(grid, n=200, seed=50 + s)
        save_predictions_csv(
            np.arange(200), data['time_to_event'], data['event_type'],
            data['cif'], data['pmf'], data['p_survive'], grid,
            str(tmp_path / 'exp' / 'fold2' / 'competing_risks'
                / f'competing_risks_{split}_finetuned_output.csv'))
    monkeypatch.setattr(sys, 'argv', [
        'script.py', 'exp', '--model_dir', str(tmp_path),
        '--experiment_config', str(config)])
    dump_baseline_csc_predictions.main()
    with pytest.raises(ValueError, match='some folds and not others'):
        evaluate_finetuned_predictions.main()
