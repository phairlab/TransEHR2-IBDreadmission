"""Competing-risks D-calibration, its chi-squared test, and AJ-k recalibration.

The buckets are checked against a cohort worked by hand. The metric and the
test are checked against an oracle -- outcomes drawn from the predictions
themselves, which a calibrated model is by definition -- and against the
same predictions with a cause's mass halved, which no amount of noise
should pass.
"""

import numpy as np
import pytest
import torch

from scipy.stats import chi2

from TransEHR2.survival import (
    MINUTES_PER_DAY, RECAL_FLOOR, TimeGrid, aalen_johansen,
    aj_recalibration_offsets, apply_aj_recalibration, cr_d_calibration,
    cr_dcal_chi2, cr_dcal_numerator, event_times_days
)

from .cr_cohort import simulate


CUTS = [30.0, 90.0, 365.0]
TWO_CAUSES = ["readmission", "death"]


def outcomes(data, grid):
    _, ci, ev = grid.discretize(
        torch.as_tensor(data['time_to_event'], dtype=torch.float64),
        torch.as_tensor(data['event_type']))
    return (event_times_days(data['time_to_event'], grid), ci.numpy(),
            ev.numpy().astype(bool))


# Three patients, every time on a cut so the spline is read at its knots.
#   A: readmitted at 90 d.  cause 0 pmf [.1 .2 .2] -> F0 = [.1 .3 .5]
#   B: died at 30 d, the same surface as A.  Contributes nothing to cause 0.
#   C: censored at 30 d.    cause 0 pmf [.2 .2 .2] -> F0 = [.2 .4 .6]
#                           cause 1 pmf [.05 .05 .1], p_survive 0.2
PMF = np.array([
    [[0.1, 0.2, 0.2], [0.1, 0.1, 0.1]],
    [[0.1, 0.2, 0.2], [0.1, 0.1, 0.1]],
    [[0.2, 0.2, 0.2], [0.05, 0.05, 0.1]],
])
P_SURVIVE = np.array([0.2, 0.2, 0.2])
HAND = {
    'cif': np.cumsum(PMF, axis=2),
    'pmf': PMF,
    'p_survive': P_SURVIVE,
    'time_to_event': np.array([90.0, 30.0, 30.0]) * MINUTES_PER_DAY,
    'event_type': np.array([1, 2, 0]),
}


def hand_args(grid):
    days, ci, ev = outcomes(HAND, grid)
    return (HAND['cif'], HAND['pmf'], HAND['p_survive'], days, ci, ev, 0,
            grid)


def test_the_buckets_match_a_hand_count():
    """Worked by hand:

        A: u = F0(90)/F0(365) = .3/.5 = .6, an event of cause 0: counts 1
           once rho >= .6.
        C: u = .2/.6 = 1/3, censored; S(30) = .2 + (.6-.2) + (.2-.05) = .75,
           so it adds (.6 rho - .2)/.75 once rho >= 1/3.
        Denominator: .5 + .5 + .6 = 1.6.

        N(.25) = 0
        N(.5)  = (.3 - .2)/.75             = 2/15
        N(.75) = 1 + (.45 - .2)/.75        = 4/3
        N(1)   = 1 + (.6 - .2)/.75         = 23/15
    """
    grid = TimeGrid(CUTS, TWO_CAUSES)
    numerator, denominator = cr_dcal_numerator(
        *hand_args(grid), np.array([0.25, 0.5, 0.75, 1.0]))
    assert denominator == pytest.approx(1.6)
    assert numerator == pytest.approx([0.0, 2 / 15, 4 / 3, 23 / 15])


def test_the_metric_is_algorithm_one_on_those_buckets():
    grid = TimeGrid(CUTS, TWO_CAUSES)
    b = np.array([0.0, 2 / 15, 4 / 3, 23 / 15]) / 1.6
    rho = np.array([0.25, 0.5, 0.75, 1.0])
    expected = np.sqrt(np.mean((b - rho) ** 2))
    assert cr_d_calibration(*hand_args(grid), alpha=2.0, n_rho=4) == \
        pytest.approx(expected)


def test_the_chi_squared_bins_are_differences_of_the_buckets():
    """Two bins: O = [2/15, 23/15 - 2/15], E = 1.6/2 each, df = 2."""
    grid = TimeGrid(CUTS, TWO_CAUSES)
    observed = np.array([2 / 15, 21 / 15])
    statistic = np.sum((observed - 0.8) ** 2 / 0.8)
    x2, p = cr_dcal_chi2(*hand_args(grid), n_bins=2)
    assert x2 == pytest.approx(statistic)
    assert p == pytest.approx(chi2.sf(statistic, df=2))


def test_a_patient_censored_at_the_horizon_adds_nothing():
    """u = 1 and F(tau) * 1 - F(tau) = 0: observed through the grid, no claim."""
    grid = TimeGrid(CUTS, TWO_CAUSES)
    data = {**HAND, 'time_to_event': np.array([90.0, 30.0, 365.0])
            * MINUTES_PER_DAY}
    days, ci, ev = outcomes(data, grid)
    numerator, _ = cr_dcal_numerator(
        data['cif'], data['pmf'], data['p_survive'], days, ci, ev, 0, grid,
        np.array([1.0]))
    assert numerator == pytest.approx([1.0])


@pytest.fixture(scope='module')
def oracle():
    grid = TimeGrid(CUTS, TWO_CAUSES)
    return grid, simulate(grid, n=20000, seed=0)


def scores(grid, data, k):
    days, ci, ev = outcomes(data, grid)
    args = (data['cif'], data['pmf'], data['p_survive'], days, ci, ev, k,
            grid)
    return cr_d_calibration(*args), cr_dcal_chi2(*args)[1]


@pytest.mark.parametrize('k', [0, 1])
def test_a_calibrated_model_scores_near_zero_and_passes(oracle, k):
    grid, data = oracle
    distance, p_value = scores(grid, data, k)
    assert distance < 0.02
    assert p_value > 0.01


def test_halving_a_cause_is_caught_by_both(oracle):
    grid, data = oracle
    halved = data['pmf'].copy()
    halved[:, 0, :] *= 0.5
    wrong = {**data, 'pmf': halved, 'cif': np.cumsum(halved, axis=2),
             'p_survive': data['p_survive'] + data['pmf'][:, 0, :].sum(1) / 2}
    distance, p_value = scores(grid, wrong, 0)
    assert distance > 0.3
    assert p_value < 1e-6


# ----------------------------------------------------- AJ-k recalibration

def test_recalibrating_a_shared_prediction_lands_it_on_aalen_johansen():
    """Everyone predicted alike: the shifted curve *is* the AJ curve.

    Nothing leaves the simplex, so nothing is repaired, and the shifted
    incidences, the rebuilt pmf and the rebuilt survival still partition
    the mass.
    """
    grid = TimeGrid(CUTS, TWO_CAUSES)
    data = simulate(grid, n=3000, seed=1)
    shared = {**data, 'pmf': np.broadcast_to(data['pmf'][:1],
                                             data['pmf'].shape).copy()}
    shared['cif'] = np.cumsum(shared['pmf'], axis=2)
    shared['p_survive'] = np.full(3000, data['p_survive'][0])
    days, ci, ev = outcomes(data, grid)

    offsets = aj_recalibration_offsets(shared['cif'], days, ci, ev, grid)
    cif, pmf, p_survive, repaired = apply_aj_recalibration(
        shared['cif'], shared['pmf'], shared['p_survive'], offsets)

    for k in range(2):
        aj = aalen_johansen(days, ci, ev, k, np.nextafter(grid.cuts, -np.inf))
        assert np.allclose(cif[:, k, :], aj[None, :], atol=1e-12)
    assert repaired == 0.0
    assert np.allclose(np.cumsum(pmf, axis=2), cif, atol=1e-12)
    assert np.allclose(cif[:, :, -1].sum(1) + p_survive, 1.0, atol=1e-12)


def test_the_repair_clips_lifts_and_rescales():
    """Two patients, one shift; the first leaves the simplex three ways.

        cause 0  F = [.01 .10 .12], d = [.03 0 .05]
                 shifted [-.02 .10 .07] -> clip [0 .10 .07] -> max [0 .10 .10]
        cause 1  F = [.3 .5 .8],    d = [0 0 -.2]
                 shifted [.3 .5 1.0]
        sum at the horizon .10 + 1.0 = 1.1 -> every cell / 1.1

    The second, F0 = [.2 .3 .4] and F1 = [.1 .2 .3], shifts to [.17 .30
    .35] and [.1 .2 .5], sums to .85, and is left alone: six of twelve
    cells repaired.
    """
    cif = np.array([
        [[0.01, 0.10, 0.12], [0.3, 0.5, 0.8]],
        [[0.2, 0.3, 0.4], [0.1, 0.2, 0.3]],
    ])
    pmf = np.diff(cif, axis=2, prepend=0.0)
    p_survive = 1.0 - cif[:, :, -1].sum(1)
    offsets = np.array([[0.03, 0.0, 0.05], [0.0, 0.0, -0.2]])

    new_cif, new_pmf, new_survive, repaired = apply_aj_recalibration(
        cif, pmf, p_survive, offsets)

    assert new_cif[0, 0] == pytest.approx(np.array([0.0, 0.10, 0.10]) / 1.1)
    assert new_cif[0, 1] == pytest.approx(np.array([0.3, 0.5, 1.0]) / 1.1)
    assert new_cif[1, 0] == pytest.approx([0.17, 0.30, 0.35])
    assert new_cif[1, 1] == pytest.approx([0.1, 0.2, 0.5])
    assert repaired == 0.5
    # The first patient's causes now use all the mass: survival is floored.
    assert new_survive == pytest.approx([RECAL_FLOOR, 0.15])
    assert np.allclose(np.cumsum(new_pmf, axis=2), new_cif, atol=1e-15)


def test_an_incidence_shifted_to_nothing_is_floored_at_the_horizon():
    """F_k(tau) divides every u; a cause shifted out entirely keeps a sliver."""
    cif = np.array([[[0.01, 0.02, 0.03], [0.1, 0.2, 0.3]]])
    pmf = np.diff(cif, axis=2, prepend=0.0)
    offsets = np.array([[0.05, 0.05, 0.05], [0.0, 0.0, 0.0]])
    new_cif, _, _, _ = apply_aj_recalibration(cif, pmf, np.array([0.67]),
                                              offsets)
    assert new_cif[0, 0].tolist() == [0.0, 0.0, RECAL_FLOOR]
