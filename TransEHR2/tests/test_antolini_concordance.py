"""Antolini's concordance on [0, tau], read on the continuous curve."""

import numpy as np
import pytest

from TransEHR2.survival import (
    TimeGrid, antolini_concordance, cause_specific_concordance,
    interpolate_cif
)

from .cr_cohort import simulate

CAUSES = ["readmission", "death", "out_migration"]
CUTS = [8.0, 31.0, 91.0, 181.0]


def exponential_curves(rates):
    """``F(t) = 1 - exp(-rate t)`` per patient, as ``curve_at`` reads it."""
    return lambda times: 1.0 - np.exp(-np.outer(rates, times))


def brute_force(curve_at, days, cause, event, k, tau):
    """Every pair, one at a time: the definition, term for term."""
    concordant = pairs = 0.0
    for i in range(days.size):
        if not (event[i] and cause[i] == k and days[i] <= tau):
            continue
        risk = curve_at(np.array([days[i]]))[:, 0]
        for j in range(days.size):
            if days[j] > days[i]:
                pairs += 1
                concordant += (1.0 if risk[i] > risk[j]
                               else 0.5 if risk[i] == risk[j] else 0.0)
    return concordant / pairs if pairs else float("nan")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_it_is_the_pairwise_definition(seed):
    rng = np.random.default_rng(seed)
    n = 80
    # Whole days, so times tie, and a few equal rates, so risks tie.
    days = rng.integers(1, 60, n).astype(float)
    cause = rng.integers(-1, 2, n)
    event = cause >= 0
    rates = rng.choice([0.01, 0.02, 0.05, 0.1], n)
    curve_at = exponential_curves(rates)
    got = antolini_concordance(curve_at, days, cause, event, 0,
                               [7.0, 30.0, 60.0], chunk=5)
    for tau in (7.0, 30.0, 60.0):
        assert got[tau] == pytest.approx(
            brute_force(curve_at, days, cause, event, 0, tau), rel=1e-12)


def test_an_event_at_the_horizon_is_a_case_and_one_after_it_is_not():
    # Cases at 30.0 and 30.5; the 30.5 case is a control of the first.
    days = np.array([30.0, 30.5, 40.0])
    cause = np.array([0, 0, -1])
    event = np.array([True, True, False])
    curve_at = exponential_curves(np.array([0.3, 0.2, 0.1]))
    got = antolini_concordance(curve_at, days, cause, event, 0, [30.0, 31.0])
    assert got[30.0] == 1.0
    assert got[31.0] == 1.0
    curve_at = exponential_curves(np.array([0.1, 0.3, 0.2]))
    got = antolini_concordance(curve_at, days, cause, event, 0, [30.0, 31.0])
    assert got[30.0] == 0.0
    assert got[31.0] == pytest.approx(1.0 / 3.0)


def test_cases_in_one_bin_are_compared():
    """Days 10 and 25 share the bin [8, 31): the binned version compares
    neither with the other; this one does."""
    grid = TimeGrid(CUTS, CAUSES, brier_integration_days=30.0)
    days = np.array([10.0, 25.0])
    cause = np.array([0, 0])
    event = np.array([True, True])
    cif = np.zeros((2, 3, len(CUTS)))
    cif[:, 0, :] = [[0.2, 0.5, 0.6, 0.7], [0.1, 0.3, 0.4, 0.5]]
    bins = np.array([1, 1])
    assert np.isnan(cause_specific_concordance(cif, bins, cause, event, 0,
                                               horizon_bin=1))
    got = antolini_concordance(
        lambda t: interpolate_cif(cif[:, 0, :], grid, t), days, cause,
        event, 0, [30.0])
    assert got[30.0] == 1.0


def test_no_case_by_the_horizon_is_nan():
    days = np.array([40.0, 50.0])
    got = antolini_concordance(exponential_curves(np.array([0.1, 0.2])),
                               days, np.array([0, -1]),
                               np.array([True, False]), 0, [30.0])
    assert np.isnan(got[30.0])


def test_the_scorer_reports_it_at_each_horizon():
    from evaluate_finetuned_predictions import outcomes, score

    grid = TimeGrid(CUTS, CAUSES, brier_integration_days=30.0)
    data = simulate(grid, n=600, seed=5)
    _, ci, ev, days = outcomes(data, grid)
    dcal = {'alpha': 2.0, 'n_rho': 200, 'chi2_bins': 10, 'test_level': 0.05}
    out = score(data, grid, dcal, [7.0, 30.0])
    for k, key in enumerate(('Readmission', 'Death', 'Out_Migration')):
        expected = antolini_concordance(
            lambda t, k=k: interpolate_cif(data['cif'][:, k, :], grid, t),
            days, ci, ev, k, [7.0, 30.0])
        assert out[f'{key}_Cindex_Antolini_to_7d'] == expected[7.0]
        assert out[f'{key}_Cindex_Antolini_to_30d'] == expected[30.0]
        assert np.isfinite(expected[30.0])
