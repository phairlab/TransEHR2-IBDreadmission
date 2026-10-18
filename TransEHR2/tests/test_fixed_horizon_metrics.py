"""Lee's concordance, the Brier score to a horizon and the chi-squared pass
rule, on synthetic cohorts."""

import numpy as np
import pytest

from TransEHR2.survival import (
    MINUTES_PER_DAY, TimeGrid, interpolate_cif, lee_weighted_concordance,
    marginal_reference_metrics
)

from .cr_cohort import simulate


CUTS = [30.0, 90.0, 365.0]
CAUSES = ["readmission", "death", "out_migration"]


# --------------------------------------- chl8856/DeepHit utils_eval.py port

def lee_censoring_prob(Y, T):
    """``CensoringProb``, with lifelines' Kaplan-Meier written out.

    lifelines' ``survival_function_`` is indexed by zero and every observed
    duration; at each, the at-risk count is every duration at or after it
    and the "deaths" are the censorings (``Y == 0``) there.
    """
    T = T.reshape([-1])
    Y = Y.reshape([-1])
    timeline = np.unique(np.concatenate([[0.0], T]))
    at_risk = np.array([np.sum(T >= t) for t in timeline])
    removed = np.array([np.sum((T == t) & (Y == 0)) for t in timeline])
    G = np.stack([timeline, np.cumprod(1.0 - removed / at_risk)])
    G[1, G[1, :] == 0] = G[1, G[1, :] != 0][-1]
    return G


def lee_weighted_c_index(T_train, Y_train, Prediction, T_test, Y_test, Time):
    G = lee_censoring_prob(Y_train, T_train)

    N = len(Prediction)
    A = np.zeros((N, N))
    Q = np.zeros((N, N))
    N_t = np.zeros((N, N))
    for i in range(N):
        tmp_idx = np.where(G[0, :] >= T_test[i])[0]

        if len(tmp_idx) == 0:
            W = (1. / G[1, -1]) ** 2
        else:
            W = (1. / G[1, tmp_idx[0]]) ** 2

        A[i, np.where(T_test[i] < T_test)] = 1. * W
        Q[i, np.where(Prediction[i] > Prediction)] = 1.

        if T_test[i] <= Time and Y_test[i] == 1:
            N_t[i, :] = 1.

    Num = np.sum(((A) * N_t) * Q)
    Den = np.sum((A) * N_t)

    if Num == 0 and Den == 0:
        return -1
    return float(Num / Den)


def lee_c_index(Prediction, Time_survival, Death, Time):
    N = len(Prediction)
    A = np.zeros((N, N))
    Q = np.zeros((N, N))
    N_t = np.zeros((N, N))
    for i in range(N):
        A[i, np.where(Time_survival[i] < Time_survival)] = 1
        Q[i, np.where(Prediction[i] > Prediction)] = 1
        if Time_survival[i] <= Time and Death[i] == 1:
            N_t[i, :] = 1
    Num = np.sum(((A) * N_t) * Q)
    Den = np.sum((A) * N_t)
    if Num == 0 and Den == 0:
        return -1
    return float(Num / Den)


def cohort(grid, n, seed, whole_days=False):
    """``(days, cause_index, is_event, cif)`` of a simulated split."""
    from evaluate_finetuned_predictions import outcomes

    data = simulate(grid, n=n, seed=seed)
    if whole_days:
        # Ties in time, within the split and across it, are where the strict
        # T_i < T_j and the lookup of G at the next training time show.
        data['time_to_event'] = (np.round(data['time_to_event']
                                          / MINUTES_PER_DAY)
                                 * MINUTES_PER_DAY)
    _, ci, ev, days = outcomes(data, grid)
    return days, ci, ev, data['cif']


@pytest.mark.parametrize('whole_days', [False, True])
@pytest.mark.parametrize('tau', [30.0, 90.0, 200.0, 365.0])
def test_lee_concordance_is_lees_with_g_from_true_censoring(tau, whole_days):
    grid = TimeGrid(CUTS, CAUSES)
    days, ci, ev, cif = cohort(grid, 300, seed=4, whole_days=whole_days)
    train_days, _, train_ev, _ = cohort(grid, 400, seed=5,
                                        whole_days=whole_days)
    assert (~train_ev).sum() > 50

    for k in range(grid.n_causes):
        risk = interpolate_cif(cif[:, k, :], grid, np.array([tau]))[:, 0]
        # Ties in the score, which the strict comparison counts as nothing.
        risk = np.round(risk, 2)
        expected = lee_weighted_c_index(
            train_days, train_ev.astype(int), risk, days,
            (ev & (ci == k)).astype(int), tau)
        assert expected != -1
        got = lee_weighted_concordance(risk, days, ci, ev, k, tau,
                                       train_days, train_ev)
        assert got == pytest.approx(expected, rel=1e-12)


def test_lee_concordance_without_censoring_is_the_unweighted_c_index():
    grid = TimeGrid(CUTS, CAUSES)
    days, ci, ev, cif = cohort(grid, 300, seed=6)
    k, tau = 0, 365.0
    risk = interpolate_cif(cif[:, k, :], grid, np.array([tau]))[:, 0]
    uncensored = np.ones(days.size, dtype=bool)
    got = lee_weighted_concordance(risk, days, ci, ev, k, tau, days,
                                   uncensored)
    assert got == pytest.approx(
        lee_c_index(risk, days, (ev & (ci == k)).astype(int), tau),
        rel=1e-12)


def test_lee_concordance_with_no_case_is_nan():
    grid = TimeGrid(CUTS, CAUSES)
    days, ci, ev, cif = cohort(grid, 50, seed=7)
    assert np.isnan(lee_weighted_concordance(
        cif[:, 0, 0], days, np.full_like(ci, -1), np.zeros_like(ev), 0, 365.0,
        days, ev))


# ---------------------------------------------------------- the score keys

def score_split(grid, n=800, seed=8):
    from evaluate_finetuned_predictions import outcomes, score

    data = simulate(grid, n=n, seed=seed)
    train = simulate(grid, n=n, seed=seed + 1)
    _, _, train_ev, train_days = outcomes(train, grid)
    dcal = {'alpha': 2.0, 'n_rho': 200, 'chi2_bins': 10, 'test_level': 0.05}
    return data, score(data, grid, dcal, [30.0, 365.0],
                       (train_days, train_ev))


def test_ibs_to_a_year_is_the_integrated_brier_to_a_year():
    grid = TimeGrid(CUTS, CAUSES, brier_integration_days=365.0)
    data, out = score_split(grid)
    reference = marginal_reference_metrics(data['time_to_event'],
                                           data['event_type'], grid)
    for cause in ('Readmission', 'Death', 'Out_Migration'):
        assert out[f'{cause}_IBS_to_1y'] == out[f'{cause}_Integrated_Brier']
        assert out[f'{cause}_IBS_Null_to_1y'] == pytest.approx(
            reference[f'{cause}_Integrated_Brier_Null'], rel=1e-12)
        assert out[f'{cause}_IPA_to_1y'] == pytest.approx(
            1 - out[f'{cause}_IBS_to_1y'] / out[f'{cause}_IBS_Null_to_1y'])
    # A shorter bound is a different number, not a relabelled one.
    assert out['Readmission_IBS_to_30d'] != out['Readmission_IBS_to_1y']


@pytest.mark.parametrize('p, passes', [(0.10, 0.0), (0.16, 1.0)])
def test_a_cause_passes_the_chi_squared_at_level_times_k(p, passes):
    from evaluate_finetuned_predictions import chi2_pass

    # K = 3 at a level of 0.05: a cause passes above 0.15.
    assert chi2_pass([p, 0.9, 0.9], 0.05) == passes
    assert chi2_pass([0.9, p, 0.9], 0.05) == passes
