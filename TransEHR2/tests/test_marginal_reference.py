"""The covariate-free reference an integrated Brier score is read against.

An IBS has no scale on its own. It is dominated by how often the event happens, so a rare
outcome scores well under a prediction of nearly zero for everyone, and 0.0605 is unreadable
until you know what a model that learned nothing would have scored.

Aalen-Johansen rather than Kaplan-Meier: KM applied to one cause treats the others as
censoring, which answers "what if nobody could die" and gives a reference that is too easy to
beat. `test_kaplan_meier_on_one_cause_would_overstate_it` is that claim, measured.
"""

import numpy as np
import pytest

from TransEHR2.survival import (TimeGrid, aalen_johansen, brier_times,
                                cause_specific_brier, integrated_brier,
                                marginal_reference_metrics)

# Five episodes, two causes. t=3 is censored; the rest fail.
#   t=1 cause 0 | t=2 cause 1 | t=3 censored | t=4 cause 0 | t=5 cause 0
DAYS = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
CAUSE = np.array([0, 1, 0, 0, 0])
EVENT = np.array([True, True, False, True, True])


def test_the_curve_matches_a_hand_run_estimator():
    """Worked by hand, increment by increment:

        t=1  at risk 5, S=1.0  ->  F += 1.0 * 1/5 = 0.2,  S = 0.8
        t=2  at risk 4, S=0.8  ->  F += 0.8 * 0/4 = 0.0,  S = 0.6   (other cause)
        t=4  at risk 2, S=0.6  ->  F += 0.6 * 1/2 = 0.3,  S = 0.3
        t=5  at risk 1, S=0.3  ->  F += 0.3 * 1/1 = 0.3,  S = 0.0
    """
    taus = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0])
    expected = np.array([0.0, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.8, 0.8])
    assert np.allclose(aalen_johansen(DAYS, CAUSE, EVENT, 0, taus), expected)


def test_the_censored_episode_leaves_without_an_increment():
    """It thins the risk set at t=4 -- 2 rather than 3 -- and adds nothing of its own."""
    curve = aalen_johansen(DAYS, CAUSE, EVENT, 0, np.array([3.0, 4.0]))
    assert curve[0] == pytest.approx(0.2)
    assert curve[1] == pytest.approx(0.5)


def test_kaplan_meier_on_one_cause_would_overstate_it():
    """The reason for Aalen-Johansen. Treating the competing cause as censoring gives

        t=1  1/5 -> S=0.8   t=4  1/2 -> S=0.4   t=5  1/1 -> S=0.0

    so 1 - S reaches 1.0 by day 5: every patient readmitted, in a cohort where one of them
    died first. The competing-risks estimate stops at 0.8.
    """
    naive = 1.0
    for t, at_risk in ((1.0, 5), (4.0, 2), (5.0, 1)):
        naive *= 1.0 - 1 / at_risk
    assert 1.0 - naive == pytest.approx(1.0)
    assert aalen_johansen(DAYS, CAUSE, EVENT, 0,
                          np.array([5.0]))[0] == pytest.approx(0.8)


def test_the_curve_never_decreases():
    taus = np.linspace(0.0, 6.0, 25)
    curve = aalen_johansen(DAYS, CAUSE, EVENT, 0, taus)
    assert np.all(np.diff(curve) >= -1e-12)


def test_a_cause_that_never_happens_has_no_incidence():
    curve = aalen_johansen(DAYS, CAUSE, EVENT, 2, np.array([1.0, 5.0, 9.0]))
    assert np.all(curve == 0.0)


def test_a_cohort_with_no_events_has_no_incidence():
    censored = np.zeros(5, dtype=bool)
    curve = aalen_johansen(DAYS, CAUSE, censored, 0, np.array([1.0, 5.0]))
    assert np.all(curve == 0.0)


def test_the_marginal_curve_beats_any_other_constant_prediction():
    """What makes it the right reference rather than an arbitrary one: among predictions that
    ignore the inputs, the marginal incidence is the one the Brier score likes best. A
    reference that could be improved on without using a single covariate would flatter the
    model."""
    taus = brier_times(TimeGrid([2.0, 6.0], ['readmission', 'death'], 6.0))
    marginal = np.tile(aalen_johansen(DAYS, CAUSE, EVENT, 0, taus),
                       (DAYS.size, 1))

    def score(curve):
        return integrated_brier(
            cause_specific_brier(curve, DAYS, CAUSE, EVENT, 0, taus), taus)

    best = score(marginal)
    for constant in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert best <= score(np.full_like(marginal, constant)) + 1e-9


def test_the_reference_metrics_key_like_the_model_s_own():
    """So a caller can divide one by the other without rewriting either name."""
    minutes = DAYS * 24 * 60
    # EVENT_TYPE as labels.csv writes it: 0 censored, 1 readmission, 2 death.
    event_type = np.array([1, 2, 0, 1, 1])
    grid = TimeGrid([2.0, 6.0], ['readmission', 'death'], 6.0)

    out = marginal_reference_metrics(minutes, event_type, grid)
    assert np.isfinite(out['Readmission_Integrated_Brier_Null'])
    assert out['Readmission_Events'] == 3.0
    assert out['Censored'] == 1.0
