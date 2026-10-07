"""The discrete-time competing-risks machinery, checked by hand.

Every assertion here is against a number worked out independently of the
implementation -- a counted pair, a hand-run Kaplan-Meier, a likelihood
written out term by term. A test that recomputes the code's own arithmetic
would pass whatever the code did.
"""

import math
import numpy as np
import pytest
import torch

from TransEHR2.losses import DeepHitLoss
from TransEHR2.survival import (
    CAUSE_CODES,
    DEFAULT_CAUSES,
    deephit_distribution,
    CENSORED,
    DEATH,
    MINUTES_PER_DAY,
    OUT_MIGRATION,
    READMISSION,
    TimeGrid,
    brier_times,
    cause_specific_brier,
    cause_specific_concordance,
    censoring_survival,
    cif_from_pmf,
    cif_logit_target,
    event_times_days,
    integrated_brier,
    interpolate_cif,
    pmf_from_logits,
    survival_from_pmf,
    survival_metrics,
)


CUTS = [30.0, 90.0, 365.0]
TWO_CAUSES = ["readmission", "death"]


def minutes(days):
    return torch.tensor([d * MINUTES_PER_DAY for d in days],
                        dtype=torch.float64)


# ------------------------------------------------------------------ the grid

def test_grid_shape_and_edges():
    grid = TimeGrid(CUTS, TWO_CAUSES)
    assert grid.n_bins == 3
    assert grid.n_causes == 2
    assert grid.horizon_days == 365.0
    assert list(grid.edges_days) == [0.0, 30.0, 90.0, 365.0]


@pytest.mark.parametrize("cuts", [[], [30.0, 30.0], [90.0, 30.0], [-1.0]])
def test_grid_rejects_bad_cuts(cuts):
    with pytest.raises(ValueError):
        TimeGrid(cuts)


def test_the_brier_bound_defaults_to_one_year():
    assert TimeGrid(CUTS, TWO_CAUSES).brier_integration_days == 365.0


def test_the_brier_bound_may_not_outrun_the_grid():
    """Past the last cut the head predicts nothing to score."""
    with pytest.raises(ValueError, match='grid ends at'):
        TimeGrid([30.0, 90.0], TWO_CAUSES, brier_integration_days=365.0)


@pytest.mark.parametrize("bound", [0.0, -1.0, float('nan')])
def test_the_brier_bound_must_be_positive(bound):
    with pytest.raises(ValueError, match='must be positive'):
        TimeGrid(CUTS, TWO_CAUSES, brier_integration_days=bound)


def test_the_brier_axis_runs_from_zero_to_the_bound_by_days():
    taus = brier_times(TimeGrid(CUTS, TWO_CAUSES,
                                brier_integration_days=90.0))
    assert taus[0] == 0.0 and taus[-1] == 90.0
    assert taus.size == 91


def test_bins_are_left_closed():
    """A time landing exactly on a cut opens the next bin, not closes one."""
    grid = TimeGrid(CUTS, TWO_CAUSES)
    got = grid.bin_of(torch.tensor([0.0, 29.9, 30.0, 89.9, 90.0, 364.9]))
    assert got.tolist() == [0, 0, 1, 1, 2, 2]


def test_discretize_maps_causes_to_their_output_position():
    grid = TimeGrid(CUTS, TWO_CAUSES)
    t = minutes([10.0, 10.0, 10.0, 10.0])
    e = torch.tensor([READMISSION, DEATH, CENSORED, OUT_MIGRATION])
    b, c, ev = grid.discretize(t, e)

    assert b.tolist() == [0, 0, 0, 0]
    assert c.tolist() == [0, 1, -1, -1]
    assert ev.tolist() == [True, True, False, False]


def test_out_migration_is_censoring():
    """EVENT_TYPE 3 and EVENT_TYPE 0 are indistinguishable to the model."""
    grid = TimeGrid(CUTS, TWO_CAUSES)
    t = minutes([50.0, 50.0])
    _, c, ev = grid.discretize(t, torch.tensor([CENSORED, OUT_MIGRATION]))
    assert c.tolist() == [-1, -1]
    assert ev.tolist() == [False, False]


def test_beyond_the_horizon_is_censored_at_the_horizon():
    grid = TimeGrid(CUTS, TWO_CAUSES)
    t = minutes([364.0, 365.0, 4000.0])
    e = torch.tensor([READMISSION, READMISSION, DEATH])
    b, c, ev = grid.discretize(t, e)

    assert ev.tolist() == [True, False, False]
    assert c.tolist() == [0, -1, -1]
    # All three sit in the last bin: the two late ones were observed through
    # the whole grid without an outcome it predicts.
    assert b.tolist() == [2, 2, 2]


def test_discretize_accepts_a_column_vector():
    """collate_tensorized hands targets back as (batch, 1)."""
    grid = TimeGrid(CUTS, TWO_CAUSES)
    t = minutes([10.0, 200.0]).reshape(-1, 1)
    e = torch.tensor([[READMISSION], [DEATH]])
    b, c, ev = grid.discretize(t, e)
    assert b.shape == (2,) and c.shape == (2,) and ev.shape == (2,)
    assert ev.tolist() == [True, True]


# ------------------------------------------------------------------- the CIF

def test_pmf_leaves_room_for_surviving_the_horizon():
    """The (cause, bin) mass must stop short of 1, or nobody can survive."""
    logits = torch.randn(7, 2 * 3)
    pmf, p_survive = deephit_distribution(logits, 2, 3)
    assert pmf.shape == (7, 2, 3) and p_survive.shape == (7,)
    assert torch.allclose(pmf.sum(dim=(1, 2)) + p_survive, torch.ones(7),
                          atol=1e-6)
    assert (survival_from_pmf(pmf, p_survive)[:, -1] > 0).all()


def test_survival_is_positive_even_when_the_head_shouts():
    """A saturated logit must not drive last-bin survival to exactly zero.

    ``1 - sum`` cancels the answer away here; summing from the tail keeps
    it.
    """
    pmf, p_survive = deephit_distribution(torch.full((3, 2 * 3), 30.0), 2, 3)
    assert (survival_from_pmf(pmf, p_survive)[:, -1] > 0).all()
    assert ((1.0 - pmf.sum(dim=(1, 2))) == 0).all()   # the route not taken


def test_cif_and_survival_partition_the_mass():
    pmf = torch.tensor([[[0.1, 0.2, 0.05], [0.05, 0.1, 0.1]]])
    cif = cif_from_pmf(pmf)
    assert torch.allclose(cif[0, 0], torch.tensor([0.1, 0.3, 0.35]))
    assert torch.allclose(cif[0, 1], torch.tensor([0.05, 0.15, 0.25]))
    # 1 - (F_1 + F_2) at each bin; 0.4 of the mass is never spent.
    assert torch.allclose(survival_from_pmf(pmf, torch.tensor([0.4]))[0],
                          torch.tensor([0.85, 0.55, 0.40]), atol=1e-6)


# ----------------------------------------------- the attribution target

def test_the_attribution_target_is_the_log_odds_of_the_cause_by_the_horizon():
    torch.manual_seed(0)
    logits = torch.randn(4, 3 * 6)
    incidence = cif_from_pmf(deephit_distribution(logits, 3, 6)[0])[:, 0, 2]
    expected = (torch.log(incidence) - torch.log1p(-incidence)).sum()
    assert torch.allclose(cif_logit_target(0, 2, 3, 6)(logits), expected,
                          atol=1e-6)


def test_the_target_is_monotone_in_the_incidence_and_does_not_saturate():
    """Why the log-odds and not the incidence: a confident head still has a gradient."""
    logits = torch.zeros(1, 2 * 3, requires_grad=True)
    with torch.no_grad():
        logits[0, 0] = 30.0                      # cause 0, bin 0: F -> 1
    target = cif_logit_target(0, 0, 2, 3)
    value = target(logits)
    value.backward()
    assert torch.isfinite(value)
    assert logits.grad.abs().sum() > 0, 'a saturated probability would give zeros'


def test_the_target_accumulates_over_the_bins_up_to_the_horizon():
    torch.manual_seed(1)
    logits = torch.randn(2, 2 * 4)
    values = [cif_logit_target(1, b, 2, 4)(logits) for b in range(4)]
    assert all(values[b] < values[b + 1] for b in range(3))


@pytest.mark.parametrize('cause,horizon_bin', [(-1, 0), (3, 0), (0, -1), (0, 6)])
def test_the_target_rejects_an_index_off_the_grid(cause, horizon_bin):
    with pytest.raises(ValueError):
        cif_logit_target(cause, horizon_bin, 3, 6)


# ------------------------------------------------------------------ the loss

def _loss(n_bins=3, **kw):
    return DeepHitLoss(n_causes=2, n_bins=n_bins, **kw)


def test_likelihood_matches_the_terms_written_out():
    """One event of each cause and one censored row, computed by hand."""
    torch.manual_seed(0)
    logits = torch.randn(3, 6)
    pmf, p_survive = deephit_distribution(logits, 2, 3)
    surv = survival_from_pmf(pmf, p_survive)

    bin_index = torch.tensor([0, 2, 1])
    cause_index = torch.tensor([0, 1, -1])
    is_event = torch.tensor([True, True, False])

    expected = -(torch.log(pmf[0, 0, 0] + 1e-8)
                 + torch.log(pmf[1, 1, 2] + 1e-8)
                 + torch.log(surv[2, 1] + 1e-8)) / 3

    _, (nll, _) = _loss(rank_weight=0.0)(logits, bin_index, cause_index,
                                         is_event)
    assert torch.allclose(nll, expected, atol=1e-6)


def test_likelihood_falls_when_mass_moves_to_the_truth():
    bin_index = torch.tensor([1])
    cause_index = torch.tensor([0])
    is_event = torch.tensor([True])
    fn = _loss(rank_weight=0.0)

    vague = torch.zeros(1, 6)
    sharp = torch.tensor([[0.0, 8.0, 0.0, 0.0, 0.0, 0.0]])
    assert fn(sharp, bin_index, cause_index, is_event)[1][0] < \
        fn(vague, bin_index, cause_index, is_event)[1][0]


def test_censored_row_rewards_survival_not_a_cause():
    """Moving mass off the grid should help a censored episode."""
    bin_index = torch.tensor([0])
    cause_index = torch.tensor([-1])
    is_event = torch.tensor([False])
    fn = _loss(rank_weight=0.0)

    # All mass in bin 0 -- the patient is predicted to have already failed.
    early = torch.tensor([[8.0, 0.0, 0.0, 8.0, 0.0, 0.0]])
    # All mass in the last bin -- they survive bin 0, which is all we know.
    late = torch.tensor([[0.0, 0.0, 8.0, 0.0, 0.0, 8.0]])
    assert fn(late, bin_index, cause_index, is_event)[1][0] < \
        fn(early, bin_index, cause_index, is_event)[1][0]


def test_ranking_term_prefers_the_right_order():
    """The case who failed should outrank the control who outlasted them."""
    bin_index = torch.tensor([0, 2])
    cause_index = torch.tensor([0, -1])
    is_event = torch.tensor([True, False])
    fn = _loss(rank_weight=1.0)

    # Row 0 failed of cause 0 in bin 0. Correct: row 0 carries more early
    # cause-0 incidence than row 1 does.
    right = torch.tensor([[6.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 6.0, 0.0, 0.0, 0.0]])
    wrong = right.flip(0)
    assert fn(right, bin_index, cause_index, is_event)[1][1] < \
        fn(wrong, bin_index, cause_index, is_event)[1][1]


def test_ranking_term_is_zero_without_an_acceptable_pair():
    """Everyone censored: no case, so nothing to order."""
    logits = torch.randn(4, 6)
    bin_index = torch.tensor([0, 1, 2, 2])
    cause_index = torch.full((4,), -1)
    is_event = torch.zeros(4, dtype=torch.bool)
    _, (_, ranking) = _loss(rank_weight=1.0)(logits, bin_index, cause_index,
                                             is_event)
    assert float(ranking) == 0.0


def test_rank_weight_zero_leaves_the_likelihood_alone():
    logits = torch.randn(5, 6)
    bin_index = torch.tensor([0, 1, 2, 1, 0])
    cause_index = torch.tensor([0, 1, -1, 0, -1])
    is_event = torch.tensor([True, True, False, True, False])

    total, (nll, ranking) = _loss(rank_weight=0.0)(
        logits, bin_index, cause_index, is_event)
    assert float(ranking) == 0.0
    assert torch.allclose(total, nll)


def test_cause_weights_scale_only_the_ranking_term():
    logits = torch.randn(6, 6)
    bin_index = torch.tensor([0, 0, 1, 1, 2, 2])
    cause_index = torch.tensor([0, 1, 0, 1, -1, -1])
    is_event = torch.tensor([True, True, True, True, False, False])

    base = _loss(rank_weight=1.0)
    heavy = _loss(rank_weight=1.0, cause_weights=[3.0, 1.0])
    _, (nll_a, rank_a) = base(logits, bin_index, cause_index, is_event)
    _, (nll_b, rank_b) = heavy(logits, bin_index, cause_index, is_event)

    assert torch.allclose(nll_a, nll_b)
    assert float(rank_b) > float(rank_a)


def test_bad_cause_weights_are_rejected():
    with pytest.raises(ValueError):
        DeepHitLoss(n_causes=2, n_bins=3, cause_weights=[1.0])


def test_loss_backpropagates():
    logits = torch.randn(8, 6, requires_grad=True)
    bin_index = torch.tensor([0, 1, 2, 0, 1, 2, 1, 0])
    cause_index = torch.tensor([0, 1, -1, 1, 0, -1, 0, -1])
    is_event = torch.tensor([True, True, False, True, True, False,
                             True, False])
    total, _ = _loss(rank_weight=1.0)(logits, bin_index, cause_index,
                                      is_event)
    total.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


# --------------------------------------------------------------- concordance

def test_concordance_counts_the_pairs_by_hand():
    """Four episodes, one cause-0 case, three comparable controls.

    Row 0 fails of cause 0 in bin 0. Rows 1-3 all outlast it. F_0 at bin 0
    is 0.9 for the case and 0.8, 0.9, 0.95 for the controls: one control
    below (concordant), one level (half) and one above (discordant), so
    C = (1 + 0.5) / 3.
    """
    cif = np.zeros((4, 2, 3))
    cif[:, 0, 0] = [0.90, 0.80, 0.90, 0.95]
    bin_index = np.array([0, 1, 2, 2])
    cause_index = np.array([0, -1, -1, -1])
    is_event = np.array([True, False, False, False])

    got = cause_specific_concordance(cif, bin_index, cause_index, is_event, 0)
    assert got == pytest.approx(1.5 / 3)


def test_concordance_ignores_controls_censored_before_the_case():
    """A control who left before the case failed is not comparable."""
    cif = np.zeros((3, 2, 3))
    cif[:, 0, 1] = [0.9, 0.1, 0.1]
    bin_index = np.array([1, 0, 2])
    cause_index = np.array([0, -1, -1])
    is_event = np.array([True, False, False])

    # Only row 2 outlasts the case, and the case outranks it: C = 1.
    assert cause_specific_concordance(
        cif, bin_index, cause_index, is_event, 0) == pytest.approx(1.0)


def test_concordance_is_nan_without_events():
    cif = np.random.rand(5, 2, 3)
    bin_index = np.array([0, 1, 2, 1, 0])
    cause_index = np.full(5, -1)
    is_event = np.zeros(5, dtype=bool)
    assert math.isnan(
        cause_specific_concordance(cif, bin_index, cause_index, is_event, 0))


def test_concordance_horizon_restricts_to_earlier_cases():
    """Restricting to bin 0 drops the bin-1 case, flipping the answer."""
    cif = np.zeros((4, 2, 3))
    cif[:, 0, 0] = [0.9, 0.1, 0.1, 0.1]   # early case ranks correctly
    cif[:, 0, 1] = [0.9, 0.9, 0.1, 0.9]   # late case ranks wrongly
    bin_index = np.array([0, 2, 1, 2])
    cause_index = np.array([0, -1, 0, -1])
    is_event = np.array([True, False, True, False])

    at_first = cause_specific_concordance(
        cif, bin_index, cause_index, is_event, 0, horizon_bin=0)
    overall = cause_specific_concordance(
        cif, bin_index, cause_index, is_event, 0)
    assert at_first == pytest.approx(1.0)
    assert overall < at_first


def test_concordance_agrees_with_the_brute_force_pair_count():
    """The binned shortcut must give what enumerating every pair gives."""
    rng = np.random.default_rng(3)
    n, n_bins = 200, 4
    cif = np.sort(rng.random((n, 2, n_bins)), axis=-1)
    bin_index = rng.integers(0, n_bins, n)
    is_event = rng.random(n) < 0.5
    cause_index = np.where(is_event, rng.integers(0, 2, n), -1)

    for k in (0, 1):
        num = den = 0.0
        for i in range(n):
            if not (is_event[i] and cause_index[i] == k):
                continue
            t = bin_index[i]
            for j in range(n):
                if bin_index[j] <= t:
                    continue
                den += 1
                if cif[i, k, t] > cif[j, k, t]:
                    num += 1
                elif cif[i, k, t] == cif[j, k, t]:
                    num += 0.5
        expected = num / den if den else float("nan")
        got = cause_specific_concordance(cif, bin_index, cause_index,
                                         is_event, k)
        assert got == pytest.approx(expected)


# ------------------------------------------------------------ Brier and IPCW

def test_event_times_are_truncated_at_the_horizon():
    """Minutes in, days out, and nothing past the last cut.

    The continuous reading of an episode and the binned one have to agree
    about where the grid ends: discretize clears the event flag past the
    horizon, and this puts the time there too.
    """
    grid = TimeGrid(CUTS, TWO_CAUSES)          # last cut 365 days
    got = event_times_days(
        np.array([0.0, 10.0, 365.0, 900.0]) * MINUTES_PER_DAY, grid)
    assert got == pytest.approx([0.0, 10.0, 365.0, 365.0])


def test_censoring_km_matches_a_hand_run():
    """Five episodes, censorings at day 10 and two at day 100.

    Day 10:  1 of the 5 still at risk is censored  -> G = 0.8
    Day 100: 2 of the 3 still at risk are censored -> G = 0.8 * (1/3)

    The episode whose outcome is observed at day 100 counts toward the
    three at risk there: a tie between an event and a censoring is
    resolved in favour of still being under observation.
    """
    times = np.array([10.0, 50.0, 100.0, 100.0, 200.0])
    is_event = np.array([False, True, False, False, True])
    jumps, surv = censoring_survival(times, is_event)
    assert jumps == pytest.approx([10.0, 100.0])
    assert surv == pytest.approx([0.8, 0.8 / 3.0])


def test_censoring_km_is_flat_without_censoring():
    times = np.array([10.0, 50.0, 100.0])
    jumps, surv = censoring_survival(times, np.ones(3, dtype=bool))
    assert jumps.size == 0 and surv.size == 0


def test_brier_is_zero_for_a_perfect_step_prediction():
    """Everyone fails of cause 0 on day 10 and the model says exactly that.

    Perfect here means the step: no incidence before day 10, certainty
    from day 10 on. A model certain of readmission at discharge is not
    right about a readmission that happens later -- see the next test.
    """
    times = np.full(6, 10.0)
    taus = np.array([5.0, 10.0, 30.0])
    cif_k = np.tile([0.0, 1.0, 1.0], (6, 1))

    b = cause_specific_brier(cif_k, times, np.zeros(6, dtype=int),
                             np.ones(6, dtype=bool), 0, taus)
    assert b == pytest.approx(np.zeros(3))


def test_brier_penalises_incidence_claimed_before_it_happens():
    """Certain of failure from day 0, when everyone fails on day 10."""
    times = np.full(6, 10.0)
    taus = np.array([5.0, 10.0])
    cif_k = np.ones((6, 2))

    b = cause_specific_brier(cif_k, times, np.zeros(6, dtype=int),
                             np.ones(6, dtype=bool), 0, taus)
    # At day 5 every episode is still event-free and the model says the
    # opposite, as wrong as it can be; by day 10 it is right.
    assert b == pytest.approx([1.0, 0.0])


def test_brier_counts_the_other_cause_as_a_non_event():
    """A patient who died has not been readmitted, and still counts."""
    times = np.array([10.0, 10.0])
    taus = np.array([30.0])
    cif_k = np.ones((2, 1))                  # both predicted certain to be
    cause_index = np.array([0, 1])           # readmitted, but one died

    b = cause_specific_brier(cif_k, times, cause_index,
                             np.array([True, True]), 0, taus)
    # One residual of 0 and one of 1, equally weighted.
    assert b[0] == pytest.approx(0.5)


def test_brier_weights_by_the_inverse_censoring_probability():
    """One hand-run horizon, worked out term by term.

    Four episodes: censored at day 10, cause 0 at day 20, cause 1 at day
    40, still at risk past day 100. The single censoring at day 10 leaves
    G = 3/4 from there on, so every weight below is 4/3.

    At tau = 50 the contributions are
        censored at 10: weight 0, out of the sum
        cause 0 at 20:  target 1, (0.80 - 1)^2 = 0.0400
        cause 1 at 40:  target 0, (0.25 - 0)^2 = 0.0625
        at risk at 50:  target 0, (0.50 - 0)^2 = 0.2500
    which is (4/3) * 0.3525 = 0.47, over the four episodes: 0.1175.
    """
    times = np.array([10.0, 20.0, 40.0, 100.0])
    is_event = np.array([False, True, True, True])
    cause_index = np.array([-1, 0, 1, 0])
    taus = np.array([50.0])
    cif_k = np.array([[0.9], [0.8], [0.25], [0.5]])

    b = cause_specific_brier(cif_k, times, cause_index, is_event, 0, taus)
    assert b[0] == pytest.approx(0.1175)


def test_an_episode_censored_before_the_horizon_contributes_nothing():
    """Its own prediction cannot move the score, however wrong it is."""
    times = np.array([10.0, 20.0, 100.0])
    is_event = np.array([False, True, True])
    cause_index = np.array([-1, 0, 0])
    taus = np.array([50.0])

    mild = np.array([[0.5], [0.8], [0.5]])
    wild = np.array([[0.0], [0.8], [0.5]])
    assert (cause_specific_brier(mild, times, cause_index, is_event, 0, taus)
            == pytest.approx(cause_specific_brier(wild, times, cause_index,
                                                  is_event, 0, taus)))


def test_the_event_weight_is_the_left_limit_of_g():
    """An episode failing at the instant of a censoring was still observed.

    Three episodes: one censored on day 10, one failing of cause 0 on day
    10, one at risk past day 100. G drops to 2/3 *at* day 10, so reading
    it there rather than just before it would weight the failure 1.5
    instead of 1.

    With the model predicting no incidence at all, the failure's residual
    is 1 and the episode at risk contributes nothing, so the score is its
    weight over three episodes: 1/3 for the left limit, 0.5 for the other
    reading.
    """
    times = np.array([10.0, 10.0, 100.0])
    is_event = np.array([False, True, True])
    cause_index = np.array([-1, 0, 0])
    taus = np.array([50.0])

    b = cause_specific_brier(np.zeros((3, 1)), times, cause_index,
                             is_event, 0, taus)
    assert b[0] == pytest.approx(1.0 / 3.0)


def test_integrated_brier_is_a_trapezoid_mean():
    """Rising from 0 to 1 over the first day, flat after: area 1.5 of 2."""
    taus = np.array([0.0, 1.0, 2.0])
    assert integrated_brier(np.array([0.0, 1.0, 1.0]), taus) == pytest.approx(
        0.75)


def test_integrated_brier_of_a_constant_is_that_constant():
    taus = np.linspace(0.0, 365.0, 366)
    assert integrated_brier(np.full(366, 0.2), taus) == pytest.approx(0.2)


# --------------------------------------------------- the interpolated curve

def test_the_interpolated_curve_passes_through_the_cuts():
    """An interpolant of the head's output, not a smoothing of it."""
    grid = TimeGrid(CUTS, TWO_CAUSES)
    cif_k = np.array([[0.1, 0.3, 0.6], [0.0, 0.0, 0.9]])
    assert interpolate_cif(cif_k, grid, grid.cuts) == pytest.approx(cif_k)


def test_the_interpolated_curve_is_anchored_at_zero():
    """Nobody has been readmitted at the moment they are discharged."""
    grid = TimeGrid(CUTS, TWO_CAUSES)
    cif_k = np.array([[0.1, 0.3, 0.6]])
    assert interpolate_cif(cif_k, grid, np.array([0.0]))[0, 0] == (
        pytest.approx(0.0))


def test_the_interpolated_curve_never_turns_back():
    """What the monotone filter is there for: incidence only accumulates."""
    grid = TimeGrid(CUTS, TWO_CAUSES)
    rng = np.random.default_rng(11)
    cif_k = np.sort(rng.random((50, 3)), axis=1) * 0.9
    curve = interpolate_cif(cif_k, grid, np.linspace(0.0, 365.0, 400))
    assert np.all(np.diff(curve, axis=1) >= -1e-12)
    assert curve.min() >= 0.0 and curve.max() <= 1.0


def test_interpolation_survives_a_float32_cumulative_sum():
    """A saturated head leaves the cumsum a hair short of monotone."""
    grid = TimeGrid(CUTS, TWO_CAUSES)
    pmf = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    cif_k = np.cumsum(pmf, axis=1).astype(np.float64)
    cif_k[0, 2] -= 1e-16                      # below the previous value
    curve = interpolate_cif(cif_k, grid, np.array([0.0, 15.0, 365.0]))
    assert np.all(np.diff(curve, axis=1) >= -1e-12)


# -------------------------------------------------------------- the reporter

def test_survival_metrics_reports_one_entry_per_cut_and_cause():
    grid = TimeGrid(CUTS, TWO_CAUSES)
    rng = np.random.default_rng(11)
    n = 300
    logits = rng.normal(size=(n, grid.n_causes * grid.n_bins))
    event_type = rng.choice([CENSORED, READMISSION, DEATH, OUT_MIGRATION],
                            size=n, p=[0.5, 0.25, 0.2, 0.05])
    tte = rng.uniform(1, 500, n) * MINUTES_PER_DAY

    m = survival_metrics(logits, tte, event_type, grid)

    for name in ("Readmission", "Death"):
        assert np.isfinite(m[f"{name}_Cindex"])
        assert np.isfinite(m[f"{name}_Integrated_Brier"])
        for cut in ("30d", "90d", "1y"):
            assert f"{name}_Cindex_at_{cut}" in m
    assert m["Readmission_Events"] + m["Death_Events"] + m["Censored"] == n
    assert np.isfinite(m["Mean_Cindex"])
    assert np.isfinite(m["Loss_DeepHit_NLL"])


def test_reported_nll_matches_the_loss_module():
    """The numpy reporter and the torch objective must agree."""
    grid = TimeGrid(CUTS, TWO_CAUSES)
    rng = np.random.default_rng(5)
    n = 64
    logits = rng.normal(size=(n, grid.n_causes * grid.n_bins))
    event_type = rng.choice([CENSORED, READMISSION, DEATH], size=n)
    tte = rng.uniform(1, 500, n) * MINUTES_PER_DAY

    reported = survival_metrics(logits, tte, event_type,
                               grid)["Loss_DeepHit_NLL"]

    b, c, ev = grid.discretize(torch.as_tensor(tte),
                               torch.as_tensor(event_type))
    _, (nll, _) = DeepHitLoss(grid.n_causes, grid.n_bins, rank_weight=0.0)(
        torch.as_tensor(logits, dtype=torch.float32), b, c, ev)
    assert reported == pytest.approx(float(nll), abs=1e-5)


def test_metrics_survive_a_cause_with_no_events():
    """A fold can lack deaths inside the horizon; that is NaN, not a crash."""
    grid = TimeGrid(CUTS, TWO_CAUSES)
    n = 40
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(n, grid.n_causes * grid.n_bins))
    event_type = np.where(np.arange(n) % 2 == 0, READMISSION, CENSORED)
    tte = np.tile([10.0, 200.0], n // 2) * MINUTES_PER_DAY

    m = survival_metrics(logits, tte, event_type, grid)
    assert m["Death_Events"] == 0
    assert math.isnan(m["Death_Cindex"])
    assert np.isfinite(m["Readmission_Cindex"])
    # One NaN cause must not poison the number early stopping reads.
    assert m["Mean_Cindex"] == pytest.approx(m["Readmission_Cindex"])


# ------------------------------------------------------------- the cause set

def test_the_default_models_out_migration_as_a_cause():
    """It is a competing risk, not censoring: a departure takes its own
    share of the probability mass rather than removing the patient."""

    grid = TimeGrid(CUTS)
    assert grid.cause_names == DEFAULT_CAUSES
    assert grid.n_causes == 3
    assert "out_migration" in grid.cause_names


def test_a_cause_left_out_of_the_config_is_censoring():
    """This is what makes the two-cause arm a config line."""

    two = TimeGrid(CUTS, ["readmission", "death"])
    three = TimeGrid(CUTS, ["readmission", "death", "out_migration"])
    t = minutes([10.0])
    e = torch.tensor([OUT_MIGRATION])

    _, c2, ev2 = two.discretize(t, e)
    _, c3, ev3 = three.discretize(t, e)

    assert c2.tolist() == [-1] and ev2.tolist() == [False]
    assert c3.tolist() == [2] and ev3.tolist() == [True]


def test_cause_order_is_the_output_order():
    grid = TimeGrid(CUTS, ["death", "readmission"])
    t = minutes([10.0, 10.0])
    _, c, _ = grid.discretize(t, torch.tensor([READMISSION, DEATH]))
    assert c.tolist() == [1, 0]


@pytest.mark.parametrize("causes", [
    ["readmission", "nonsense"],
    ["readmission", "readmission"],
    [],
    ["censored"],
])
def test_bad_cause_sets_are_rejected(causes):
    with pytest.raises(ValueError):
        TimeGrid(CUTS, causes)


def test_every_modelled_cause_is_a_real_event_type():
    assert set(CAUSE_CODES.values()) == {READMISSION, DEATH, OUT_MIGRATION}
    assert CENSORED not in CAUSE_CODES.values(), (
        "administrative censoring is the absence of a cause, not one of them"
    )


def test_metrics_are_reported_per_modelled_cause():
    grid = TimeGrid(CUTS)
    rng = np.random.default_rng(7)
    n = 300
    logits = rng.normal(size=(n, grid.n_causes * grid.n_bins))
    event_type = rng.choice([CENSORED, READMISSION, DEATH, OUT_MIGRATION],
                            size=n, p=[0.4, 0.25, 0.2, 0.15])
    tte = rng.uniform(1, 500, n) * MINUTES_PER_DAY

    m = survival_metrics(logits, tte, event_type, grid)

    for name in ("Readmission", "Death", "Out_Migration"):
        assert f"{name}_Cindex" in m
        assert f"{name}_Integrated_Brier" in m
        assert f"{name}_Events" in m
    total = sum(m[f"{n}_Events"] for n in
                ("Readmission", "Death", "Out_Migration"))
    assert total + m["Censored"] == n


def test_head_width_follows_the_cause_count():
    """The head is n_causes * n_bins wide, so the two must agree."""

    for causes, expected in ((["readmission"], 1),
                             (["readmission", "death"], 2),
                             (list(DEFAULT_CAUSES), 3)):
        grid = TimeGrid(CUTS, causes)
        assert grid.n_causes == expected
        logits = torch.randn(4, grid.n_causes * grid.n_bins)
        pmf, p_survive = deephit_distribution(logits, grid.n_causes,
                                              grid.n_bins)
        assert pmf.shape == (4, expected, grid.n_bins)
        assert torch.allclose(pmf.sum(dim=(1, 2)) + p_survive,
                              torch.ones(4), atol=1e-6)
