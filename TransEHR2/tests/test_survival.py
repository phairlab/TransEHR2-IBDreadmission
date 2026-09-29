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
    cause_specific_brier,
    cause_specific_concordance,
    censoring_survival,
    cif_from_pmf,
    integrated_brier,
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

def test_censoring_km_matches_a_hand_run():
    """Five episodes: censorings in bins 0 and 2, events in 1 and 2.

    Bin 0: 1 of 5 censored          -> G = 0.8
    Bin 1: 0 of 4 censored          -> G = 0.8
    Bin 2: 1 of 3 censored          -> G = 0.8 * (2/3)
    """
    bin_index = np.array([0, 1, 2, 2, 2])
    is_event = np.array([False, True, False, True, True])
    g = censoring_survival(bin_index, is_event, 3)
    assert g == pytest.approx([0.8, 0.8, 0.8 * 2 / 3])


def test_censoring_km_is_flat_without_censoring():
    bin_index = np.array([0, 1, 2])
    is_event = np.ones(3, dtype=bool)
    assert censoring_survival(bin_index, is_event, 3) == pytest.approx(
        [1.0, 1.0, 1.0])


def test_brier_is_zero_for_a_perfect_uncensored_prediction():
    """Everyone fails of cause 0 in bin 0 and the model says so."""
    n, n_bins = 6, 3
    cif = np.ones((n, 2, n_bins))
    cif[:, 1, :] = 0.0
    bin_index = np.zeros(n, dtype=int)
    cause_index = np.zeros(n, dtype=int)
    is_event = np.ones(n, dtype=bool)

    b = cause_specific_brier(cif, bin_index, cause_index, is_event, 0)
    assert b == pytest.approx(np.zeros(n_bins))


def test_brier_counts_the_other_cause_as_a_non_event():
    """A patient who died has not been readmitted, and still counts."""
    n_bins = 2
    cif = np.zeros((2, 2, n_bins))
    cif[:, 0, :] = 1.0                       # both predicted certain to be
    bin_index = np.array([0, 0])             # readmitted at once
    cause_index = np.array([0, 1])           # but one died instead
    is_event = np.array([True, True])

    b = cause_specific_brier(cif, bin_index, cause_index, is_event, 0)
    # One residual of 0 and one of 1, equally weighted.
    assert b[0] == pytest.approx(0.5)


def test_integrated_brier_weights_by_bin_width():
    """A wide bin must count for more than a narrow one."""
    grid = TimeGrid([30.0, 90.0, 365.0])     # widths 30, 60, 275
    brier = np.array([0.0, 0.0, 1.0])
    assert integrated_brier(brier, grid) == pytest.approx(275.0 / 365.0)


def test_integrated_brier_skips_empty_bins():
    grid = TimeGrid([30.0, 90.0, 365.0])
    brier = np.array([0.2, np.nan, 0.2])
    assert integrated_brier(brier, grid) == pytest.approx(0.2)


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
