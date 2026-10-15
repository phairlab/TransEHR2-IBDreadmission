"""Discrete-time competing risks: the grid, the CIF, and the metrics.

The two outcomes this study models -- unplanned readmission and death --
compete. A patient who dies cannot be readmitted, so a model that predicts
each in isolation double-counts the population at risk and reports a
readmission probability no cohort can realize. DeepHit handles this by
predicting one joint distribution over (cause, time bin) per patient, from
which each cause's cumulative incidence follows directly.

Readings this module commits to
-------------------------------

* **Out-migration is a competing cause, not censoring** (decided 2026
  September 28). Leaving the province is a period of missing data rather
  than the end of the record, and the record is already incomplete -- it
  holds no trace of most of a patient's contacts with the health system.
  Treating a departure as censoring would assume it says nothing about
  readmission risk; modelling it as a cause lets it take its own share of
  the probability mass, and each cause's incidence is then read as "before
  any of the others". Which causes a run models is
  ``MODELLED_CAUSES`` in the experiment config, and anything left out of it
  falls back to censoring -- so the two-cause arm costs a config line.
* **Bins are left-closed, ``[lo, hi)``.** A time falling exactly on a cut
  belongs to the bin the cut opens. The choice only matters for an event at
  exactly 43,200 minutes, but it has to be stated somewhere or the report
  and the model will eventually disagree about one episode.
* **Beyond the horizon is censored at the horizon.** The grid ends at its
  last cut. An episode whose event falls past that is not an event this
  model predicts; it is an episode observed through the whole grid without
  one. Dropping those episodes instead would throw away the follow-up they
  do contribute.
* **Between the cuts, the cumulative incidence is a monotone cubic.**
  The head predicts at six times and says nothing between them, but the
  Brier score is integrated over a continuous axis, and reading the curve
  as a step function would hand three quarters of a one-year integral to
  whatever the model said at exactly 365 days. The gaps are filled by
  ``splinefun(method = "hyman")`` -- the smoothing MTLR draws its survival
  curves with, ported in :mod:`TransEHR2.hyman` -- anchored at
  ``F_k(0) = 0``. This adds no information: the head still emits six
  numbers per cause, and the curve between them is an assumption about
  smoothness. It does make the assumption part of the predictor, so
  scoring and reporting both read the same curve.
* **The Brier score is integrated from discharge to a stated bound**,
  ``TimeGrid.brier_integration_days``, one year by default, and the
  interpolation is what frees that bound from having to land on a cut.
* **Times arrive in minutes and are held in days.** ``TIME_TO_EVENT`` is
  minutes, because that is the resolution the source records have. Every
  cut, every axis and every printed number here is in days, because that is
  the unit the grid is argued in.
"""

from __future__ import annotations

import numpy as np
import torch

from scipy.stats import chi2
from torch import Tensor
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from TransEHR2.hyman import hyman_spline


MINUTES_PER_DAY = 1440.0

# build_labels.py's EVENT_TYPE coding.
CENSORED = 0
READMISSION = 1
DEATH = 2
OUT_MIGRATION = 3

# Every EVENT_TYPE that can be modelled as a cause, by the name the config
# uses for it. EVENT_TYPE 0 is not here: administrative censoring is the
# absence of a cause, not one of them.
CAUSE_CODES: Dict[str, int] = {
    "readmission": READMISSION,
    "death": DEATH,
    "out_migration": OUT_MIGRATION,
}

# The causes a run models, in output order, unless MODELLED_CAUSES says
# otherwise. Out-migration is one of them: leaving the province is a period
# of missing data rather than the end of the record, and treating it as
# censoring would assume it uninformative about readmission risk. As a
# competing cause it takes its own share of the probability mass, and each
# cause's incidence is then read as "before any of the others".
#
# A cause left out of this tuple is censoring, which is what makes the
# two-cause arm a config change rather than a code change.
DEFAULT_CAUSES: Tuple[str, ...] = ("readmission", "death", "out_migration")

# The default grid, in days: 30, 60 and 90 days, then 1, 3 and 5 years.
# Overridden by TIME_GRID_CUTS_DAYS in the experiment config.
DEFAULT_CUTS_DAYS: Tuple[float, ...] = (30.0, 60.0, 90.0, 365.0, 1095.0,
                                        1825.0)

# How far the cause-specific Brier score is integrated, in days, unless
# BRIER_INTEGRATION_DAYS says otherwise. One year, because the model is
# meant to be re-run whenever the patient sees a physician and a patient is
# expected to see one at least annually, which makes the year the window
# any single prediction has to cover. The lower bound is always zero.
DEFAULT_BRIER_INTEGRATION_DAYS = 365.0

# Competing-risks D-calibration (Alberge et al., AISTATS 2026), unless the
# experiment config's CR_DCAL_* keys say otherwise. alpha = 2 makes the
# distance Euclidean. The paper leaves the Riemann step count M unstated;
# at 1000 the step is finer than any bucket a cohort this size can resolve.
# Ten bins is Haider et al.'s choice for the chi-squared test.
DEFAULT_CR_DCAL_ALPHA = 2.0
DEFAULT_CR_DCAL_N_RHO = 1000
DEFAULT_CR_DCAL_CHI2_BINS = 10
DEFAULT_CR_DCAL_TEST_LEVEL = 0.05

# The floor under a recalibrated incidence at the horizon and under the
# recalibrated probability of surviving it, as censoring_survival floors G:
# a zero would divide the D-calibration buckets by nothing.
RECAL_FLOOR = 1e-8


class TimeGrid:
    """The discrete time axis a DeepHit head predicts on.

    ``cuts`` are the right edges of the bins, in days, ascending. The bins
    are ``[0, cuts[0])``, ``[cuts[0], cuts[1])``, ..., ``[cuts[-2],
    cuts[-1])``, so there are as many bins as cuts and the last cut is the
    horizon rather than the start of an open bin.
    """

    def __init__(
        self,
        cuts_days: Sequence[float] = DEFAULT_CUTS_DAYS,
        causes: Sequence[str] = DEFAULT_CAUSES,
        brier_integration_days: float = DEFAULT_BRIER_INTEGRATION_DAYS,
    ):
        unknown = [name for name in causes if name not in CAUSE_CODES]
        if unknown:
            raise ValueError(f"unknown cause(s) {unknown}; known: "
                             f"{sorted(CAUSE_CODES)}")
        if len(set(causes)) != len(causes):
            raise ValueError(f"causes must be distinct, got {list(causes)}")
        if not causes:
            raise ValueError("at least one cause must be modelled")
        self.cause_names: Tuple[str, ...] = tuple(causes)
        self.causes: Tuple[int, ...] = tuple(CAUSE_CODES[n] for n in causes)

        cuts = np.asarray(cuts_days, dtype=np.float64)
        if cuts.ndim != 1 or cuts.size == 0:
            raise ValueError("cuts_days must be a non-empty 1-D sequence")
        if np.any(cuts <= 0):
            raise ValueError(f"cuts_days must all be positive, got {cuts_days}")
        if np.any(np.diff(cuts) <= 0):
            raise ValueError(f"cuts_days must be strictly ascending, got "
                             f"{cuts_days}")
        self.cuts = cuts

        # The Brier integration bound rides on the grid rather than being
        # passed beside it: every scoring call already receives the grid,
        # and threading one more number through five signatures to reach
        # the same places would only make it easier to leave one behind.
        if not np.isfinite(brier_integration_days) or \
                brier_integration_days <= 0:
            raise ValueError(f"brier_integration_days must be positive, got "
                             f"{brier_integration_days}")
        if brier_integration_days > float(cuts[-1]):
            raise ValueError(
                f"brier_integration_days is {brier_integration_days} but the "
                f"grid ends at {float(cuts[-1])}; the score cannot be "
                f"integrated past the last cut, where the model stops "
                f"predicting.")
        self.brier_integration_days = float(brier_integration_days)

    @property
    def n_bins(self) -> int:
        return int(self.cuts.size)

    @property
    def n_causes(self) -> int:
        return len(self.causes)

    @property
    def horizon_days(self) -> float:
        return float(self.cuts[-1])

    @property
    def edges_days(self) -> np.ndarray:
        """Bin edges including the implicit zero, for histograms."""
        return np.concatenate([[0.0], self.cuts])

    def labels(self) -> List[str]:
        lo = np.concatenate([[0.0], self.cuts[:-1]])
        return [f"[{format_days(a)}, {format_days(b)})" for a, b in zip(lo, self.cuts)]

    def bin_of(self, days: Tensor) -> Tensor:
        """Which bin each time falls in, clamped to the last bin.

        Clamping is what makes an event past the horizon a censored
        observation at the horizon rather than an index error: the caller
        pairs this with :meth:`discretize`, which also clears the event
        flag for those rows.
        """
        cuts = torch.as_tensor(self.cuts, dtype=days.dtype,
                               device=days.device)
        idx = torch.searchsorted(cuts, days.contiguous(), right=True)
        return idx.clamp_(max=self.n_bins - 1).long()

    def discretize(
        self,
        time_to_event: Tensor,
        event_type: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Turn labels into the three tensors the DeepHit loss needs.

        Args:
            time_to_event: minutes from INDEX_TIME, shape (batch,) or
                (batch, 1).
            event_type: build_labels.py's EVENT_TYPE, same shape.

        Returns:
            bin_index: (batch,) long, which bin the episode ends in.
            cause_index: (batch,) long, position in the grid's own
                ``causes``, or -1 for a censored episode -- which includes
                any EVENT_TYPE this run does not model.
            is_event: (batch,) bool, True where an outcome was observed
                inside the grid.
        """
        t = time_to_event.reshape(-1).to(torch.float64) / MINUTES_PER_DAY
        e = event_type.reshape(-1).long()

        within = t < self.horizon_days
        bin_index = self.bin_of(t)

        cause_index = torch.full_like(e, -1)
        for k, code in enumerate(self.causes):
            cause_index = torch.where(e == code,
                                      torch.full_like(e, k), cause_index)

        # Past the horizon there is no outcome to predict, and EVENT_TYPE 0
        # and 3 have none to begin with.
        is_event = (cause_index >= 0) & within
        cause_index = torch.where(is_event, cause_index,
                                  torch.full_like(cause_index, -1))
        return bin_index, cause_index, is_event


def metric_label(cause_name: str) -> str:
    """The metric-key form of a cause name: ``out_migration`` -> ``Out_Migration``."""
    return "_".join(part.capitalize() for part in cause_name.split("_"))


def format_days(d: float) -> str:
    if d == 0:
        return "0"
    if d < 365:
        return f"{d:g} d"
    y = d / 365.0
    return f"{y:g} y" if abs(y - round(y)) < 0.02 else f"{y:.1f} y"


# --------------------------------------------------------------- the CIF

def deephit_distribution(
    logits: Tensor, n_causes: int, n_bins: int
) -> Tuple[Tensor, Tensor]:
    """The full simplex a DeepHit head defines: ``(pmf, p_survive)``.

    One softmax over the whole ``n_causes * n_bins`` space, not one per
    cause: the mass a patient does not spend on any (cause, bin) is their
    probability of surviving the grid, and that only has a meaning if the
    causes share a simplex.

    That simplex needs one more slot than the head has outputs. A softmax
    over ``n_causes * n_bins`` alone sums to exactly 1, which makes
    surviving the whole grid a probability-zero event and hands an infinite
    loss to every episode censored in the last bin. A constant zero logit
    is appended to carry the leftover mass; softmax is shift-invariant, so
    fixing it at zero costs no expressiveness and keeps the head's output
    width at ``n_causes * n_bins``.

    Args:
        logits: (batch, n_causes * n_bins).

    Returns:
        ``pmf`` of shape (batch, n_causes, n_bins), and ``p_survive`` of
        shape (batch,) holding the leftover. The two together sum to 1.
    """
    flat = logits.reshape(logits.shape[0], -1)
    padded = torch.cat([flat, flat.new_zeros(flat.shape[0], 1)], dim=1)
    full = torch.softmax(padded, dim=1)
    pmf = full[:, :-1].reshape(-1, n_causes, n_bins)
    return pmf, full[:, -1]


def pmf_from_logits(logits: Tensor, n_causes: int, n_bins: int) -> Tensor:
    """Just the (cause, bin) block of :func:`deephit_distribution`."""
    return deephit_distribution(logits, n_causes, n_bins)[0]


def cif_from_pmf(pmf: Tensor) -> Tensor:
    """Cause-specific cumulative incidence, ``F_k(t) = sum_{m<=t} y_{k,m}``.

    Shape in and out is (batch, n_causes, n_bins).
    """
    return pmf.cumsum(dim=-1)


def survival_from_pmf(pmf: Tensor, p_survive: Tensor) -> Tensor:
    """Overall survival past each bin: (batch, n_bins).

    Summed forward from the tail rather than taken as ``1 - sum_{m<=t}``.
    The two are the same algebraically, but the subtraction cancels away
    the whole answer once the head saturates -- a confident model leaves a
    leftover near 1e-14, which is exactly the number needed and exactly the
    number that vanishes when it is computed as one minus its complement.

    The mass after each bin is the inclusive tail of the *next* bin, not the
    inclusive tail of this one minus this bin: that subtraction cancels the
    same way whenever one bin holds nearly all of a patient's mass. In
    float32, the precision the loss is trained in, a true survival of 3e-8
    behind a saturated bin came out as 6e-8, and 1e-10 as nothing at all.
    """
    total = pmf.sum(dim=1)                                  # (batch, n_bins)
    tail_inclusive = total.flip(-1).cumsum(-1).flip(-1)
    tail_exclusive = torch.cat(
        [tail_inclusive[:, 1:], torch.zeros_like(tail_inclusive[:, :1])],
        dim=-1)
    return tail_exclusive + p_survive.unsqueeze(-1)


def cif_logit_target(
    cause: int, horizon_bin: int, n_causes: int, n_bins: int
) -> Callable[[Tensor], Tensor]:
    """A scalar for attribution: the log-odds that cause ``k`` happens by a horizon.

    Gradient attribution needs one scalar per backward, and the training
    objective is the wrong one to pick. DeepHit's loss is a likelihood plus
    a pairwise ranking penalty, so a gradient of it answers "what moved this
    patient's contribution to the objective", which is a question about the
    fit and not about the patient. The cumulative incidence of unplanned
    readmission by a stated horizon is the quantity a reader cares about,
    and it is monotone in the risk.

    The log-odds rather than the incidence itself, for the usual reason a
    saliency target is taken before the squashing function: ``F_k`` is a
    probability, so once the head is confident its gradient goes to zero in
    every direction at once and the map reports numerical floor rather than
    the inputs. ``log(F / (1 - F))`` is unbounded and monotone in ``F``, so
    it ranks patients and inputs identically without saturating. The odds is
    also the one a reader already has a denominator for: ``1 - F_k(t)`` is
    not surviving the grid, it is "not readmitted by t", counting the
    patients who died or left first.

    There is no single head logit that would do instead. The head emits one
    per ``(cause, bin)`` cell, and an incidence by a horizon is a sum over
    bins taken after a softmax the causes share -- a raw cell logit can move
    without ``F_k`` moving at all.

    Computed as a difference of log-sum-exps rather than from the pmf. The
    cells inside the horizon and the cells outside it partition the simplex
    `deephit_distribution` defines, and the shared normalizer cancels in the
    ratio, so ``log(F / (1 - F))`` is exactly
    ``logsumexp(inside) - logsumexp(outside)``. Taking it that way matters
    rather than being a tidiness: forming ``F`` first and clamping it off 1
    to keep the logarithm finite would hand back a gradient of exactly zero
    at saturation, which is the failure the log-odds was chosen to avoid.

    Args:
        cause (int): Position of the cause in the head's output order, i.e.
            its index in ``TimeGrid.cause_names``, not its EVENT_TYPE code.
        horizon_bin (int): Bin whose right edge is the horizon, 0-based.
        n_causes (int): Number of competing causes the head predicts.
        n_bins (int): Number of bins on the discrete time grid.

    Returns:
        A callable taking the head's ``(batch, n_causes * n_bins)`` logits
        and returning the summed log-odds over the batch. Summing is what
        `tsr.saliency` wants: no item's output depends on another's inputs,
        so one backward carries every item's gradient unmixed.
    """
    if not 0 <= cause < n_causes:
        raise ValueError(f'cause {cause} outside [0, {n_causes})')
    if not 0 <= horizon_bin < n_bins:
        raise ValueError(f'horizon_bin {horizon_bin} outside [0, {n_bins})')

    # The layout `deephit_distribution` softmaxes over: cause-major, flat,
    # with the survive-the-horizon slot appended.
    inside = torch.zeros(n_causes * n_bins + 1, dtype=torch.bool)
    inside[cause * n_bins:cause * n_bins + horizon_bin + 1] = True

    def target(logits: Tensor) -> Tensor:
        flat = logits.reshape(logits.shape[0], -1)
        padded = torch.cat([flat, flat.new_zeros(flat.shape[0], 1)], dim=1)
        return (torch.logsumexp(padded[:, inside], dim=1)
                - torch.logsumexp(padded[:, ~inside], dim=1)).sum()

    return target


# --------------------------------------------------------------- metrics

def cause_specific_concordance(
    cif: np.ndarray,
    bin_index: np.ndarray,
    cause_index: np.ndarray,
    is_event: np.ndarray,
    k: int,
    horizon_bin: Optional[int] = None,
) -> float:
    """Antolini's time-dependent concordance for one cause.

    A pair ``(i, j)`` is comparable when ``i`` failed of cause ``k`` in bin
    ``t`` and ``j`` was still event-free after ``t`` -- censored later or
    failing later, of any cause. A ``j`` censored before ``t`` says nothing
    about the order and is excluded, which is what lets a rank measure work
    here without inverse-probability weights. The pair is concordant when
    ``i`` carried the higher ``F_k(t)``, both read at the *case's* time.
    Ties count a half, as in Harrell's C.

    The pair set is never materialized: cases are grouped by their bin, and
    within a bin every case faces the same control set, so one sort and a
    binary search per bin replaces an ``n_cases x n`` comparison matrix.

    Returns NaN when no comparable pair exists, which is the honest answer
    for a cause with no events inside the horizon.
    """
    n_bins = cif.shape[-1]
    last = n_bins - 1 if horizon_bin is None else horizon_bin
    concordant = 0.0
    n_pairs = 0

    for t in range(min(last, n_bins - 1) + 1):
        cases = np.flatnonzero(is_event & (cause_index == k)
                               & (bin_index == t))
        if cases.size == 0:
            continue
        controls = np.flatnonzero(bin_index > t)
        if controls.size == 0:
            continue

        ctrl = np.sort(cif[controls, k, t])
        risk = cif[cases, k, t]
        # Controls strictly below the case, plus half of those level with it.
        below = np.searchsorted(ctrl, risk, side="left")
        at_or_below = np.searchsorted(ctrl, risk, side="right")
        concordant += float(np.sum(below) + 0.5 * np.sum(at_or_below - below))
        n_pairs += cases.size * controls.size

    return concordant / n_pairs if n_pairs else float("nan")


def event_times_days(time_to_event: np.ndarray,
                    grid: TimeGrid) -> np.ndarray:
    """Event times in days, truncated at the horizon.

    The same reading :meth:`TimeGrid.discretize` applies, in continuous
    time: an episode whose event falls past the last cut is one observed
    through the whole grid without an outcome this model predicts, so its
    time is the horizon and :meth:`~TimeGrid.discretize` has already
    cleared its event flag. Keeping the rule in one function is what stops
    the binned and continuous readings of the same episode from drifting
    apart.
    """
    days = np.asarray(time_to_event, dtype=np.float64) / MINUTES_PER_DAY
    return np.minimum(days, grid.horizon_days)


def brier_times(grid: TimeGrid) -> np.ndarray:
    """The axis the Brier score is integrated over: 0 to the bound, daily.

    One point per day. The outcome times are recorded to the minute but
    argued in days, and a day is already finer than any distinction this
    cohort supports; a denser axis would cost time without moving the
    integral. Zero is on the axis because ``F_k(0) = 0`` for every patient
    and nobody has failed yet, so the score there is exactly zero -- a
    real point of the curve rather than an extrapolation.
    """
    n_steps = max(int(round(grid.brier_integration_days)), 1)
    return np.linspace(0.0, grid.brier_integration_days, n_steps + 1)


def interpolate_cif(
    cif_k: np.ndarray, grid: TimeGrid, times: np.ndarray,
    per_row: bool = False
) -> np.ndarray:
    """One cause's cumulative incidence, read at arbitrary times.

    The head predicts ``F_k`` at the cuts and nowhere else. Between them
    the curve is filled in by a monotone cubic with Hyman filtering --
    ``splinefun(method = "hyman")``, the smoothing MTLR draws its survival
    curves with -- anchored at the one point that needs no model:
    ``F_k(0) = 0``, because no patient has failed at the moment they are
    discharged. Without that anchor the spline has nothing to say below
    the first cut and would extrapolate a cubic fitted to the four cuts
    above it across the whole of the first bin.

    This interpolation is part of the predictor, not part of the scoring.
    The model whose Brier score is integrated below is DeepHit *and* this
    curve, which is why the reporting script reads it through the same
    function rather than keeping the step function for itself.

    Args:
        cif_k: (n, n_bins) cumulative incidence of one cause at the cuts.
        grid: the grid those cuts come from.
        times: (p,) days at which to read the curve, within ``[0,
            horizon]``.
        per_row: read row ``i`` at ``times[i]`` alone, with ``p == n``.

    Returns:
        (n, p) cumulative incidence, or (n,) with ``per_row``.
    """
    anchored = np.concatenate(
        [np.zeros((cif_k.shape[0], 1), dtype=np.float64),
         np.asarray(cif_k, dtype=np.float64)], axis=1)
    # A cumulative sum of non-negative mass is monotone by construction,
    # but it reaches here through a float32 softmax, and the filter
    # refuses input that steps backwards by even one ulp.
    np.maximum.accumulate(anchored, axis=1, out=anchored)

    curve = hyman_spline(grid.edges_days, anchored, times, per_row=per_row)
    # Monotone interpolation between knots inside [0, 1] cannot leave it;
    # the clip is against rounding at the ends, not against the spline.
    return np.clip(curve, 0.0, 1.0)


def interpolate_tail(
    pmf_k: np.ndarray, grid: TimeGrid, times: np.ndarray,
    per_row: bool = False
) -> np.ndarray:
    """One cause's mass still to come, ``F_k(tau) - F_k(t)``, at arbitrary times.

    The same curve :func:`interpolate_cif` draws, read from the other end.
    The Hyman spline commutes with ``y -> c - y`` -- the FMM slopes are
    linear in the knots and the filter is symmetric in sign -- so splining
    the tail gives ``F_k(tau)`` minus the CIF's spline, to rounding (8e-16
    on random curves). Splining the tail directly, with its knots summed
    forward from the pmf as in :func:`survival_from_pmf`, is what keeps a
    small remainder behind a large incidence from being computed as the
    difference of two numbers near one.

    Args:
        pmf_k: (n, n_bins) one cause's mass in each bin.
        grid: the grid those bins come from.
        times: (p,) days, within ``[0, horizon]``.
        per_row: read row ``i`` at ``times[i]`` alone, with ``p == n``.

    Returns:
        (n, p) mass of cause ``k`` after each time, or (n,) with
        ``per_row``.
    """
    pmf_k = np.asarray(pmf_k, dtype=np.float64)
    # The inclusive tail of bin j is the mass after edge j: the whole of
    # F_k(tau) at t = 0, and nothing after the last cut.
    inclusive = np.flip(np.cumsum(np.flip(pmf_k, axis=1), axis=1), axis=1)
    anchored = np.concatenate(
        [inclusive, np.zeros((pmf_k.shape[0], 1), dtype=np.float64)], axis=1)
    # Non-increasing by construction; the running minimum is the mirror of
    # interpolate_cif's running maximum, against input that steps the wrong
    # way by an ulp.
    np.minimum.accumulate(anchored, axis=1, out=anchored)

    curve = hyman_spline(grid.edges_days, anchored, times, per_row=per_row)
    return np.clip(curve, 0.0, 1.0)


def censoring_survival(
    times_days: np.ndarray, is_event: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier estimate of the censoring distribution, ``G``.

    The Brier score below weights each episode by the inverse of its
    probability of still being under observation, so a cohort whose
    follow-up ends early does not look well calibrated merely because its
    late bins are empty. Censoring is the "event" in this fit.

    Fitted on the recorded times rather than on bin indices: the score is
    read at every day between zero and the bound, and a ``G`` that could
    only step at the six cuts would hold a smooth numerator over a
    six-step denominator.

    An episode whose outcome is observed at exactly a censoring time is
    counted as still at risk there, the usual convention for a tie
    between the two.

    Returns:
        ``(jump_times, surv)``: ``surv[j]`` is ``G`` at and after
        ``jump_times[j]``. ``G`` is 1 before the first.
    """
    t = np.asarray(times_days, dtype=np.float64)
    censored = ~np.asarray(is_event, dtype=bool)
    if not censored.any():
        return np.empty(0), np.empty(0)

    jump_times, counts = np.unique(t[censored], return_counts=True)
    # Everyone whose record runs to the jump time or beyond is at risk.
    at_risk = t.size - np.searchsorted(np.sort(t), jump_times, side='left')
    surv = np.cumprod(1.0 - counts / at_risk)
    # A zero here would make the weights infinite. The floor costs a little
    # bias in the last bins and buys a finite score.
    return jump_times, np.maximum(surv, 1e-8)


def _censoring_at(
    jump_times: np.ndarray, surv: np.ndarray, times: np.ndarray,
    left_limit: bool
) -> np.ndarray:
    """``G`` read off the step function, at ``times`` or just before them.

    Equation 8.7 asks for both: ``G(t_i)`` for an episode weighted at its
    own event time -- which is the left limit, the probability of being
    under observation up to the instant the outcome happened -- and
    ``G(tau)`` for one still at risk at the horizon.
    """
    if jump_times.size == 0:
        return np.ones(np.shape(times), dtype=np.float64)
    side = 'left' if left_limit else 'right'
    idx = np.searchsorted(jump_times, times, side=side) - 1
    return np.where(idx < 0, 1.0, surv[np.clip(idx, 0, None)])


def cause_specific_brier(
    cif_k: np.ndarray,
    times_days: np.ndarray,
    cause_index: np.ndarray,
    is_event: np.ndarray,
    k: int,
    times: np.ndarray,
) -> np.ndarray:
    """IPCW Brier score of ``F_k`` at each time in ``times``.

    Schoop et al.'s non-integrated cause-specific survival Brier score,
    equation 8.7, term for term. Three groups contribute at horizon
    ``tau``: those who failed of cause ``k`` by ``tau`` (target 1,
    weighted by ``G`` just before their own event time), those who failed
    of another cause by ``tau`` (target 0, same weighting), and those
    still event-free at ``tau`` (target 0, weighted by ``G(tau)``). An
    episode censored before ``tau`` contributes zero. The second group is
    what makes this a competing-risks score rather than one that pretends
    the other cause removed the patient from the cohort.

    The sum is divided by the number of episodes, as in the paper, rather
    than by the total weight. The two are closer than they look: the
    inverse-probability weights of a Kaplan-Meier censoring fit sum to
    exactly ``n`` when no event time coincides with a censoring time, and
    to within a fraction of a percent of it when some do. The paper's form
    is used because it is the paper's, not because it moves the number.

    Args:
        cif_k: (n, p) cumulative incidence of cause ``k``, already read at
            ``times`` -- see :func:`interpolate_cif`.
        times_days: (n,) days to the outcome, truncated at the horizon.
        cause_index: (n,) position in the grid's causes, -1 if censored.
        is_event: (n,) whether an outcome was observed inside the grid.
        k: which cause.
        times: (p,) the horizons ``tau``.

    Returns:
        (p,) the score at each ``tau``.
    """
    t = np.asarray(times_days, dtype=np.float64)[:, None]
    ev = np.asarray(is_event, dtype=bool)[:, None]
    failed_k = (ev & (np.asarray(cause_index)[:, None] == k))

    jump_times, surv = censoring_survival(times_days, is_event)
    g_event = _censoring_at(jump_times, surv, times_days, left_limit=True)
    g_tau = _censoring_at(jump_times, surv, times, left_limit=False)

    by_now = t <= np.asarray(times, dtype=np.float64)[None, :]
    weight = np.where(by_now, (1.0 / g_event)[:, None], 1.0 / g_tau[None, :])
    weight = np.where(by_now & ~ev, 0.0, weight)   # censored: uninformative

    residual = (cif_k - (by_now & failed_k)) ** 2
    return (weight * residual).sum(axis=0) / t.size


def integrated_brier(brier: np.ndarray, times: np.ndarray) -> float:
    """Mean Brier score over ``times``, by the trapezoid rule.

    Divided by the width of the interval rather than left as an integral,
    so the number is on the same scale as the scores it averages and two
    runs integrated to different bounds are at least comparable in
    magnitude.

    The rule is written out rather than taken from NumPy because the
    function was renamed between the versions this runs on.
    """
    ok = np.isfinite(brier)
    if ok.sum() < 2:
        return float('nan')
    y, x = np.asarray(brier)[ok], np.asarray(times, dtype=np.float64)[ok]
    span = x[-1] - x[0]
    if span <= 0:
        return float('nan')
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * np.diff(x)) / span)


def aalen_johansen(
    times_days: np.ndarray,
    cause_index: np.ndarray,
    is_event: np.ndarray,
    k: int,
    taus: np.ndarray,
) -> np.ndarray:
    r"""Marginal cumulative incidence of cause ``k``, read at ``taus``.

    The covariate-free estimator: one curve for the whole cohort, which is what a model that
    had learned nothing from its inputs could still predict. Aalen-Johansen rather than
    Kaplan-Meier, because KM applied to one cause treats the others as censoring and so answers
    "what if nobody could die" -- an incidence this cohort cannot realise, and a reference too
    easy to beat.

    .. math::
        \hat F_k(t) = \sum_{t_i \le t} \hat S(t_{i-1}) \frac{d_{k,i}}{n_i}

    ``S`` is the event-free survival over events of *any* cause, and it is evaluated just
    before each event time -- a patient must still be event-free to fail of cause ``k`` at
    ``t_i``. Censored episodes leave the risk set without contributing an increment.

    Args:
        times_days: (n,) event or censoring time of each episode, in days.
        cause_index: (n,) which modelled cause, where an event occurred.
        is_event: (n,) True where an event was observed rather than censoring.
        k: The cause to estimate.
        taus: (p,) ascending times to read the curve at.

    Returns:
        (p,) the marginal CIF, non-decreasing, starting at 0.
    """
    times_days = np.asarray(times_days, dtype=np.float64)
    is_event = np.asarray(is_event, dtype=bool)
    cause_index = np.asarray(cause_index)

    event_times = np.unique(times_days[is_event])
    if event_times.size == 0:
        # A cohort in which nothing was observed to happen. The incidence is zero everywhere
        # rather than undefined, and the step lookup below has no array to index.
        return np.zeros(np.asarray(taus).size, dtype=np.float64)

    survival, incidence = 1.0, 0.0
    curve = np.zeros(event_times.size, dtype=np.float64)

    for i, t in enumerate(event_times):
        at_risk = int(np.sum(times_days >= t))
        if at_risk == 0:
            curve[i] = incidence
            continue
        at_t = (times_days == t) & is_event
        # S is read *before* this time's events are removed, which is the whole content of the
        # t_{i-1} in the formula: the increment is the hazard of cause k among those who
        # reached t_i event-free.
        incidence += survival * (int(np.sum(at_t & (cause_index == k))) / at_risk)
        survival *= 1.0 - int(np.sum(at_t)) / at_risk
        curve[i] = incidence

    # A step function: the value at tau is the last increment at or before it, and 0 before the
    # first event. `searchsorted` with 'right' puts a tau exactly on an event time after that
    # event, which is the convention cause_specific_brier reads F at.
    index = np.searchsorted(event_times, np.asarray(taus, dtype=np.float64),
                            side='right') - 1
    return np.where(index >= 0, curve[np.clip(index, 0, None)], 0.0)


# ------------------------------------------ competing-risks D-calibration

def cr_dcal_numerator(
    cif: np.ndarray,
    pmf: np.ndarray,
    p_survive: np.ndarray,
    times_days: np.ndarray,
    cause_index: np.ndarray,
    is_event: np.ndarray,
    k: int,
    grid: TimeGrid,
    rhos: np.ndarray,
) -> Tuple[np.ndarray, float]:
    r"""The bucket counts of CR D-calibration for cause ``k``, at each ``rho``.

    Alberge et al. (AISTATS 2026, definition 10 and algorithm 1). Under a
    calibrated model, the normalized incidence at the observed time,
    ``u = F_k(t|x) / F_k(\infty|x)``, is uniform among the patients who
    fail of cause ``k``. The bucket ``[0, rho]`` counts those patients
    whose ``u`` lands in it, and a censored patient, whose cause is
    unknown, contributes what the model expects of them given that they
    were event-free at ``C``:

    .. math::
        N_k(\rho) = \#\{\delta_i = k, u_i \le \rho\}
            + \sum_{\delta_i = 0,\, u_i \le \rho}
              \frac{F_k(\infty|x_i)\rho - F_k(C_i|x_i)}{S(C_i|x_i)}

    A patient who failed of another cause contributes nothing.

    ``F_k(\infty)`` is read as ``F_k(\tau)``, the incidence at the horizon:
    the head's leftover mass, ``p_survive``, belongs to no cause, so the
    incidence past the grid is not identified. An event past the horizon
    is already censored at it, so this is D-calibration of the
    distribution given an event by ``tau`` -- and a patient censored there
    has ``u = 1`` and a numerator of exactly zero.

    ``S(C)`` is ``p_survive`` plus every cause's mass after ``C``, read
    through :func:`interpolate_tail`, never ``1 - sum_k F_k(C)``: it is a
    denominator, and a patient the head is sure will fail early would
    otherwise divide by rounding error.

    Args:
        cif: (n, n_causes, n_bins) cumulative incidence at the cuts.
        pmf: (n, n_causes, n_bins) the mass in each bin, the same
            prediction.
        p_survive: (n,) probability of reaching the horizon event-free.
        times_days: (n,) days to the outcome, truncated at the horizon.
        cause_index: (n,) position in the grid's causes, -1 if censored.
        is_event: (n,) whether an outcome was observed inside the grid.
        k: which cause.
        grid: the grid the prediction was made on.
        rhos: (p,) bucket upper bounds in ``[0, 1]``.

    Returns:
        ``(N_k(rhos), sum_i F_k(tau|x_i))``: the numerators, and the
        denominator that turns them into ``b_k(rho)``.
    """
    days = np.asarray(times_days, dtype=np.float64)
    ev = np.asarray(is_event, dtype=bool)
    total = np.asarray(cif, dtype=np.float64)[:, k, -1]
    f_at_t = interpolate_cif(cif[:, k, :], grid, days, per_row=True)
    # The spline returns F_k(tau) at the horizon to rounding; a u of
    # 1 + 1e-16 would fall outside the last bucket it belongs in.
    u = np.minimum(f_at_t / total, 1.0)

    failed_k = ev & (np.asarray(cause_index) == k)
    censored = ~ev
    surv_at_t = np.asarray(p_survive, dtype=np.float64) + sum(
        interpolate_tail(pmf[:, j, :], grid, days, per_row=True)
        for j in range(pmf.shape[1]))

    # Sorted by u, every bucket is a prefix: counts and the censored terms'
    # two parts, rho * F(tau)/S and F(C)/S, are prefix sums read off at
    # each rho.
    order = np.argsort(u, kind='stable')
    u_sorted = u[order]
    weight_rho = np.where(censored, total / surv_at_t, 0.0)[order]
    weight_const = np.where(censored, f_at_t / surv_at_t, 0.0)[order]

    def prefix(x):
        return np.concatenate([[0.0], np.cumsum(x)])

    upto = np.searchsorted(u_sorted, np.asarray(rhos, dtype=np.float64),
                           side='right')
    rhos = np.asarray(rhos, dtype=np.float64)
    numerator = (prefix(failed_k[order].astype(np.float64))[upto]
                 + rhos * prefix(weight_rho)[upto]
                 - prefix(weight_const)[upto])
    return numerator, float(total.sum())


def cr_d_calibration(
    cif: np.ndarray,
    pmf: np.ndarray,
    p_survive: np.ndarray,
    times_days: np.ndarray,
    cause_index: np.ndarray,
    is_event: np.ndarray,
    k: int,
    grid: TimeGrid,
    alpha: float = DEFAULT_CR_DCAL_ALPHA,
    n_rho: int = DEFAULT_CR_DCAL_N_RHO,
) -> float:
    r"""Cause ``k``'s term of CR D-calibration: zero for a calibrated model.

    .. math::
        D_k = \Big(\frac{1}{M} \sum_{m=1}^{M}
              |b_k(m/M) - m/M|^\alpha\Big)^{1/\alpha},
        \quad b_k(\rho) = N_k(\rho) / \textstyle\sum_i F_k(\tau|x_i)

    Algorithm 1's right-endpoint Riemann sum, term for term; the metric
    the paper reports is the sum of ``D_k`` over causes. It has no scale
    of its own -- there is no threshold below which a model is calibrated
    -- so it ranks models; :func:`cr_dcal_chi2` is the test of one.
    See :func:`cr_dcal_numerator` for the buckets.
    """
    rhos = np.arange(1, n_rho + 1, dtype=np.float64) / n_rho
    numerator, denominator = cr_dcal_numerator(
        cif, pmf, p_survive, times_days, cause_index, is_event, k, grid, rhos)
    gap = np.abs(numerator / denominator - rhos) ** alpha
    return float(np.mean(gap) ** (1.0 / alpha))


def cr_dcal_chi2(
    cif: np.ndarray,
    pmf: np.ndarray,
    p_survive: np.ndarray,
    times_days: np.ndarray,
    cause_index: np.ndarray,
    is_event: np.ndarray,
    k: int,
    grid: TimeGrid,
    n_bins: int = DEFAULT_CR_DCAL_CHI2_BINS,
) -> Tuple[float, float]:
    r"""Haider et al.'s chi-squared D-calibration test, on the CR buckets.

    The paper proposes a Kolmogorov-Smirnov test (appendix E.2) without
    saying how the censored patients' fractional weights enter it; this is
    Haider's Pearson test instead, which has more power. The ``[0, 1]``
    axis is cut into ``B`` equal bins, and bin ``j`` holds
    ``N_k(j/B) - N_k((j-1)/B)`` -- which is appendix D.2's per-bin weight,
    including the full-width share a censored patient spreads into every
    bin above its own.

    The expected count is the paper's null, ``E[B_{[a,b]}] = (b - a)
    F_k(\infty|x)``, summed: ``sum_i F_k(tau|x_i) / B`` in every bin. That
    tests the total mass as well as its shape, as the metric does, and
    nothing is estimated from the counts, so ``df = B``. Haider rescales to
    the observed total and takes ``df = B - 1``; here the observed total is
    itself part of what is being tested.

    Returns:
        ``(X^2, p)``. Decide on a family of causes with a Bonferroni level,
        as the paper does.
    """
    edges = np.arange(1, n_bins + 1, dtype=np.float64) / n_bins
    numerator, denominator = cr_dcal_numerator(
        cif, pmf, p_survive, times_days, cause_index, is_event, k, grid, edges)
    observed = np.diff(numerator, prepend=0.0)
    expected = denominator / n_bins
    statistic = float(np.sum((observed - expected) ** 2) / expected)
    return statistic, float(chi2.sf(statistic, df=n_bins))


def aj_recalibration_offsets(
    cif: np.ndarray,
    times_days: np.ndarray,
    cause_index: np.ndarray,
    is_event: np.ndarray,
    grid: TimeGrid,
) -> np.ndarray:
    r"""AJ-k recalibration (Alberge et al., appendix F.1), fitted on one set.

    .. math::
        d_{k,j} = \frac{1}{n} \sum_i \hat F_k(c_j|x_i) - \hat F^{AJ}_k(c_j)

    Per cause and cut, how far the model's average incidence sits above
    the marginal Aalen-Johansen curve of the same patients. Subtracting it
    from another set's predictions makes their average what this set's
    outcomes say it should be, and leaves the ordering of patients at each
    cut untouched. The survival offset is implied: the AJ incidences and
    the KM survival sum to one, as the predictions do.

    The sign is the model's excess, so that subtracting it lands the mean
    on AJ. The paper's appendix F.1, as read here, writes the difference
    the other way round and still subtracts it, which would move the mean
    to ``2 mean - AJ``; the recalibration test pins the direction.

    The AJ curve is read just before each cut, because the bins are
    left-closed: an event at exactly a cut is in the bin the cut opens,
    and so not in the model's ``F_k`` at that cut either.

    Args:
        cif: (n, n_causes, n_bins) the calibration set's predictions.
        times_days, cause_index, is_event: its outcomes, as
            :func:`cause_specific_brier` takes them.
        grid: the grid of the predictions.

    Returns:
        (n_causes, n_bins) offsets.
    """
    just_before = np.nextafter(grid.cuts, -np.inf)
    mean_cif = np.asarray(cif, dtype=np.float64).mean(axis=0)
    return np.stack([
        mean_cif[k]
        - aalen_johansen(times_days, cause_index, is_event, k, just_before)
        for k in range(grid.n_causes)])


def apply_aj_recalibration(
    cif: np.ndarray,
    pmf: np.ndarray,
    p_survive: np.ndarray,
    offsets: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Shift a set's predictions by :func:`aj_recalibration_offsets`, and repair them.

    The shift is one constant per (cause, cut) for every patient, so it can
    leave the space of distributions: a low-risk patient goes negative
    where the model over-predicts on average, a curve whose own rise
    between two cuts is smaller than the offset's turns back on itself,
    and where the model under-predicts the causes can sum past one. The
    paper says nothing of this. The repair is:

    * clip to ``[0, 1]`` and take the running maximum across cuts --
      ``make_mono_quantiles(method="ceil")`` in shi-ang/
      MakeSurvivalCalibratedAgain, per cause. :func:`interpolate_cif`
      would impose the maximum on its knots anyway; it is done here so the
      repaired arrays are valid without the spline;
    * where a patient's causes sum past one at the horizon, divide all of
      that patient's incidences by the sum, which keeps every curve
      monotone because the sum is largest at the horizon;
    * floor each ``F_k(tau)`` and the survival at :data:`RECAL_FLOOR`.

    The pmf and the survival are built on the originals, ``pmf + diff(F~ -
    F)`` and ``p_survive - sum_k (F~_k(tau) - F_k(tau))``, rather than as
    differences of the new incidences, so the head's own small numbers
    survive; the corrections are offset-sized, and their rounding sits far
    below the offsets' own estimation error.

    Returns:
        ``(cif, pmf, p_survive, repaired_fraction)``: the recalibrated
        prediction, and the fraction of (patient, cause, cut) cells the
        repair changed after the shift.
    """
    cif = np.asarray(cif, dtype=np.float64)
    shifted = cif - offsets[None, :, :]

    repaired = np.clip(shifted, 0.0, 1.0)
    np.maximum.accumulate(repaired, axis=2, out=repaired)
    total = repaired[:, :, -1].sum(axis=1)
    over = total > 1.0
    repaired[over] /= total[over, None, None]
    repaired[:, :, -1] = np.maximum(repaired[:, :, -1], RECAL_FLOOR)
    repaired_fraction = float(np.mean(repaired != shifted))

    delta = repaired - cif
    new_pmf = np.asarray(pmf, dtype=np.float64) + np.diff(
        delta, axis=2, prepend=0.0)
    new_survive = np.maximum(
        np.asarray(p_survive, dtype=np.float64) - delta[:, :, -1].sum(axis=1),
        RECAL_FLOOR)
    return repaired, new_pmf, new_survive, repaired_fraction


def marginal_reference_metrics(
    time_to_event: np.ndarray,
    event_type: np.ndarray,
    grid: TimeGrid,
) -> Dict[str, float]:
    """What a model that used none of its inputs would score.

    An integrated Brier score has no scale on its own -- it is dominated by how often the event
    happens, so a rare outcome scores well under a prediction of nearly zero for everyone. This
    is the number the model's is competing against, computed over the same episodes, the same
    time grid and the same censoring weights, so the two are comparable term for term.

    Reported beside it is the Index of Prediction Accuracy, ``1 - IBS_model / IBS_null``, which
    is the fraction of the reference's error the model removes.

    Args:
        time_to_event: (n,) minutes, as ``labels.csv`` writes it.
        event_type: (n,) EVENT_TYPE, as ``labels.csv`` writes it.
        grid: The time grid the model was scored on.

    Returns:
        ``{<cause>_Integrated_Brier_Null: float, <cause>_Events: float}``, keyed as
        ``survival_metrics`` keys its own, so a caller can divide one by the other without
        rewriting either name.
    """
    t = torch.as_tensor(time_to_event, dtype=torch.float64)
    e = torch.as_tensor(event_type)
    _, cause_index, is_event = grid.discretize(t, e)

    ci = cause_index.numpy()
    ev = is_event.numpy().astype(bool)
    days = event_times_days(t.numpy(), grid)
    taus = brier_times(grid)

    out: Dict[str, float] = {}
    for k, name in enumerate(grid.cause_names):
        key = metric_label(name)
        reference = np.tile(aalen_johansen(days, ci, ev, k, taus),
                            (days.size, 1))
        out[f"{key}_Integrated_Brier_Null"] = integrated_brier(
            cause_specific_brier(reference, days, ci, ev, k, taus), taus)
        out[f"{key}_Events"] = float(np.sum(ev & (ci == k)))
    out["Censored"] = float(np.sum(~ev))
    return out


def survival_metrics(
    logits: np.ndarray,
    time_to_event: np.ndarray,
    event_type: np.ndarray,
    grid: TimeGrid,
) -> Dict[str, float]:
    """Every score the finetuning routines report, from raw model output.

    Args:
        logits: (n, n_causes * n_bins), the head's output.
        time_to_event: (n,) minutes, as ``labels.csv`` writes it.
        event_type: (n,) EVENT_TYPE, as ``labels.csv`` writes it.

    Returns:
        A flat dict of scalars: the joint NLL, then per cause the
        time-dependent concordance overall and at each cut, and the
        Brier score integrated over ``grid.brier_integration_days``. The
        bound is not in the key: like the cuts and the modelled causes, it
        is a property of the run, and the run records it.
    """
    t = torch.as_tensor(time_to_event, dtype=torch.float64)
    e = torch.as_tensor(event_type)
    bin_index, cause_index, is_event = grid.discretize(t, e)

    pmf, p_survive = deephit_distribution(
        torch.as_tensor(logits, dtype=torch.float64),
        grid.n_causes, grid.n_bins)
    cif = cif_from_pmf(pmf).numpy()
    surv = survival_from_pmf(pmf, p_survive).numpy()
    bi = bin_index.numpy()
    ci = cause_index.numpy()
    ev = is_event.numpy().astype(bool)
    days = event_times_days(t.numpy(), grid)
    taus = brier_times(grid)

    out: Dict[str, float] = {"Loss_DeepHit_NLL": float(
        _nll_numpy(pmf.numpy(), surv, bi, ci, ev))}

    for k, name in enumerate(grid.cause_names):
        key = metric_label(name)
        out[f"{key}_Cindex"] = cause_specific_concordance(
            cif, bi, ci, ev, k)
        for b, cut in enumerate(grid.cuts):
            out[f"{key}_Cindex_at_{format_days(float(cut)).replace(' ', '')}"] = (
                cause_specific_concordance(cif, bi, ci, ev, k,
                                           horizon_bin=b))
        out[f"{key}_Integrated_Brier"] = integrated_brier(
            cause_specific_brier(interpolate_cif(cif[:, k, :], grid, taus),
                                 days, ci, ev, k, taus), taus)
        out[f"{key}_Events"] = float(np.sum(ev & (ci == k)))

    finite = [out[f"{metric_label(n)}_Cindex"] for n in grid.cause_names]
    finite = [v for v in finite if np.isfinite(v)]
    out["Mean_Cindex"] = float(np.mean(finite)) if finite else float("nan")
    out["Censored"] = float(np.sum(~ev))
    return out


def _nll_numpy(pmf: np.ndarray, surv: np.ndarray, bin_index: np.ndarray,
               cause_index: np.ndarray, is_event: np.ndarray) -> float:
    """The DeepHit likelihood term, for reporting. See ``DeepHitLoss``."""
    eps = 1e-8
    rows = np.arange(bin_index.size)
    event_p = pmf[rows, np.maximum(cause_index, 0), bin_index]
    cens_p = np.maximum(surv[rows, bin_index], 0.0)
    ll = np.log(np.where(is_event, event_p, cens_p) + eps)
    return float(-np.mean(ll))
