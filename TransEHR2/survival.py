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

from torch import Tensor
from typing import Dict, List, Optional, Sequence, Tuple

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
    """
    total = pmf.sum(dim=1)                                  # (batch, n_bins)
    tail_exclusive = total.flip(-1).cumsum(-1).flip(-1) - total
    return tail_exclusive + p_survive.unsqueeze(-1)


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
    cif_k: np.ndarray, grid: TimeGrid, times: np.ndarray
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

    Returns:
        (n, p) cumulative incidence.
    """
    anchored = np.concatenate(
        [np.zeros((cif_k.shape[0], 1), dtype=np.float64),
         np.asarray(cif_k, dtype=np.float64)], axis=1)
    # A cumulative sum of non-negative mass is monotone by construction,
    # but it reaches here through a float32 softmax, and the filter
    # refuses input that steps backwards by even one ulp.
    np.maximum.accumulate(anchored, axis=1, out=anchored)

    curve = hyman_spline(grid.edges_days, anchored, times)
    # Monotone interpolation between knots inside [0, 1] cannot leave it;
    # the clip is against rounding at the ends, not against the spline.
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
