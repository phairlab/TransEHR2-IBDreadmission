"""Flagging patients at a horizon: the outcome, its IPCW weights, and the
operating characteristics of a score (plan 2026-11-18, Task C).

A flag at horizon ``H`` is judged against the composite of the first event
being readmission or death by ``H`` -- LACE's outcome (van Walraven 2010:
early death or unplanned readmission). Every patient falls in one of four
groups, weighted as :func:`TransEHR2.survival.cause_specific_brier` weighs
them, so that the flags and the Brier score read the same outcome the same
way:

* **positive**: the first event is a composite cause, by ``H``; weight
  ``1 / G(y-)``;
* **negative, observed**: the first event is another cause (out-migration)
  by ``H``, weight ``1 / G(y-)``; or event-free at ``H`` (``y > H``),
  weight ``1 / G(H)``;
* **censored before ``H``**: weight 0.

``G`` is the Kaplan-Meier of censoring on the scored patients themselves.

A flag is ``score >= threshold``. The weighted counts are sums of weights,
so with no censoring before ``H`` every weight is 1 and the rates are the
plain ones.
"""

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .survival import _censoring_at, censoring_survival

# numpy 2 renamed trapz; the cluster and the laptop may differ.
_trapezoid = getattr(np, 'trapezoid', None) or np.trapz


def composite_outcome(days: np.ndarray, cause_index: np.ndarray,
                      is_event: np.ndarray, causes: Iterable[int],
                      horizon: float) -> Tuple[np.ndarray, np.ndarray]:
    """``(positive, weight)`` at ``horizon`` for the composite of
    ``causes`` (positions in the grid's causes), as the module describes.

    Args:
        days: (n,) days to the first outcome, as ``event_times_days``.
        cause_index: (n,) the first outcome's cause, -1 when censored.
        is_event: (n,) whether an outcome was observed.
    """
    t = np.asarray(days, dtype=np.float64)
    ev = np.asarray(is_event, dtype=bool)
    by_now = t <= horizon
    positive = by_now & ev & np.isin(cause_index, list(causes))

    jump_times, surv = censoring_survival(t, ev)
    g_event = _censoring_at(jump_times, surv, t, left_limit=True)
    g_horizon = _censoring_at(jump_times, surv, np.array([horizon]),
                              left_limit=False)[0]
    weight = np.where(by_now, 1.0 / g_event, 1.0 / g_horizon)
    weight = np.where(by_now & ~ev, 0.0, weight)
    return positive, weight


def rates(flag: np.ndarray, positive: np.ndarray,
          weight: np.ndarray) -> Dict[str, float]:
    """IPCW sensitivity, specificity and PPV of ``flag``, and the plain
    fraction of patients flagged. A rate with no weight behind it is NaN.
    """
    flag = np.asarray(flag, dtype=bool)
    positive = np.asarray(positive, dtype=bool)
    tp = float(np.sum(weight[flag & positive]))
    fp = float(np.sum(weight[flag & ~positive]))
    fn = float(np.sum(weight[~flag & positive]))
    tn = float(np.sum(weight[~flag & ~positive]))

    def ratio(a, b):
        return a / b if b > 0 else float('nan')

    return {'Sensitivity': ratio(tp, tp + fn),
            'Specificity': ratio(tn, tn + fp),
            'PPV': ratio(tp, tp + fp),
            'Flag_Rate': float(flag.mean()) if flag.size else float('nan')}


def sensitivity_at(score: np.ndarray, positive: np.ndarray,
                   weight: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """IPCW sensitivity of ``score >= c`` for each ``c`` in
    ``thresholds``."""
    total = float(np.sum(weight[positive]))
    if total <= 0:
        return np.full(np.shape(thresholds), np.nan)
    pos_scores = np.sort(score[positive])
    pos_weights = weight[positive][np.argsort(score[positive],
                                              kind='stable')]
    # Weight at or above each threshold: the total less what is below it.
    below = np.concatenate([[0.0], np.cumsum(pos_weights)])[
        np.searchsorted(pos_scores, thresholds, side='left')]
    return (total - below) / total


def roc_curve(score: np.ndarray, positive: np.ndarray,
              weight: np.ndarray) -> Dict[str, np.ndarray]:
    """The IPCW ROC curve of ``score``, a point per distinct value.

    Returns ``threshold`` (descending, starting at +inf, flagging no one),
    ``fpr`` and ``tpr``, and the trapezoidal ``auc``. On an integer score
    such as LACE this is the step ROC, one point per score value.
    """
    score = np.asarray(score, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    thresholds = np.concatenate([[np.inf], np.unique(score)[::-1]])
    flagged = score[None, :] >= thresholds[:, None]
    w_pos, w_neg = weight * positive, weight * ~positive
    tpr = (flagged @ w_pos) / w_pos.sum() if w_pos.sum() > 0 else \
        np.full(thresholds.size, np.nan)
    fpr = (flagged @ w_neg) / w_neg.sum() if w_neg.sum() > 0 else \
        np.full(thresholds.size, np.nan)
    auc = float(_trapezoid(tpr, fpr)) if np.all(np.isfinite(tpr + fpr)) \
        else float('nan')
    return {'threshold': thresholds, 'fpr': fpr, 'tpr': tpr, 'auc': auc}


def net_benefit(flag: np.ndarray, positive: np.ndarray, weight: np.ndarray,
                p_t: np.ndarray) -> np.ndarray:
    """Net benefit of one flag policy at each threshold probability.

    ``NB = TP_w / n - FP_w / n * p_t / (1 - p_t)``, with ``n`` every patient
    scored, the censored included: the weights of the uncensored stand in
    for them, as in the Brier score's division by ``n``.
    """
    flag = np.asarray(flag, dtype=bool)
    positive = np.asarray(positive, dtype=bool)
    n = flag.size
    tp = float(np.sum(weight[flag & positive])) / n
    fp = float(np.sum(weight[flag & ~positive])) / n
    p_t = np.asarray(p_t, dtype=np.float64)
    return tp - fp * p_t / (1.0 - p_t)


def decision_curve(score: np.ndarray, positive: np.ndarray,
                   weight: np.ndarray, p_t: np.ndarray) -> np.ndarray:
    """A risk score's net benefit when it flags ``score >= p_t``, at each
    ``p_t``: the score is read as the probability of the outcome."""
    return np.array([net_benefit(score >= p, positive, weight, p)
                     for p in np.asarray(p_t, dtype=np.float64)])


def matched_rate_threshold(score: np.ndarray,
                           fraction: float) -> Optional[float]:
    """Rule (a): the threshold at which ``score >= threshold`` flags
    ``fraction`` of the patients, up to ties: the ``k``-th highest score,
    ``k = round(fraction * n)``. None when ``k`` is 0 (flag no one)."""
    k = int(round(fraction * score.size))
    if k <= 0:
        return None
    return float(np.sort(score)[::-1][min(k, score.size) - 1])


def matched_sensitivity_threshold(score: np.ndarray, positive: np.ndarray,
                                  weight: np.ndarray,
                                  target: float) -> Optional[float]:
    """Rule (b): the largest threshold among the score's values whose IPCW
    sensitivity is at least ``target``; None when no positive has
    weight."""
    candidates = np.unique(score)
    sensitivity = sensitivity_at(score, positive, weight, candidates)
    if not np.isfinite(target) or not np.any(np.isfinite(sensitivity)):
        return None
    meets = candidates[sensitivity >= target - 1e-12]
    return float(meets.max()) if meets.size else float(candidates.min())
