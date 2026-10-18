"""The cause-specific Cox baseline's curves, rebuilt from its parameters.

The CSC is fitted in R (``scripts/fit_baseline_csc.R``, with
``riskRegression::CSC``), which exports, per fold, what its predictions are
made of rather than the predictions themselves: every patient's linear
predictor per cause, and each cause's baseline cumulative hazard at the
training event times. The curve is rebuilt here, exactly, at whatever times
a metric reads it -- each patient's own event time included, out to the
horizon -- where storing it would take one column per day per cause.

The reconstruction is ``riskRegression``'s own (``predictCIF_cpp`` with its
default ``product.limit = TRUE``, on an uncentred baseline):

.. math::
    F_k(t|x) = \\sum_{s \\le t} S(s-|x)\\, e^{lp_k}\\, \\Delta\\Lambda_{0k}(s),
    \\quad S(s-|x) = \\prod_{u < s} \\big(1 - \\textstyle\\sum_j
        e^{lp_j} \\Delta\\Lambda_{0j}(u)\\big)

a step function that changes only at the training event times, read
right-continuously. Past the last training event time the curve is held
flat, where ``riskRegression`` returns NA.

One departure (user, 2026-10-18): ``riskRegression`` does not clip the
product-limit factor, which goes negative wherever a patient's summed
hazard step ``h = \\sum_j e^{lp_j} \\Delta\\Lambda_{0j}(s)`` exceeds one. That
happens in the far tail, where the risk set is a handful of patients and a
higher-risk patient's step outgrows it; from there the CIFs sum past one
and fall. Such a step is normalised: ``e^{lp_k} \\Delta\\Lambda_{0k}(s)``
becomes ``e^{lp_k} \\Delta\\Lambda_{0k}(s) / h``, so the patient's remaining
mass goes to the causes in proportion and ``S`` drops to zero. Every step
with ``h <= 1`` is ``riskRegression``'s exactly, so the curves agree up to a
patient's first such step, and ``sum_k F_k = 1 - S`` stays in [0, 1].

Files, under ``{model_dir}/{experiment}/fold{i}/competing_risks/``:

* ``competing_risks_{split}_csc_lp.csv``, split in train/val/test: ``idx``,
  ``time_to_event`` (minutes) and ``event_type`` as ``labels.csv`` has them,
  then ``lp_<cause>`` for every modelled cause, ``x beta_k`` uncentred.
* ``competing_risks_csc_baseline_hazard.csv``: ``time_days``, ascending and
  distinct, then ``cumhaz_<cause>``: the uncentred baseline cumulative
  hazard ``Lambda_0k`` at that time, the right-continuous value. One row per
  training event time of any cause; a cause with no event at a row repeats
  its previous value.
* ``competing_risks_csc_recal_offsets.csv``, written by
  ``recalibrate_predictions.py``: ``time_days`` and ``offset_<cause>``, the
  dense AJ-k offsets on the recalibration axis (:func:`recalibration_axis`).
"""

from __future__ import annotations

import numpy as np
import os
import pandas as pd

from typing import Dict, Optional, Tuple

from TransEHR2.survival import (
    TimeGrid, apply_dense_aj_recalibration, dense_aj_offsets
)


LP_FILE = '{task}_{split}_csc_lp.csv'
BASELINE_FILE = '{task}_csc_baseline_hazard.csv'
OFFSETS_FILE = '{task}_csc_recal_offsets.csv'

# Patients rebuilt at once. Each holds (n_causes, training event times)
# float64s several times over, so this bounds memory at tens of megabytes.
CHUNK = 64


def _columns(frame: pd.DataFrame, prefix: str, grid: TimeGrid,
             path: str) -> np.ndarray:
    names = [f'{prefix}_{cause}' for cause in grid.cause_names]
    missing = [c for c in names if c not in frame.columns]
    if missing:
        raise ValueError(f'{path} lacks {missing}')
    return frame[names].to_numpy(dtype=np.float64)


def read_csc_lp(path: str, grid: TimeGrid) -> Optional[Dict]:
    """One split's linear predictors and outcomes, or None if absent.

    Returns ``{'idx', 'time_to_event', 'event_type', 'lp'}``, ``lp`` of
    shape (n, n_causes) in the grid's cause order.
    """
    if not os.path.exists(path):
        return None
    frame = pd.read_csv(path, float_precision='round_trip')
    return {'idx': frame['idx'].to_numpy(),
            'time_to_event': frame['time_to_event'].to_numpy(),
            'event_type': frame['event_type'].to_numpy(),
            'lp': _columns(frame, 'lp', grid, path)}


def read_csc_baseline(path: str, grid: TimeGrid
                      ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """``(times, cumhaz)``, cumhaz (n_causes, m), or None if absent."""
    if not os.path.exists(path):
        return None
    frame = pd.read_csv(path, float_precision='round_trip')
    times = frame['time_days'].to_numpy(dtype=np.float64)
    cumhaz = _columns(frame, 'cumhaz', grid, path).T
    if times.size == 0 or np.any(np.diff(times) <= 0) or times[0] < 0:
        raise ValueError(f'{path}: time_days must be non-negative, distinct '
                         f'and ascending')
    if np.any(cumhaz < 0) or np.any(np.diff(cumhaz, axis=1) < 0):
        raise ValueError(f'{path}: a cumulative hazard must be non-negative '
                         f'and non-decreasing')
    return times, cumhaz


def read_csc_offsets(path: str, grid: TimeGrid
                     ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """``(times, offsets)``, offsets (n_causes, p), or None if absent."""
    if not os.path.exists(path):
        return None
    frame = pd.read_csv(path, float_precision='round_trip')
    return (frame['time_days'].to_numpy(dtype=np.float64),
            _columns(frame, 'offset', grid, path).T)


def save_csc_offsets(path: str, times: np.ndarray, offsets: np.ndarray,
                     grid: TimeGrid) -> None:
    columns = {'time_days': times}
    for k, cause in enumerate(grid.cause_names):
        columns[f'offset_{cause}'] = offsets[k]
    pd.DataFrame(columns).to_csv(path, index=False)
    print(f"    Wrote {path} ({len(times):,} rows)")


def recalibration_axis(baseline_times: np.ndarray, val_days: np.ndarray,
                       val_is_event: np.ndarray, grid: TimeGrid) -> np.ndarray:
    """Every time the recalibrated curve can change, to the horizon.

    Zero, the training event times (where every patient's CIF steps), the
    cuts, and the val event times (where AJ steps). ``d_k(t) = mean F_k(t)
    - AJ_k(t-)`` reads AJ just before ``t``, so at a val event time ``v``
    the offset takes one value at ``v`` itself and another for every time
    after it; the axis carries the next float above ``v`` as well, which is
    what makes "the last point at or before ``t``" the curve's value at
    every ``t``. The horizon is the last point, where the repair rescales.
    """
    events = np.asarray(val_days, dtype=np.float64)[
        np.asarray(val_is_event, dtype=bool)]
    axis = np.unique(np.concatenate([
        [0.0], baseline_times, grid.cuts, events,
        np.nextafter(events, np.inf)]))
    return axis[axis <= grid.horizon_days]


class CSCCurve:
    """Patients' CSC incidences, raw or AJ-k recalibrated, read exactly.

    Args:
        lp: (n, n_causes) linear predictors.
        baseline: ``(times, cumhaz)`` from :func:`read_csc_baseline`.
        offsets: ``(axis, d)`` from :func:`csc_dense_offsets`, to read the
            recalibrated curve: ``F_k(t) - d_k(t)`` on ``axis``, repaired
            once along it (``survival.apply_dense_aj_recalibration``), and
            read at the last axis point at or before each time.
    """

    def __init__(self, lp: np.ndarray,
                 baseline: Tuple[np.ndarray, np.ndarray],
                 offsets: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 chunk: int = CHUNK):
        self.risk = np.exp(np.asarray(lp, dtype=np.float64))
        self.times, cumhaz = baseline
        self.hazard = np.diff(cumhaz, axis=1, prepend=0.0)
        self.offsets = offsets
        self.chunk = chunk
        self.repaired_fraction: Optional[float] = None

    @property
    def n(self) -> int:
        return self.risk.shape[0]

    def _at_events(self, rows: slice) -> np.ndarray:
        """(c, n_causes, 1 + m): zero, then F_k at each training event time."""
        hazard = self.risk[rows, :, None] * self.hazard[None, :, :]
        # A step whose summed hazard exceeds one is scaled down to one; every
        # other step is divided by exactly 1.0 and so left untouched.
        scale = np.maximum(hazard.sum(axis=1), 1.0)
        hazard = hazard / scale[:, None, :]
        surv = np.cumprod(1.0 - hazard.sum(axis=1), axis=1)
        before = np.concatenate([np.ones((surv.shape[0], 1)), surv[:, :-1]],
                                axis=1)
        cif = np.cumsum(before[:, None, :] * hazard, axis=2)
        return np.concatenate([np.zeros(cif.shape[:2] + (1,)), cif], axis=2)

    def _knots(self, rows: slice) -> Tuple[np.ndarray, np.ndarray, float]:
        """The curve as values at knots: ``(knots, values, repaired)``.

        A time ``t`` reads ``values[..., searchsorted(knots, t, 'right')]``.
        """
        events = self._at_events(rows)
        if self.offsets is None:
            return self.times, events, 0.0
        axis, d = self.offsets
        on_axis = events[:, :, np.searchsorted(self.times, axis,
                                               side='right')]
        repaired, fraction = apply_dense_aj_recalibration(on_axis, d)
        # The axis starts at zero, so nothing reads before its first point.
        return axis[1:], repaired, fraction

    def read(self, times: np.ndarray, row_times: Optional[np.ndarray] = None
             ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """``F`` at shared ``times``, (n, n_causes, p), and at each row's own.

        ``row_times`` (n,) gives each patient's own time; its reads come back
        as (n, n_causes). With offsets, ``repaired_fraction`` is set to the
        fraction of (patient, cause, axis point) cells the repair changed.
        """
        times = np.asarray(times, dtype=np.float64)
        shared = np.empty((self.n, self.hazard.shape[0], times.size))
        rows_out = (None if row_times is None
                    else np.empty((self.n, self.hazard.shape[0])))
        changed = 0.0
        for lo in range(0, self.n, self.chunk):
            rows = slice(lo, min(lo + self.chunk, self.n))
            knots, values, fraction = self._knots(rows)
            changed += fraction * values.size
            shared[rows] = values[:, :, np.searchsorted(knots, times,
                                                        side='right')]
            if row_times is not None:
                at = np.searchsorted(knots, np.asarray(row_times)[rows],
                                     side='right')
                rows_out[rows] = values[np.arange(at.size), :, at]
        if self.offsets is not None:
            self.repaired_fraction = changed / (
                self.n * self.hazard.shape[0] * self.offsets[0].size)
        return shared, rows_out

    def mean_at(self, times: np.ndarray) -> np.ndarray:
        """(n_causes, p) the patients' mean ``F`` at ``times``."""
        times = np.asarray(times, dtype=np.float64)
        total = np.zeros((self.hazard.shape[0], times.size))
        for lo in range(0, self.n, self.chunk):
            rows = slice(lo, min(lo + self.chunk, self.n))
            knots, values, _ = self._knots(rows)
            total += values[:, :, np.searchsorted(knots, times,
                                                  side='right')].sum(axis=0)
        return total / self.n


def csc_dense_offsets(
    val_curve: CSCCurve, baseline_times: np.ndarray, val_days: np.ndarray,
    val_cause_index: np.ndarray, val_is_event: np.ndarray, grid: TimeGrid,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(axis, d)``: the dense AJ-k offsets of the raw val curve.

    ``d_k(t) = mean_i F_k(t|x_i) - AJ_k(t-)`` on val, at every point of
    :func:`recalibration_axis`; at the cuts these are the grid offsets of
    the val dump written from the same curve.
    """
    axis = recalibration_axis(baseline_times, val_days, val_is_event, grid)
    return axis, dense_aj_offsets(val_curve.mean_at(axis), axis, val_days,
                                  val_cause_index, val_is_event)
