"""A synthetic competing-risks cohort whose outcomes come from its own predictions.

Each patient gets a random DeepHit surface, and their outcome is drawn from
that surface as the predictor reads it -- the cell by its mass, the time
within the bin by inverting the Hyman spline -- so the predictions are
calibrated by construction. Censoring is independent and uniform.
"""

import numpy as np

from TransEHR2.survival import MINUTES_PER_DAY, TimeGrid, interpolate_cif


def simulate(grid: TimeGrid, n: int, seed: int) -> dict:
    """Predictions and outcomes for ``n`` patients, in the dump's layout.

    Returns:
        ``{'cif', 'pmf', 'p_survive', 'time_to_event', 'event_type'}``, with
        ``time_to_event`` in minutes and ``event_type`` in EVENT_TYPE codes,
        as ``read_predictions`` returns them.
    """
    rng = np.random.default_rng(seed)
    n_causes, n_bins = grid.n_causes, grid.n_bins
    n_cells = n_causes * n_bins

    # Concentrations below one give saturated heads as well as diffuse ones.
    concentration = rng.uniform(0.2, 2.0, size=(n, 1))
    full = rng.gamma(np.broadcast_to(concentration, (n, n_cells + 1)))
    full /= full.sum(axis=1, keepdims=True)
    pmf = full[:, :-1].reshape(n, n_causes, n_bins)
    p_survive = full[:, -1]
    cif = np.cumsum(pmf, axis=2)

    cell = (np.cumsum(full, axis=1) < rng.uniform(size=(n, 1))).sum(axis=1)
    cell = np.minimum(cell, n_cells)
    survived = cell == n_cells
    cause = np.where(survived, 0, cell // n_bins)
    bin_ = np.where(survived, 0, cell % n_bins)

    days = np.full(n, 2.0 * grid.horizon_days)
    fine = np.linspace(0.0, grid.horizon_days, 501)
    for k in range(n_causes):
        rows = np.flatnonzero(~survived & (cause == k))
        j = bin_[rows]
        lo = np.where(j == 0, 0.0, cif[rows, k, np.maximum(j - 1, 0)])
        hi = cif[rows, k, j]
        target = lo + rng.uniform(size=rows.size) * (hi - lo)

        curve = interpolate_cif(cif[rows, k, :], grid, fine)
        idx = np.clip((curve < target[:, None]).sum(axis=1), 1, fine.size - 1)
        r = np.arange(rows.size)
        c0, c1 = curve[r, idx - 1], curve[r, idx]
        frac = np.where(c1 > c0, (target - c0) / np.where(c1 > c0, c1 - c0, 1.0),
                        0.5)
        days[rows] = fine[idx - 1] + frac * (fine[idx] - fine[idx - 1])

    censor = rng.uniform(0.0, 1.5 * grid.horizon_days, size=n)
    observed = ~survived & (days <= censor)
    codes = np.asarray(grid.causes)
    return {
        'cif': cif,
        'pmf': pmf,
        'p_survive': p_survive,
        'time_to_event': np.minimum(days, censor) * MINUTES_PER_DAY,
        'event_type': np.where(observed, codes[cause], 0),
    }
