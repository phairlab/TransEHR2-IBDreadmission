"""Monotone cubic interpolation, as R's ``splinefun(method = "hyman")``.

A DeepHit head predicts on a handful of bins -- six, by default, the last
three of them years wide. Read literally, a patient's cumulative incidence
is a step function that moves only at the cuts, and anything asked of it
between them has to be invented. Interpolating with a monotone cubic is
one way to invent it, and the way MTLR draws its survival curves (see
``plotcurves.R`` in haiderstats/MTLR, lines 48-55): fit a spline, filter it
so it cannot turn back on itself, and read the curve off that.

What this buys is a cumulative incidence defined at every time rather than
at six of them, which is what lets the Brier score be integrated over a
dense axis instead of over four unequal bins where the widest one carries
three quarters of the weight. What it does not buy is information: the head
still emits six numbers per cause, and the curve between them is an
assumption about smoothness, not an estimate. The assumption becomes part
of the predictor -- the model is DeepHit *and* this spline -- which is why
scoring and reporting both go through it.

Why a port rather than a call into R
------------------------------------

``splinefun(method = "hyman")`` is not in SciPy. Its ``PchipInterpolator``
is Fritsch-Carlson, which is R's ``monoH.FC``, a different curve. So the
algorithm had to come from somewhere, and the choice was between calling R
through rpy2 and porting it.

Porting won on two counts. The curves are rebuilt every validation epoch,
and every stay in a split shares the same knots -- the grid's cuts -- so
the tridiagonal system is the same matrix for all of them and differs only
in its right-hand side. One elimination serves the whole cohort, which is a
saving ``splinefun`` cannot take because it is handed one curve at a time:
15,000 curves at 366 points take 59 ms here against 732 ms looping in R,
before anything is marshalled across the process boundary. The second count
is that the training runs on the cluster, where rpy2 and an R installation
would have to stand next to torch.

The port is pinned to R rather than trusted: ``tests/test_hyman.py`` checks
it against curves ``splinefun`` itself produced.

What the algorithm is
---------------------

``method = "hyman"`` is not its own spline. R fits ordinary FMM
coefficients -- ``splinefun``'s dispatch is ``min(3L, iMeth)``, and 3 is
``fmm`` -- then filters the first derivatives so that no interval can
overshoot, then rebuilds the quadratic and cubic coefficients from the
filtered derivatives. The three steps are :func:`_fmm_coefficients`,
:func:`_hyman_filter` and :func:`_spline_coef_conv` below, each a
transcription of its counterpart in R (``src/library/stats/src/splines.c``
and ``stats:::hyman_filter`` / ``stats:::spl_coef_conv``), with the curve
index carried as a leading axis.

The filter requires monotone input and R refuses non-monotone ``y``. A
cumulative incidence is a cumulative sum of non-negative mass, so it
qualifies by construction; :func:`hyman_spline` still checks, because a
float32 cumsum can land a few ulps the wrong way.
"""

from __future__ import annotations

import numpy as np


def _fmm_coefficients(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """First derivatives of the FMM cubic spline through each row of ``y``.

    R's ``fmm_spline`` from ``splines.c``, with two differences. Only the
    ``b`` coefficients are returned, because the hyman path discards ``c``
    and ``d`` and rebuilds them after filtering. And the elimination runs
    once rather than once per curve: the tridiagonal matrix is built from
    ``x`` alone, which every curve shares, so only the right-hand side
    carries a curve index.

    The end conditions are FMM's: the third derivative at each end is
    taken from a divided difference over the outermost four knots, which
    is what distinguishes it from a natural spline. With fewer than four
    knots there is no such difference to take and R leaves the ends at
    zero; this follows it.

    Args:
        x: (n,) knots, strictly ascending.
        y: (m, n) values, one curve per row.

    Returns:
        (m, n) first derivatives at the knots.
    """
    n = x.size
    h = np.diff(x)                                       # (n - 1,)

    diag = np.empty(n, dtype=np.float64)
    diag[0] = -h[0]
    diag[-1] = -h[-1]
    diag[1:-1] = 2.0 * (h[:-1] + h[1:])

    slope = np.diff(y, axis=1) / h                       # (m, n - 1)
    rhs = np.zeros_like(y)
    rhs[:, 1:-1] = np.diff(slope, axis=1)
    if n > 3:
        rhs[:, 0] = (rhs[:, 2] / (x[3] - x[1])
                     - rhs[:, 1] / (x[2] - x[0]))
        rhs[:, -1] = (rhs[:, -2] / (x[-1] - x[-3])
                      - rhs[:, -3] / (x[-2] - x[-4]))
        rhs[:, 0] *= h[0] ** 2 / (x[3] - x[0])
        rhs[:, -1] *= -h[-1] ** 2 / (x[-1] - x[-4])

    for i in range(1, n):                                # forward elimination
        factor = h[i - 1] / diag[i - 1]
        diag[i] -= factor * h[i - 1]
        rhs[:, i] -= factor * rhs[:, i - 1]

    rhs[:, -1] /= diag[-1]                               # back substitution
    for i in range(n - 2, -1, -1):
        rhs[:, i] = (rhs[:, i] - h[i] * rhs[:, i + 1]) / diag[i]

    b = np.empty_like(y)
    b[:, :-1] = slope - h * (rhs[:, 1:] + 2.0 * rhs[:, :-1])
    b[:, -1] = slope[:, -1] + h[-1] * (rhs[:, -2] + 2.0 * rhs[:, -1])
    return b


def _hyman_filter(x: np.ndarray, y: np.ndarray,
                  b: np.ndarray) -> np.ndarray:
    """Clamp each derivative into the range a monotone cubic allows.

    ``stats:::hyman_filter``, verbatim but batched. A Hermite cubic on an
    interval stays monotone when the derivatives at its ends sit between
    zero and three times the secant slope; the filter projects each
    derivative onto that range, taking its sign from the neighbouring
    secants where they agree.
    """
    secant = np.diff(y, axis=1) / np.diff(x)
    before = np.concatenate([secant[:, :1], secant], axis=1)
    after = np.concatenate([secant, secant[:, -1:]], axis=1)
    bound = 3.0 * np.minimum(np.abs(before), np.abs(after))

    sign = np.where(before * after > 0, after, b)
    return np.where(
        sign >= 0,
        np.minimum(np.maximum(0.0, b), bound),
        np.maximum(np.minimum(0.0, b), -bound),
    )


def _spline_coef_conv(x: np.ndarray, y: np.ndarray,
                      b: np.ndarray) -> tuple:
    """Rebuild the quadratic and cubic coefficients from ``b``.

    ``stats:::spl_coef_conv``, verbatim but batched. Filtering changed the
    first derivatives, so the higher coefficients no longer describe the
    same curve; these are the ones a Hermite cubic through the same knots
    with these derivatives would have.
    """
    h = np.diff(x)
    drop = -np.diff(y, axis=1)
    b0, b1 = b[:, :-1], b[:, 1:]

    c = -(3.0 * drop + (2.0 * b0 + b1) * h) / h ** 2
    c_last = ((3.0 * drop[:, -1] + (b0[:, -1] + 2.0 * b1[:, -1]) * h[-1])
              / h[-1] ** 2)
    d = (2.0 * drop / h + b0 + b1) / h ** 2
    return (np.concatenate([c, c_last[:, None]], axis=1),
            np.concatenate([d, d[:, -1:]], axis=1))


def hyman_spline(x: np.ndarray, y: np.ndarray,
                 xout: np.ndarray) -> np.ndarray:
    """Evaluate ``splinefun(x, y, method = "hyman")`` at ``xout``.

    Args:
        x: (n,) knots, strictly ascending, ``n >= 3``.
        y: (m, n) monotone curves, one per row. A single (n,) curve is
            accepted and comes back as a (1, p) row.
        xout: (p,) times to evaluate at. Values outside ``x`` extrapolate
            along the end cubics, as R's does.

    Returns:
        (m, p) interpolated values.

    Raises:
        ValueError: if a row of ``y`` is not monotone, or ``n < 3``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    xout = np.asarray(xout, dtype=np.float64)

    if x.ndim != 1 or x.size < 3:
        raise ValueError(f"x must be a 1-D sequence of at least 3 knots, "
                         f"got shape {x.shape}")
    if y.shape[-1] != x.size:
        raise ValueError(f"y has {y.shape[-1]} columns but x has {x.size} "
                         f"knots")

    step = np.diff(y, axis=1)
    if not (np.all(step >= 0.0, axis=1) | np.all(step <= 0.0, axis=1)).all():
        raise ValueError("hyman interpolation needs monotone y; at least "
                         "one row turns back on itself")

    b = _hyman_filter(x, y, _fmm_coefficients(x, y))
    c, d = _spline_coef_conv(x, y, b)

    i = np.clip(np.searchsorted(x, xout, side="right") - 1, 0, x.size - 1)
    dx = xout - x[i]
    return y[:, i] + dx * (b[:, i] + dx * (c[:, i] + dx * d[:, i]))
