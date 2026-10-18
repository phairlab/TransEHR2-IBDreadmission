"""The Hyman spline of :mod:`TransEHR2.hyman`, in torch, for autograd.

The predictor is DeepHit *and* the spline, so a gradient of anything read
off the curve -- a cumulative incidence at a horizon, a tail mass, a score
-- with respect to the model has to pass through the interpolation. The
NumPy port cannot carry one; this can.

It is the same algorithm, op for op: the FMM slopes from one tridiagonal
elimination shared by every curve on the grid's cuts, Hyman's filter, then
the Hermite coefficients rebuilt from the filtered slopes. Only the storage
changed. The elimination writes into a list of columns rather than into an
array in place, because autograd refuses in-place writes to tensors it
still needs, and every operation is one autograd can differentiate.

What the gradient is
--------------------

The knots are fixed, so the FMM slopes are a linear map of the knot
values. The filter clamps each slope with ``min``/``max`` against bounds
that are themselves linear in the knot values, and picks a branch by the
signs of the neighbouring secants. So the curve is piecewise linear in the
knot values -- smooth almost everywhere, with kinks where a slope meets
its bound, meets zero, or a secant changes sign. At a kink autograd
returns one side's derivative; which side is torch's convention for
``minimum``/``maximum``/``where``, not a choice made here.

The curve is also differentiable in ``xout`` -- its derivative there is
the interpolated density -- since ``dx`` is computed in torch. The knots
``x`` are meant to be fixed; nothing stops them carrying a gradient, but
nothing tests it either.
"""

from __future__ import annotations

import torch


def _fmm_coefficients(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """First derivatives of the FMM cubic spline through each row of ``y``.

    :func:`TransEHR2.hyman._fmm_coefficients`, with the right-hand side
    held as a list of (m,) columns so that elimination never writes into a
    tensor autograd has saved.

    Args:
        x: (n,) knots, strictly ascending.
        y: (m, n) values, one curve per row.

    Returns:
        (m, n) first derivatives at the knots.
    """
    n = x.numel()
    h = torch.diff(x)                                    # (n - 1,)

    diag = [-h[0]] + [2.0 * (h[i - 1] + h[i]) for i in range(1, n - 1)] \
        + [-h[-1]]

    slope = torch.diff(y, dim=1) / h                     # (m, n - 1)
    interior = torch.diff(slope, dim=1)                  # (m, n - 2)
    zero = torch.zeros_like(y[:, 0])
    rhs = [zero] + list(interior.unbind(1)) + [zero]
    if n > 3:
        first = rhs[2] / (x[3] - x[1]) - rhs[1] / (x[2] - x[0])
        last = rhs[-2] / (x[-1] - x[-3]) - rhs[-3] / (x[-2] - x[-4])
        rhs[0] = first * (h[0] ** 2 / (x[3] - x[0]))
        rhs[-1] = last * (-h[-1] ** 2 / (x[-1] - x[-4]))

    for i in range(1, n):                                # forward elimination
        factor = h[i - 1] / diag[i - 1]
        diag[i] = diag[i] - factor * h[i - 1]
        rhs[i] = rhs[i] - factor * rhs[i - 1]

    rhs[-1] = rhs[-1] / diag[-1]                         # back substitution
    for i in range(n - 2, -1, -1):
        rhs[i] = (rhs[i] - h[i] * rhs[i + 1]) / diag[i]
    sigma = torch.stack(rhs, dim=1)                      # (m, n)

    b_last = slope[:, -1] + h[-1] * (sigma[:, -2] + 2.0 * sigma[:, -1])
    return torch.cat(
        [slope - h * (sigma[:, 1:] + 2.0 * sigma[:, :-1]), b_last[:, None]],
        dim=1)


def _hyman_filter(x: torch.Tensor, y: torch.Tensor,
                  b: torch.Tensor) -> torch.Tensor:
    """:func:`TransEHR2.hyman._hyman_filter`, in torch."""
    secant = torch.diff(y, dim=1) / torch.diff(x)
    before = torch.cat([secant[:, :1], secant], dim=1)
    after = torch.cat([secant, secant[:, -1:]], dim=1)
    bound = 3.0 * torch.minimum(before.abs(), after.abs())

    sign = torch.where(before * after > 0, after, b)
    return torch.where(
        sign >= 0,
        torch.minimum(b.clamp(min=0.0), bound),
        torch.maximum(b.clamp(max=0.0), -bound),
    )


def _spline_coef_conv(x: torch.Tensor, y: torch.Tensor,
                      b: torch.Tensor) -> tuple:
    """:func:`TransEHR2.hyman._spline_coef_conv`, in torch."""
    h = torch.diff(x)
    drop = -torch.diff(y, dim=1)
    b0, b1 = b[:, :-1], b[:, 1:]

    c = -(3.0 * drop + (2.0 * b0 + b1) * h) / h ** 2
    c_last = ((3.0 * drop[:, -1] + (b0[:, -1] + 2.0 * b1[:, -1]) * h[-1])
              / h[-1] ** 2)
    d = (2.0 * drop / h + b0 + b1) / h ** 2
    return (torch.cat([c, c_last[:, None]], dim=1),
            torch.cat([d, d[:, -1:]], dim=1))


def hyman_spline(x, y, xout, per_row: bool = False) -> torch.Tensor:
    """Evaluate ``splinefun(x, y, method = "hyman")`` at ``xout``, in torch.

    The arguments and checks are :func:`TransEHR2.hyman.hyman_spline`'s.
    The arithmetic runs in ``y``'s dtype and on its device when ``y`` is a
    floating-point tensor, and in float64 otherwise; ``x`` and ``xout``
    are brought to match.

    Args:
        x: (n,) knots, strictly ascending, ``n >= 3``.
        y: (m, n) monotone curves, one per row. A single (n,) curve comes
            back as a (1, p) row.
        xout: (p,) times to evaluate at. Values outside ``x`` extrapolate
            along the end cubics, as R's does.
        per_row: read row ``r`` at ``xout[r]`` alone, with ``p == m``.

    Returns:
        (m, p) interpolated values, or (m,) with ``per_row``.

    Raises:
        ValueError: if a row of ``y`` is not monotone, or ``n < 3``.
    """
    if not (torch.is_tensor(y) and y.is_floating_point()):
        y = torch.as_tensor(y, dtype=torch.float64)
    y = torch.atleast_2d(y)
    x = torch.as_tensor(x, dtype=y.dtype, device=y.device)
    xout = torch.as_tensor(xout, dtype=y.dtype, device=y.device)

    if x.ndim != 1 or x.numel() < 3:
        raise ValueError(f"x must be a 1-D sequence of at least 3 knots, "
                         f"got shape {tuple(x.shape)}")
    if y.shape[-1] != x.numel():
        raise ValueError(f"y has {y.shape[-1]} columns but x has "
                         f"{x.numel()} knots")

    step = torch.diff(y.detach(), dim=1)
    if not ((step >= 0.0).all(dim=1) | (step <= 0.0).all(dim=1)).all():
        raise ValueError("hyman interpolation needs monotone y; at least "
                         "one row turns back on itself")

    b = _hyman_filter(x, y, _fmm_coefficients(x, y))
    c, d = _spline_coef_conv(x, y, b)

    i = (torch.searchsorted(x.detach().contiguous(),
                            xout.detach().contiguous(), right=True) - 1)
    i = i.clamp(0, x.numel() - 1)
    dx = xout - x[i]
    if per_row:
        if tuple(xout.shape) != (y.shape[0],):
            raise ValueError(f"per_row needs one time per row: y has "
                             f"{y.shape[0]} rows, xout shape "
                             f"{tuple(xout.shape)}")
        r = torch.arange(y.shape[0], device=y.device)
        return y[r, i] + dx * (b[r, i] + dx * (c[r, i] + dx * d[r, i]))
    return y[:, i] + dx * (b[:, i] + dx * (c[:, i] + dx * d[:, i]))
