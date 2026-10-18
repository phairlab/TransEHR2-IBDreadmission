"""The torch Hyman spline, checked against the NumPy port and for gradients.

Both implementations are kept: the NumPy one as the port pinned to R, the
torch one as what ``survival.interpolate_cif`` and ``interpolate_tail``
read curves through, so that a gradient can be traced through them. The
first tests here are what stops them drifting apart. They
do the same arithmetic in the same order, and agreed bit for bit when the
torch one was written, so the check against NumPy is exact rather than a
tolerance; the check against R's own curves is the 1e-12 of
``test_hyman.py``.

The gradient tests run on synthetic curves kept away from the filter's
switching points. The curve is piecewise linear in its knot values, with
kinks where a slope meets its bound or zero; a finite difference that
straddles one measures neither side, so curves whose slopes sit too near
one are dropped before testing, and the batch is checked to still contain
clamped slopes of both kinds -- otherwise the test would pass without the
filter's gradient ever having been exercised.
"""

import numpy as np
import pytest
import torch

from pathlib import Path

from TransEHR2.hyman import _fmm_coefficients, hyman_spline as numpy_spline
from TransEHR2.hyman_torch import hyman_spline


REFERENCE = Path(__file__).parent / 'fixtures' / 'hyman_reference.npz'
CASES = ('cif6', 'cif3', 'surv6')
KNOTS = np.array([0.0, 30.0, 60.0, 90.0, 365.0, 1095.0, 1825.0])


@pytest.fixture(scope='module')
def reference():
    with np.load(REFERENCE) as data:
        return {name: data[name] for name in data.files}


@pytest.mark.parametrize('case', CASES)
def test_torch_matches_numpy_and_r(reference, case):
    x = reference[f'{case}_x'].ravel()
    y = reference[f'{case}_y']
    out = reference[f'{case}_xout'].ravel()
    got = hyman_spline(x, torch.tensor(y), out)
    assert got.dtype == torch.float64
    got = got.numpy()
    np.testing.assert_array_equal(got, numpy_spline(x, y, out))
    assert np.abs(got - reference[f'{case}_fit']).max() < 1e-12


def test_per_row_matches_numpy(reference):
    x = reference['cif6_x'].ravel()
    y = reference['cif6_y']
    times = np.linspace(0.0, x[-1], len(y))
    got = hyman_spline(x, torch.tensor(y), times, per_row=True).numpy()
    np.testing.assert_array_equal(
        got, numpy_spline(x, y, times, per_row=True))


def test_non_monotone_input_is_refused():
    y = torch.tensor([[0.0, 0.5, 0.2, 0.9]], dtype=torch.float64)
    with pytest.raises(ValueError, match='monotone'):
        hyman_spline(KNOTS[:4], y, np.array([15.0]))


def _filter_margins(x, y):
    """Distance of each unfiltered slope from the filter's kinks.

    For a strictly monotone curve the filter clamps each FMM slope into
    ``[0, bound]`` (or its mirror); the kinks are at zero and at the
    bound. Returned relative to the bound, with the clamp each slope hit.
    """
    b = _fmm_coefficients(x, y)
    secant = np.diff(y, axis=1) / np.diff(x)
    before = np.concatenate([secant[:, :1], secant], axis=1)
    after = np.concatenate([secant, secant[:, -1:]], axis=1)
    bound = 3.0 * np.minimum(np.abs(before), np.abs(after))
    b = b * np.sign(after)                     # mirror falling curves
    margin = np.minimum(np.abs(b), np.abs(b - bound)) / bound
    return margin.min(axis=1), (b > bound).sum(), (b < 0).sum()


@pytest.fixture(scope='module')
def synthetic():
    """Rising CIFs and their falling tails, away from the filter's kinks."""
    rng = np.random.default_rng(20261016)
    inc = (rng.dirichlet(np.ones(7), size=60)[:, :6]
           * rng.uniform(0.05, 0.9, (60, 1)))
    cif = np.concatenate([np.zeros((60, 1)), np.cumsum(inc, axis=1)],
                         axis=1)
    y = np.concatenate([cif, cif[:, -1:] - cif])
    margin, _, _ = _filter_margins(KNOTS, y)
    y = y[(margin > 1e-2) & (np.abs(np.diff(y, axis=1)).min(axis=1) > 1e-3)]
    _, above, below = _filter_margins(KNOTS, y)
    assert above > 0 and below > 0, 'no clamped slopes left to test'
    assert (y[:, -1] > y[:, 0]).any() and (y[:, -1] < y[:, 0]).any()
    return y


def test_gradcheck_through_the_filter(synthetic):
    y = torch.tensor(synthetic[:20], requires_grad=True)
    out = torch.linspace(0.0, 1825.0, 25, dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda y: hyman_spline(KNOTS, y, out), (y,))
    times = torch.linspace(5.0, 1800.0, 20, dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda y: hyman_spline(KNOTS, y, times, per_row=True), (y,))


def test_gradient_matches_finite_differences_of_the_numpy_port(synthetic):
    """Autograd against central differences taken on the other port."""
    y = synthetic
    out = np.linspace(0.0, 1825.0, 61)
    weight = np.random.default_rng(1).normal(size=(len(y), out.size))

    y_t = torch.tensor(y, requires_grad=True)
    (hyman_spline(KNOTS, y_t, out) * torch.tensor(weight)).sum().backward()

    step = 1e-7
    fd = np.empty_like(y)
    for r, k in np.ndindex(*y.shape):
        up, down = y.copy(), y.copy()
        up[r, k] += step
        down[r, k] -= step
        fd[r, k] = ((weight * (numpy_spline(KNOTS, up, out)
                               - numpy_spline(KNOTS, down, out))).sum()
                    / (2.0 * step))
    np.testing.assert_allclose(y_t.grad.numpy(), fd, rtol=1e-6, atol=1e-8)
