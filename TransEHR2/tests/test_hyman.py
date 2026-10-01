"""The monotone cubic port, checked against the R it was ported from.

``TransEHR2/hyman.py`` is a transcription of ``splinefun(method =
"hyman")``. Nothing about it is independently derivable, so the gate is
agreement with R: ``fixtures/hyman_reference.npz`` holds curves that R
produced, and the port has to reproduce them. The fixture is rebuilt by
``fixtures/make_hyman_reference.py``, which needs Rscript; the tests do
not.

Tolerance is 1e-12 absolute. The two implementations do the same
arithmetic in the same order, so the gap is rounding in the last couple
of ulps -- it measured 8.3e-15 when the port was written. A failure at
1e-12 is a different algorithm, not a different compiler.

The properties below are checked separately from the fixture because
agreement with R would also be agreement with a shared mistake: that the
curve passes through its knots, and that it never turns back on itself,
are the two things the caller actually relies on.
"""

import numpy as np
import pytest

from pathlib import Path

from TransEHR2.hyman import hyman_spline


REFERENCE = Path(__file__).parent / 'fixtures' / 'hyman_reference.npz'
CASES = ('cif6', 'cif3', 'surv6')


@pytest.fixture(scope='module')
def reference():
    with np.load(REFERENCE) as data:
        return {name: data[name] for name in data.files}


@pytest.mark.parametrize('case', CASES)
def test_port_reproduces_r(reference, case):
    """The gate: same knots, same curves, same answer as splinefun."""
    got = hyman_spline(reference[f'{case}_x'].ravel(),
                       reference[f'{case}_y'],
                       reference[f'{case}_xout'].ravel())
    assert got.shape == reference[f'{case}_fit'].shape
    assert np.abs(got - reference[f'{case}_fit']).max() < 1e-12


@pytest.mark.parametrize('case', CASES)
def test_curves_stay_monotone(reference, case):
    """What the filter is for: no interval may double back."""
    got = hyman_spline(reference[f'{case}_x'].ravel(),
                       reference[f'{case}_y'],
                       reference[f'{case}_xout'].ravel())
    step = np.diff(got, axis=1)
    rising = reference[f'{case}_y'][:, -1] >= reference[f'{case}_y'][:, 0]
    assert np.all(step[rising] >= -1e-12)
    assert np.all(step[~rising] <= 1e-12)


def test_the_curve_passes_through_its_knots(reference):
    """An interpolant, not a smoother."""
    x = reference['cif6_x'].ravel()
    y = reference['cif6_y']
    assert hyman_spline(x, y, x) == pytest.approx(y, abs=1e-12)


def test_a_flat_curve_stays_flat():
    """Zero mass everywhere leaves no room for a cubic to wander."""
    x = np.array([0.0, 30.0, 90.0, 365.0])
    out = np.linspace(0.0, 365.0, 101)
    assert hyman_spline(x, np.zeros((1, 4)), out) == pytest.approx(
        np.zeros((1, 101)))


def test_a_single_curve_is_accepted_as_one_row():
    x = np.array([0.0, 30.0, 90.0, 365.0])
    y = np.array([0.0, 0.1, 0.3, 0.5])
    out = np.array([0.0, 15.0, 365.0])
    assert hyman_spline(x, y, out).shape == (1, 3)


def test_every_curve_is_the_same_as_fitting_it_alone(reference):
    """Batching is an optimization, not a different estimator.

    The tridiagonal matrix is shared across curves because they share
    knots; this is the check that sharing it changes nothing.
    """
    x = reference['cif6_x'].ravel()
    y = reference['cif6_y']
    out = reference['cif6_xout'].ravel()
    batched = hyman_spline(x, y, out)
    for i in (0, 7, len(y) // 2, len(y) - 1):
        assert hyman_spline(x, y[i], out)[0] == pytest.approx(batched[i])


def test_non_monotone_input_is_refused():
    x = np.array([0.0, 30.0, 90.0, 365.0])
    y = np.array([[0.0, 0.5, 0.2, 0.9]])
    with pytest.raises(ValueError, match='monotone'):
        hyman_spline(x, y, np.array([15.0]))


def test_too_few_knots_is_refused():
    with pytest.raises(ValueError, match='at least 3 knots'):
        hyman_spline(np.array([0.0, 365.0]), np.array([[0.0, 1.0]]),
                     np.array([100.0]))
