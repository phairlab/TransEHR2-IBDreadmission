"""Probes for the scale a numeric feature's values are divided by.

Standardization divides by the 5th-95th percentile range of the observed magnitudes. That range
collapses to zero whenever a feature's distribution is concentrated enough for both percentiles
to land on the same value, and the caller then zeroes the feature to avoid dividing by zero --
which leaves the occurrence indicator standing and discards every value the feature holds.

Two real cases upstream: a laboratory result reported at a detection limit, where over 90% of
values are identical and the informative mass is the tail above it; and a two-level assessment
coded 0, abnormal in a few percent of cases. Both are features carried on the value stream
precisely so their magnitudes reach the encoder, so a degenerate range has to widen rather than
collapse. A feature that really is constant still gets no scale, because its indicator already
says everything it has to say.
"""

import numpy as np

from TransEHR2.data.preprocessing import feature_scale


def test_an_ordinary_spread_uses_the_five_ninetyfive_range():
    """The estimator of record, unchanged wherever it is non-degenerate."""
    norms = np.arange(0.0, 100.0)
    expected = float(np.percentile(norms, 95) - np.percentile(norms, 5))
    assert feature_scale(norms) == expected
    assert expected > 0


def test_a_two_level_assessment_keeps_a_usable_scale():
    """Capillary refill rate: 0 with a few percent at 1, so p5 and p95 are both 0."""
    norms = np.zeros(1000)
    norms[:40] = 1.0
    assert np.percentile(norms, 5) == np.percentile(norms, 95) == 0.0
    # The 1st-99th range separates the levels, so the abnormal reading survives.
    assert feature_scale(norms) == 1.0


def test_a_result_at_a_detection_limit_keeps_a_usable_scale():
    """Troponin T: over 90% at the reporting limit, with an informative tail above it."""
    norms = np.full(1000, 0.01)
    norms[:20] = np.linspace(0.5, 8.0, 20)
    assert np.percentile(norms, 5) == np.percentile(norms, 95) == 0.01
    scale = feature_scale(norms)
    assert scale > 0
    # Scaled by the 1st-99th range rather than the full span, so a tail value stays order one
    # rather than being compressed by the largest outlier.
    assert scale < norms.max() - norms.min()


def test_the_full_span_is_the_last_resort():
    """When even the outermost percentiles coincide, the span is all that is left."""
    norms = np.zeros(1000)
    norms[0] = 5.0
    assert np.percentile(norms, 1) == np.percentile(norms, 99) == 0.0
    assert feature_scale(norms) == 5.0


def test_a_constant_feature_gets_no_scale():
    """Nothing to recover: the indicator already carries the occurrence."""
    assert feature_scale(np.full(500, 0.01)) == 0.0
    assert feature_scale(np.zeros(500)) == 0.0


def test_no_observations_gets_no_scale():
    assert feature_scale(np.array([])) == 0.0


def test_the_widening_order_favours_typical_values():
    """The outermost percentiles come before the span, and it matters.

    A heavy tail sets the span, so scaling by it would compress every typical value toward
    zero. The 1st-99th range is bounded by the bulk instead.
    """
    norms = np.full(10_000, 2.0)
    norms[:200] = 3.0
    norms[0] = 10_000.0
    assert feature_scale(norms) == 1.0


# ------------------------------------------------------------------------------------------
# Through standardize_feats, which is what decides whether the values survive
# ------------------------------------------------------------------------------------------

class _Dims:
    n_numeric_feats = 3


def _arrays():
    """Three features: a two-level assessment, a result at a limit with a tail, a constant."""
    n_episodes, n_steps = 200, 5
    flat = n_episodes * n_steps
    binary = np.zeros(flat)
    binary[:40] = 1.0
    limit = np.full(flat, 0.01)
    limit[:20] = np.linspace(0.5, 8.0, 20)
    constant = np.full(flat, 0.01)

    indicators = np.ones((n_episodes, n_steps, 3), dtype=np.float32)
    values = [np.zeros((n_episodes, n_steps, 1), dtype=np.float32) for _ in range(3)]
    for index, column in enumerate((binary, limit, constant)):
        values[index][:, :, 0] = column.reshape(n_episodes, n_steps)
    return {'val_numeric_indicators': indicators, 'val_numeric_values': values}


def _scaled(arrays, stats_path, rows=None):
    """The three features as ``__getitem__`` would serve them.

    Standardization is applied at load time here, not in place at extraction
    (section 3): the arrays are cohort-wide and the statistics are per fold, so
    one array cannot carry one fold's scaling. That makes the round trip through
    the npz the thing to probe rather than an incidental detail of it.
    """
    from TransEHR2.data.preprocessing import standardize_feats

    n_episodes = arrays['val_numeric_indicators'].shape[0]
    if rows is None:
        rows = np.arange(n_episodes)
    standardize_feats(arrays, _Dims(), rows, stats_path)

    stats = dict(np.load(stats_path))
    means = stats['means']
    span = (stats['scale'] if 'scale' in stats
            else stats['p95'] - stats['p5'])

    out = []
    for f, values in enumerate(arrays['val_numeric_values']):
        column = np.array(values, dtype=np.float32)
        if span[f] == 0:
            column[:] = 0.0
        else:
            column = (column - means[f]) / span[f]
        out.append(column)
    return out


def test_degenerate_range_features_keep_their_values(tmp_path):
    binary, limit, constant = _scaled(_arrays(), str(tmp_path / 'stats.npz'))
    # Both levels survive, distinguishable and of order one.
    assert len(np.unique(binary)) == 2
    assert np.ptp(binary) == 1.0
    # The tail survives as a graded signal rather than collapsing.
    assert len(np.unique(limit)) > 2
    assert np.isfinite(limit).all()
    # A constant feature is still zeroed: the indicator carries its occurrence.
    assert np.all(constant == 0)


def test_the_saved_statistics_reproduce_the_training_standardization(tmp_path):
    """A held-out partition standardizes from the training fold's npz.

    The fork writes one npz per fold over that fold's training rows and applies
    it to every partition, so the scaling a validation episode receives is the
    training fold's -- which is the property this checks, on the same values.
    """
    path = str(tmp_path / 'stats.npz')
    train = _scaled(_arrays(), path)

    stats = dict(np.load(path))
    span = stats['scale']
    held_out = []
    for f, values in enumerate(_arrays()['val_numeric_values']):
        column = np.array(values, dtype=np.float32)
        if span[f] == 0:
            column[:] = 0.0
        else:
            column = (column - stats['means'][f]) / span[f]
        held_out.append(column)

    for index in range(3):
        assert np.allclose(held_out[index], train[index])


def test_statistics_written_before_scale_existed_keep_their_meaning(tmp_path):
    """An npz with no ``scale`` reproduces the range it replaced.

    All three features here have a degenerate 5th-95th range, so the old
    statistics zero all three -- the behaviour this change exists to end. The
    fallback has to reproduce it exactly, or an existing extraction would
    silently change meaning rather than being re-run.
    """
    current = str(tmp_path / 'current.npz')
    _scaled(_arrays(), current)
    saved = np.load(current)
    assert np.all(saved['p95'] - saved['p5'] == 0)

    legacy = str(tmp_path / 'legacy.npz')
    np.savez(legacy, means=saved['means'], p5=saved['p5'], p95=saved['p95'])

    stats = dict(np.load(legacy))
    assert 'scale' not in stats
    span = stats['p95'] - stats['p5']
    for f in range(3):
        assert span[f] == 0
