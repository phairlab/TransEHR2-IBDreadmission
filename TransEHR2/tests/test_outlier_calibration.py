"""The three outlier rules, against distributions with planted defects.

Each probe builds a feature whose defect is known and asserts which rule
finds it -- and, where it matters, which rule does not. The rules exist
because they fail in different places, so a probe that only showed each one
working would not distinguish them from each other.

The arrays are written in section 4.4's layout, so ``collect`` is exercised
through the same reader the real cohort goes through rather than a stub.
"""

import numpy as np
import pickle
import pytest

from TransEHR2.data.preprocessing import feature_scale

from calibrate_outlier_filter import (
    Accumulator,
    acting_cuts,
    collect,
    emit_cuts,
    robust_log_cut,
    standardized_cut,
    standardized_extent,
    tail_gap_cut,
)

T = 50


def write_extracted(tmp_path, features):
    """A minimal ``extracted/`` holding one array per named feature.

    Args:
        tmp_path: Directory to write into.
        features: ``{name: 1-d array of observed values}``. Values are laid
            out row-major over an (n, T) axis wide enough to hold the longest
            one, and every position a feature has no value for gets indicator
            0 -- which is how a real cohort looks, features being observed at
            different rates.

    Returns:
        The directory written.
    """
    base = tmp_path / 'extracted'
    base.mkdir()
    names = list(features)
    longest = max(len(v) for v in features.values())
    n_episodes = -(-longest // T)

    indicators = np.zeros((n_episodes, T, len(names)), dtype=np.float32)
    for f, name in enumerate(names):
        values = np.zeros(n_episodes * T, dtype=np.float32)
        observed = np.asarray(features[name], dtype=np.float32)
        values[:observed.size] = observed
        flags = np.zeros(n_episodes * T, dtype=np.float32)
        flags[:observed.size] = 1.0
        indicators[:, :, f] = flags.reshape(n_episodes, T)
        np.save(base / f'val_numeric_values_{f}.npy',
                values.reshape(n_episodes, T, 1))
    np.save(base / 'val_numeric_indicators.npy', indicators)

    with open(base / 'metadata.pkl', 'wb') as f_out:
        pickle.dump({'numeric_feats': names,
                     'n_numeric_feats': len(names),
                     'numeric_feat_dims': [1] * len(names)}, f_out)
    return base


def body(size=2000, centre=1.0, spread=0.15, seed=0):
    """A clean lognormal body: no gaps, nothing to cut."""
    rng = np.random.default_rng(seed)
    return centre * 10.0 ** rng.normal(0.0, spread, size)


# --- the tail gap rule ------------------------------------------------

def test_the_tail_gap_finds_a_detached_cluster():
    """A decimal slip lands three decades out with nothing in between, which
    is the case the rule was written for. The cut sits at the geometric
    midpoint of the gap, not on either end of it: a value round-tripped
    through log10 and back does not always compare equal to itself, so a cut
    placed on the datum can fail to exclude it."""
    clean = body()
    values = np.concatenate([clean, [1000.0, 1200.0]])
    low, high = tail_gap_cut(values, gap=0.5, quantile=0.999, min_fold=3.0)
    assert high == pytest.approx(np.sqrt(clean.max() * 1000.0))
    assert clean.max() < high < 1000.0
    assert not np.isfinite(low)


def test_the_tail_gap_is_bridged_by_a_chain_of_values():
    """Its stated weakness, made explicit: values spaced under --gap apart
    all the way out leave no jump anywhere, so the rule finds nothing even
    though the far end is plainly wrong."""
    chain = 3.0 * 2.0 ** np.arange(1, 10)  # 0.30 decades per step
    values = np.concatenate([body(), chain])
    low, high = tail_gap_cut(values, gap=0.5, quantile=0.999, min_fold=3.0)
    assert not np.isfinite(high)


def test_the_robust_spread_catches_what_the_chain_bridged():
    """The rule that does not depend on a gap existing. It must cut into the
    chain while leaving the body whole, so both bounds are asserted: a rule
    that cut everything above the median would also satisfy 'the chain is
    gone'."""
    clean = body()
    chain = 3.0 * 2.0 ** np.arange(1, 10)
    values = np.concatenate([clean, chain])
    low, high, scale = robust_log_cut(values, k=8.0)
    assert np.isfinite(scale) and scale > 0
    assert clean.max() < high < chain[2]        # keeps the body, cuts from 24
    assert low < clean.min()


def test_neither_log_rule_cuts_a_clean_body():
    """No rule may cut a feature with no defect: a false positive here
    removes real data from every episode that recorded the feature."""
    values = body(size=5000)
    assert tail_gap_cut(values, 0.5, 0.999, 3.0) == (-np.inf, np.inf)
    low, high, _ = robust_log_cut(values, k=8.0)
    assert low < values.min() and high > values.max()


def test_a_feature_with_too_few_values_is_never_cut():
    """Below MIN_VALUES a quantile is noise wearing a threshold's clothing."""
    values = np.concatenate([body(size=50), [1e6]])
    assert tail_gap_cut(values, 0.5, 0.999, 3.0) == (-np.inf, np.inf)
    assert robust_log_cut(values, 8.0)[:2] == (-np.inf, np.inf)


# --- the standardized magnitude rule ----------------------------------

def test_the_standardized_rule_separates_impossible_from_destabilizing():
    """The distinction the rule exists to draw, on one cut rather than two.

    Both planted values are clinically impossible. Only the sentinel's
    standardized magnitude is large, and only the sentinel destabilizes a
    loss computed over the feature -- so the rule must keep the one and cut
    the other, and one cut has to do both.
    """
    tight = body(size=5000, centre=170.0, spread=0.02)  # a height, in cm
    low, high, centre, scale = standardized_cut(
        np.concatenate([tight, [419.0, 1e6]]), u=20.0
    )
    assert 419.0 < high < 1e6
    # And it is the span, not the absolute size, that decides: 419 survives
    # because it is under 20 spans out, and the span is ~26 cm.
    assert standardized_extent(419.0, centre, scale) < 20.0
    assert standardized_extent(1e6, centre, scale) > 20.0


def test_the_standardized_scale_is_the_divisor_the_loader_uses():
    """standardize_feats measures over the L2 norm, so the scale is over |v|,
    and it is feature_scale's rather than a bare 5th-to-95th range. Asserted
    against the function the loader calls, not against a reimplementation of
    it, so the two cannot drift apart."""
    values = np.concatenate([body(size=4000), -body(size=1000, seed=1)])
    _, _, centre, scale = standardized_cut(values, u=20.0)
    assert scale == pytest.approx(feature_scale(np.abs(values)))
    assert centre == pytest.approx(float(np.median(values)))


def test_a_detection_limit_feature_still_gets_a_cut():
    """The case the widening exists for. Over 90% of values identical means a
    degenerate 5th-to-95th range, which would leave the rule with no scale and
    so no cut at all -- on exactly the feature whose tail is the signal."""
    # Above 95% at the limit, or the 95th percentile lands in the tail and the
    # range is not degenerate at all -- which is the fixture bug this comment
    # exists to stop being reintroduced.
    at_limit = np.full(4900, 0.01)
    tail = body(size=100, centre=0.5, seed=4)
    values = np.concatenate([at_limit, tail])

    p5, p95 = np.percentile(np.abs(values), [5, 95])
    assert p95 - p5 == 0          # the range the widening replaces

    _, high, _, scale = standardized_cut(values, u=20.0)
    assert np.isfinite(scale) and scale > 0
    assert np.isfinite(high)


# --- zeros and negatives ----------------------------------------------

def test_zeros_and_negatives_are_counted_and_left_to_the_third_rule():
    """No log rule can see them, and charging a feature's sentinel share
    against a log cut would read the rule as removing what it never saw."""
    acc = Accumulator()
    acc.add(np.concatenate([body(size=500), np.zeros(30), -np.ones(7)]))
    assert acc.zeros == 30
    assert acc.negatives == 7
    assert acc.positive_count == 500

    # The same cuts, charged both ways.
    positive_only, _ = acc.beyond(-np.inf, np.inf, positive_only=True)
    everything, _ = acc.beyond(-1e9, 1e9, positive_only=False)
    assert positive_only == 0 and everything == 0
    # A low cut just under the body counts exactly the zeros and negatives,
    # so the two ways of charging a cut differ by precisely what the log
    # rules cannot see.
    floor = float(body(size=500).min()) * 0.99
    charged, _ = acc.beyond(floor, np.inf, positive_only=False)
    spared, _ = acc.beyond(floor, np.inf, positive_only=True)
    assert charged == 37
    assert spared == 0


# --- the reader and the cut table -------------------------------------

def test_collect_reads_the_extracted_layout(tmp_path):
    """Through the real reader: indicators say which positions are observed,
    and an unobserved zero must not be counted as a sentinel."""
    base = write_extracted(tmp_path, {
        'CRP': body(size=1500),
        'ALT': np.concatenate([body(size=1200, seed=2), np.zeros(11)]),
    })
    names, accumulators = collect(str(base), block=8, seed=0)
    assert names == ['CRP', 'ALT']
    assert accumulators[0].count == 1500
    assert accumulators[0].zeros == 0
    assert accumulators[1].count == 1211
    assert accumulators[1].zeros == 11


def test_collect_refuses_a_wider_numeric_feature(tmp_path):
    """A cut on a multi-column feature would have to say which column."""
    base = write_extracted(tmp_path, {'CRP': body(size=200)})
    with open(base / 'metadata.pkl', 'rb') as f_in:
        metadata = pickle.load(f_in)
    metadata['numeric_feat_dims'] = [3]
    with open(base / 'metadata.pkl', 'wb') as f_out:
        pickle.dump(metadata, f_out)
    with pytest.raises(ValueError, match='width-1'):
        collect(str(base), block=8, seed=0)


def test_the_cut_table_names_only_the_features_it_cuts(tmp_path):
    """A feature absent from the table is left alone, so an empty table and
    no table mean the same thing."""
    clean, dirty = Accumulator(), Accumulator()
    clean.add(body(size=500))
    dirty.add(np.concatenate([body(size=500, seed=7), [1e6]]))

    path = tmp_path / 'numeric_cuts.csv'
    written = emit_cuts(str(path), ['CRP', 'ALT'], [clean, dirty],
                        [(0.01, 900.0), (0.01, 900.0)], 'standardized', False)
    assert written == 1
    text = path.read_text()
    assert 'FEATURE,LOW,HIGH' in text
    assert 'ALT,0.01,900.0' in text
    # CRP has the same cuts and nothing outside them, so it is a no-op and
    # must not appear: the table is read as what the filter acts on.
    assert 'CRP' not in text


def test_a_band_that_removes_nothing_is_not_a_cut(tmp_path):
    """The standardized rule always returns a finite band, so 'finite' is not
    a detection. Without this, every numeric feature would enter the table
    whether or not the filter touches it."""
    clean = Accumulator()
    clean.add(body(size=5000))
    low, high, _, _ = standardized_cut(clean.body_sample(), u=20.0)
    assert np.isfinite(low) and np.isfinite(high)
    assert list(acting_cuts(['CLEAN'], [clean], [(low, high)], False)) == []


def test_the_two_objectives_disagree_on_a_tightly_held_feature():
    """Why the standardized rule earns its place beside the log rules.

    On a feature whose body spans little in log terms -- a height in cm, a
    temperature -- the two objectives part company. 419 cm is clinically
    impossible, so the log rules remove it; but it standardizes to single
    digits, so removing it buys a loss computed over the feature nothing.
    The sentinel is what destabilizes training, and only the standardized
    rule draws the line between them.

    Neither rule cuts into the genuine body here, which is the other half of
    the claim: the disagreement is about a value that is wrong, not about
    real pathology.
    """
    tight = body(size=5000, centre=170.0, spread=0.02)
    values = np.concatenate([tight, [419.0, 1e6]])
    acc = Accumulator()
    acc.add(values)

    _, robust_high, _ = robust_log_cut(acc.tail_sample(), k=8.0)
    _, standardized_high, centre, scale = standardized_cut(
        acc.body_sample(), u=20.0
    )

    assert robust_high < 419.0 < standardized_high < 1e6
    # Both leave the body whole.
    assert acc.surviving(-np.inf, robust_high)[1] == pytest.approx(
        float(tight.max()))
    assert acc.surviving(-np.inf, standardized_high)[1] == pytest.approx(419.0)
    # And what the disagreement is worth to the loss: 419 is a handful of
    # spans out, the sentinel is tens of thousands.
    assert abs(standardized_extent(419.0, centre, scale)) < 20.0
    assert abs(standardized_extent(1e6, centre, scale)) > 1000.0
