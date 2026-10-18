"""The corrected resampled t test, the BH adjustment and the corrected SE.

Ported from the MIMIC-IV fork's ``tests/test_reporting.py``. The ratio is a
required argument here, so the tests that leaned on the 1 / (k - 1) default
pass it explicitly, and the one that checked the default itself is replaced by
one that checks the design ratio survives a lost fold.
"""

import math

import numpy as np
import pytest

from TransEHR2.stats import (
    benjamini_hochberg,
    corrected_resampled_ttest,
    corrected_standard_error,
)


# ------------------------------------------------------------------
# Corrected resampled t test
# ------------------------------------------------------------------

def test_correction_inflates_the_standard_error_as_theory_predicts():
    """At k folds the statistic shrinks by sqrt((1/k + 1/(k-1)) / (1/k))."""
    rng = np.random.default_rng(0)
    for k in (3, 5, 10):
        a = rng.normal(0.85, 0.01, k)
        b = rng.normal(0.83, 0.01, k)
        result = corrected_resampled_ttest(a, b, 1 / (k - 1))
        d = a - b
        paired = d.mean() / (d.std(ddof=1) / math.sqrt(k))
        expected = math.sqrt((1 / k + 1 / (k - 1)) / (1 / k))
        assert result.df == k - 1
        assert paired / result.statistic == pytest.approx(expected, abs=1e-12)


def test_the_p_value_is_two_sided():
    a = [0.86, 0.85, 0.85, 0.84, 0.84]
    b = [0.84, 0.83, 0.85, 0.82, 0.83]
    forward = corrected_resampled_ttest(a, b, 0.25)
    backward = corrected_resampled_ttest(b, a, 0.25)
    assert backward.statistic == pytest.approx(-forward.statistic)
    assert backward.p_value == pytest.approx(forward.p_value)
    from scipy import stats
    assert forward.p_value == pytest.approx(
        2 * stats.t.sf(abs(forward.statistic), 4))


def test_the_test_is_paired_not_unpaired():
    """Reordering one arm's folds must change the result."""
    a = [0.86, 0.85, 0.85, 0.84, 0.84]
    b = [0.84, 0.83, 0.85, 0.82, 0.83]
    original = corrected_resampled_ttest(a, b, 0.25).statistic
    shuffled = corrected_resampled_ttest(a, list(reversed(b)), 0.25).statistic
    assert original != pytest.approx(shuffled)


def test_identical_arms_give_no_p_value():
    """Prevalence is identical in every column; P can never equal 1."""
    result = corrected_resampled_ttest([0.17] * 5, [0.17] * 5, 0.25)
    assert math.isnan(result.p_value)
    assert result.note


def test_constant_non_zero_difference_is_reported_as_highly_significant():
    result = corrected_resampled_ttest([0.2] * 5, [0.1] * 5, 0.25)
    assert result.p_value == 0.0
    assert result.statistic == math.inf


def test_nan_folds_are_dropped_and_reduce_the_df():
    result = corrected_resampled_ttest(
        [0.86, float('nan'), 0.85, 0.84, 0.84],
        [0.84, 0.83, 0.85, 0.82, 0.83],
        0.25
    )
    assert result.df == 3


def test_mismatched_fold_counts_are_rejected():
    with pytest.raises(ValueError):
        corrected_resampled_ttest([0.1, 0.2, 0.3], [0.1, 0.2], 0.25)


def test_too_few_folds_are_rejected():
    with pytest.raises(ValueError):
        corrected_resampled_ttest([0.1], [0.2], 0.25)


def test_a_reduced_fold_set_keeps_the_design_ratio():
    """Three folds averaged, but each model still trained on the same share of
    the data, so the ratio stays the design's and only the df falls."""
    a = [0.835, 0.837, 0.833]
    b = [0.830, 0.831, 0.829]
    design = corrected_resampled_ttest(a, b, 0.25)
    from_survivors = corrected_resampled_ttest(a, b, 1 / 2)
    assert design.df == from_survivors.df == 2
    assert abs(design.statistic) > abs(from_survivors.statistic)
    assert design.p_value < from_survivors.p_value


# ------------------------------------------------------------------
# Benjamini-Hochberg
# ------------------------------------------------------------------

def test_bh_matches_r_p_adjust():
    """Values verified against R's p.adjust(method = "BH")."""
    assert benjamini_hochberg([0.01, 0.02, 0.03, 0.04, 0.05]) == \
        pytest.approx([0.05] * 5)
    assert benjamini_hochberg([0.001, 0.008, 0.039, 0.041, 0.042]) == \
        pytest.approx([0.005, 0.02, 0.042, 0.042, 0.042])
    # Order of the input must not matter.
    assert benjamini_hochberg([0.042, 0.001, 0.041, 0.008, 0.039]) == \
        pytest.approx([0.042, 0.005, 0.042, 0.02, 0.042])


def test_bh_is_monotone():
    rng = np.random.default_rng(3)
    raw = rng.random(40)
    adjusted = benjamini_hochberg(raw)
    order = np.argsort(raw)
    assert np.all(np.diff(adjusted[order]) >= -1e-12)


def test_bh_never_exceeds_one():
    assert np.all(benjamini_hochberg([0.9, 0.95, 0.99]) <= 1.0)


def test_bh_passes_nan_through_and_excludes_it_from_the_family():
    adjusted = benjamini_hochberg([0.01, 0.02, float('nan')])
    assert math.isnan(adjusted[2])
    # Two tests, not three: 0.01 * 2 / 1 = 0.02.
    assert adjusted[0] == pytest.approx(0.02)


# ------------------------------------------------------------------
# Corrected standard error
# ------------------------------------------------------------------

def test_the_reported_se_is_the_corrected_one():
    """What goes in a table cell. sd/sqrt(k) describes a narrower interval than
    the test printed beside it."""
    values = [0.86, 0.85, 0.85, 0.84, 0.84]
    k = len(values)
    expected = math.sqrt((1 / k + 0.25) * np.var(values, ddof=1))
    assert corrected_standard_error(values, 0.25) == pytest.approx(expected)


def test_the_correction_inflates_the_se_by_one_and_a_half_at_five_folds():
    values = [0.86, 0.85, 0.85, 0.84, 0.84]
    uncorrected = np.std(values, ddof=1) / math.sqrt(len(values))
    assert corrected_standard_error(values, 0.25) / uncorrected == \
        pytest.approx(1.5)


def test_the_se_and_the_test_agree_on_the_ratio_they_were_given():
    """Both take the ratio from the caller, so a cell's SE and its P cannot be
    computed on different assumptions about the design."""
    a = [0.84, 0.83, 0.85, 0.82, 0.83]
    b = [0.80, 0.81, 0.79, 0.80, 0.80]
    d = np.array(a) - np.array(b)
    se = corrected_standard_error(d, 0.25)
    test = corrected_resampled_ttest(a, b, 0.25)
    assert test.statistic == pytest.approx(np.mean(d) / se)


def test_a_larger_ratio_gives_a_larger_se():
    values = [0.86, 0.85, 0.85, 0.84, 0.84]
    assert corrected_standard_error(values, 1 / 2) > \
        corrected_standard_error(values, 1 / 4)


def test_the_corrected_se_drops_nan_and_needs_two_values():
    assert corrected_standard_error([0.8, float('nan'), 0.9], 0.25) == \
        pytest.approx(corrected_standard_error([0.8, 0.9], 0.25))
    assert math.isnan(corrected_standard_error([0.8], 0.25))
    assert math.isnan(corrected_standard_error([], 0.25))
