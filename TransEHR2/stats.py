"""Statistical tests for comparing cross-validated model performance.

Ported from the MIMIC-IV fork's ``reporting/stats.py``.

The comparison of interest is between two models evaluated on the same
set of cross-validation folds. A paired Student t test on the per-fold
differences is anti-conservative in this setting because the training
sets of the folds overlap, so the per-fold performance estimates are
positively correlated and their sample variance underestimates the
variance of the mean difference.

Nadeau and Bengio (2003) correct for this by inflating the variance of
the mean difference by the ratio of test set size to training set size.
Their corrected resampled t test uses

    t = mean(d) / sqrt((1 / k + n_test / n_train) * var(d, ddof=1))

on k - 1 degrees of freedom, where d holds the per-fold performance
differences.

The ratio is always the caller's. Here it is not the textbook
1 / (k - 1) of a plain k-fold split: ``fold0`` is reserved for tuning and
each fold's training patients lose a validation share, so the design's
n_test / n_train has to be worked out from the split and passed in. It
lives in the comparison config, beside the comparisons it applies to. A
default derived from the number of folds averaged would also be wrong
whenever a fold is missing, since each model still trained on the same
share of the data.

The same argument applies to the mean of one model's per-fold scores,
which is an average over the same overlapping training sets, so
``corrected_standard_error`` applies the identical inflation there. That
is what the reported standard errors are: an uncorrected sd / sqrt(k)
beside a corrected P value describes a narrower interval than the test
it sits next to.

Reference:
    Nadeau C, Bengio Y. Inference for the generalization error. Machine
    Learning 2003;52(3):239-281. doi:10.1023/A:1024068626366
"""

import math

import numpy as np
from scipy import stats


class TestResult:
    """Outcome of a single corrected resampled t test.

    Attributes:
        statistic: The t statistic, or NaN when the test is undefined.
        df: Degrees of freedom.
        p_value: Two-sided P value, or NaN when the test is undefined.
        mean_difference: Mean per-fold difference (experiment - control).
        note: ``None``, or a short explanation when the test could not be
            carried out.
    """

    __test__ = False  # not a pytest test class, despite the name

    __slots__ = ('statistic', 'df', 'p_value', 'mean_difference', 'note')

    def __init__(self, statistic, df, p_value, mean_difference, note=None):
        self.statistic = statistic
        self.df = df
        self.p_value = p_value
        self.mean_difference = mean_difference
        self.note = note

    def __repr__(self):
        return (f'TestResult(statistic={self.statistic!r}, df={self.df!r}, '
                f'p_value={self.p_value!r}, '
                f'mean_difference={self.mean_difference!r}, '
                f'note={self.note!r})')


def corrected_resampled_ttest(experiment, control, n_train_test_ratio):
    """Run the two-sided corrected resampled t test of Nadeau and Bengio.

    Args:
        experiment: Per-fold metric values for the experimental model.
        control: Per-fold metric values for the control model, in the
            same fold order.
        n_train_test_ratio: The value of n_test / n_train for the design.

    Returns:
        A :class:`TestResult`. Folds where either model has NaN are
        dropped, and the degrees of freedom are one less than the folds
        that remain.

    Raises:
        ValueError: If the two sequences differ in length or hold fewer
            than two folds with values in both.
    """
    a = np.asarray(experiment, dtype=np.float64)
    b = np.asarray(control, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(
            f'Fold counts differ: {a.shape[0]} vs {b.shape[0]}'
        )

    valid = ~(np.isnan(a) | np.isnan(b))
    d = a[valid] - b[valid]
    k = d.size

    if k < 2:
        raise ValueError(
            f'At least 2 folds with values in both models are required, '
            f'got {k}'
        )

    mean_d = float(np.mean(d))
    var_d = float(np.var(d, ddof=1))
    df = k - 1

    variance = (1.0 / k + n_train_test_ratio) * var_d

    if variance == 0.0:
        if mean_d == 0.0:
            # The two models produced identical values in every fold, so
            # there is nothing to test. A P value can never equal 1.
            return TestResult(
                float('nan'), df, float('nan'), mean_d,
                note='identical in every fold; test not applicable'
            )
        # A constant non-zero difference gives an infinite statistic.
        return TestResult(
            math.inf if mean_d > 0 else -math.inf, df, 0.0, mean_d,
            note='zero variance across folds'
        )

    t = mean_d / math.sqrt(variance)
    p = float(2.0 * stats.t.sf(abs(t), df))

    return TestResult(float(t), df, p, mean_d)


def benjamini_hochberg(p_values):
    """Adjust P values to control the false discovery rate.

    Implements the Benjamini-Hochberg step-up procedure, enforcing
    monotonicity of the adjusted values. NaN entries are passed through
    unchanged and are excluded from the number of tests.

    Args:
        p_values: Sequence of raw P values, possibly containing NaN.

    Returns:
        A numpy array of adjusted P values, in the input order, capped
        at 1.0.
    """
    p = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(p.shape, np.nan)

    finite = np.flatnonzero(~np.isnan(p))
    m = finite.size
    if m == 0:
        return adjusted

    values = p[finite]
    order = np.argsort(values, kind='stable')
    ranked = values[order]

    # q_(i) = min over j >= i of (m / j) * p_(j)
    scaled = ranked * m / np.arange(1, m + 1)
    running = np.minimum.accumulate(scaled[::-1])[::-1]
    running = np.clip(running, 0.0, 1.0)

    out = np.empty(m)
    out[order] = running
    adjusted[finite] = out
    return adjusted


def corrected_standard_error(values, n_train_test_ratio):
    """Standard error of a mean over overlapping cross-validation folds.

    The same inflation the corrected resampled t test applies, taken on one
    model's per-fold values rather than on a difference. Nadeau and Bengio's
    correction is a statement about averaging over splits whose training sets
    overlap, not about differences in particular: the folds' sample variance
    understates the variance of their mean whatever statistic is averaged.

        se = sqrt((1 / k + n_test / n_train) * var(values, ddof=1))

    Args:
        values: Per-fold metric values, possibly containing NaN.
        n_train_test_ratio: The value of n_test / n_train for the design.

    Returns:
        The corrected standard error, or NaN if fewer than two values are
        available.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    k = v.size
    if k < 2:
        return float('nan')
    return float(math.sqrt((1.0 / k + n_train_test_ratio)
                           * np.var(v, ddof=1)))
