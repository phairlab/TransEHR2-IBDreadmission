#!/usr/bin/env python
"""What an algorithmic outlier filter would remove, measured on the cohort.

The numeric features are LAB test results, and ``prepare_RMT23345_LAB.R``
already curates each test's *units* by hand across ~1,700 lines. Curating
plausible *ranges* the same way does not scale to 94 tests, so this measures
three rules that need no per-feature clinical knowledge:

    tail gap    Sort the log values and cut at the first jump wider than
                ``--gap`` decades. A genuine tail is continuous however far it
                runs; decimal slips, unit slips and sentinels sit in a separate
                cluster orders of magnitude out, and the cut adapts per
                feature. Its weakness is that one erroneous value between the
                body and a far one bridges the gap meant to separate them.

    robust      Cut at ``--z`` robust spreads from the median, in log space,
    spread      with the spread measured from the feature itself. Nothing
                bridges anything, and the threshold means the same for a
                feature whose body spans a factor of two as for one spanning
                three orders of magnitude.

    standardized  Cut beyond ``--u`` units of the span ``standardize_feats``
    magnitude     divides by. The other two rules ask whether a value is
                  clinically possible, which is not the question the model
                  needs answered: an impossible value that standardizes to
                  single digits costs the loss nothing, while a sentinel that
                  standardizes to thousands dominates every step it appears in.

**Nothing is applied here.** This reports what each rule would do, so the
thresholds are set from evidence before a cut table is emitted.

**No pipeline stage is re-run.** The extracted arrays are read memory-mapped
and never written, so calibrating costs one pass over ``extracted/`` and no
re-extraction. That is the whole reason the filter is a table applied at load
time rather than a filter applied to the arrays: re-tuning ``--u`` re-runs
this script and ``standardize_feats``, not the ~75 GB extraction.

The cut table ``--emit_cuts`` writes belongs **outside** ``extracted/``.
``extract_data.py`` clears that directory on every run, which is why the
lookup tables do not live there either.

Usage:
    python scripts/calibrate_outlier_filter.py <data_dir>/extracted \\
        --report outlier_calibration.txt --emit_cuts <data_dir>/numeric_cuts.csv
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys

import numpy as np

import _path  # noqa: F401  (repository root on sys.path)

from TransEHR2.data.preprocessing import feature_scale


# Values kept per feature. The reservoir carries the body for quantiles; the
# extremes carry both tails in full, which is what the gap search walks.
#
# Bounded because the alternative is not affordable: one feature's observed
# values run to n * T entries, so holding all 94 features' outright is tens of
# gigabytes for statistics that need a sample of the body and an exact tail.
RESERVOIR = 200_000
EXTREMES = 20_000

# Episodes per read. The indicator array is (n, T, n_numeric): one feature's
# column is strided across the whole of it, so reading a column at a time
# would touch every page 94 times over. A row block is read once, sequentially,
# and yields every feature's mask at once.
BLOCK = 512


class Accumulator:
    """Bounded sample of one feature: a reservoir, plus both tails in full.

    Backed by numpy rather than Python lists -- a float in a list costs 32
    bytes against 8 -- and the tails are trimmed lazily. Trimming on every
    call would re-scan the retained tail once per block per feature, which is
    quadratic in the wrong quantity.
    """

    def __init__(self, reservoir: int = RESERVOIR, extremes: int = EXTREMES):
        self.count = 0
        # Neither log rule can see these: log10 is undefined at and below
        # zero, so both drop them before searching and neither can report on
        # them. Counted here so the third rule can be charged for them.
        self.zeros = 0
        self.negatives = 0
        self.capacity = reservoir
        self.extremes = extremes
        self._reservoir = np.empty(reservoir, dtype=np.float64)
        self._filled = 0
        self._high = np.empty(0, dtype=np.float64)
        self._low = np.empty(0, dtype=np.float64)
        self._pending: list = []

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return

        # Reservoir sampling: every value seen keeps an equal chance of being
        # retained, so the body sample does not favour the first episodes read.
        take = min(self.capacity - self._filled, values.size)
        if take:
            self._reservoir[self._filled:self._filled + take] = values[:take]
            self._filled += take
        for offset in range(take, values.size):
            j = random.randrange(self.count + offset + 1)
            if j < self.capacity:
                self._reservoir[j] = values[offset]
        self.count += values.size
        self.zeros += int((values == 0).sum())
        self.negatives += int((values < 0).sum())

        self._pending.append(values)
        if sum(p.size for p in self._pending) >= self.extremes:
            self._trim()

    def _trim(self) -> None:
        """Fold pending values into the tails. O(n) by partition, not a sort."""
        if not self._pending:
            return
        combined = np.concatenate([self._high, self._low] + self._pending)
        self._pending = []
        k = min(self.extremes, combined.size)
        self._high = np.partition(combined, -k)[-k:]
        self._low = np.partition(combined, k - 1)[:k]

    @property
    def high(self) -> np.ndarray:
        self._trim()
        return self._high

    @property
    def low(self) -> np.ndarray:
        self._trim()
        return self._low

    @property
    def reservoir(self) -> np.ndarray:
        return self._reservoir[:self._filled]

    @property
    def positive_count(self) -> int:
        """Observations both log rules could actually see."""
        return self.count - self.zeros - self.negatives

    def tail_sample(self) -> np.ndarray:
        """Reservoir plus both tails, for the gap search.

        The two overlap while the reservoir is not yet full, so this is not a
        sample of the distribution and must not be counted against. Finding a
        gap does not care.
        """
        return np.concatenate([self.reservoir, self.high, self.low])

    def body_sample(self) -> np.ndarray:
        """The reservoir alone: a uniform sample of every value seen."""
        return self.reservoir

    def surviving(self, low: float, high: float):
        """The smallest and largest values the cuts would keep.

        The cuts alone do not say how much room is left between them and the
        data: a high cut of 4500 on a feature whose largest real value is 22 is
        a different proposition from one whose largest is 4000. NaN on the side
        where a saturated tail put the answer outside the retained values.
        """
        kept_low = self.low[self.low >= low]
        kept_high = self.high[self.high <= high]
        return (float(kept_low.min()) if kept_low.size else np.nan,
                float(kept_high.max()) if kept_high.size else np.nan)

    def beyond(self, low: float, high: float, positive_only: bool = True):
        """How many observations fall outside the cuts, and whether saturated.

        Counted from the retained extremes rather than the reservoir, so exact
        rather than sampled -- unless more than ``extremes`` values lie beyond
        a cut, which would mean the rule removes far too much to adopt anyway.

        With ``positive_only``, zeros and negatives are excluded: the log rules
        derive their cuts from positive values alone, so counting values they
        never saw against them reads a feature's sentinel share as if the rule
        had chosen to remove it. A rule working in original units sees
        everything and passes False.
        """
        tail_high, tail_low = self.high, self.low
        if positive_only:
            tail_high = tail_high[tail_high > 0]
            tail_low = tail_low[tail_low > 0]
        above = int((tail_high > high).sum())
        below = int((tail_low < low).sum())
        saturated = (above >= tail_high.size == self.extremes
                     or below >= tail_low.size == self.extremes)
        return above + below, saturated


def tail_gap_cut(values: np.ndarray, gap: float, quantile: float,
                 min_fold: float):
    """Where the tails detach from the body, in original units.

    Args:
        values: Observed values for one feature.
        gap: Width in decades of the jump that counts as a detachment.
        quantile: Only search beyond this quantile, so the body is never cut.
        min_fold: Never cut closer to the median than this multiple, whatever
            the search finds. A guard against slicing into a feature whose
            tail is genuinely sparse.

    Returns:
        ``(low_cut, high_cut)``, either bound infinite if no detachment was
        found. Each cut sits in the middle of the gap rather than on the first
        detached value: a value round-tripped through log10 and back does not
        always compare equal to itself, so a cut set on the datum can fail to
        exclude it.
    """
    positive = values[values > 0]
    if positive.size < MIN_VALUES:
        return -np.inf, np.inf

    y = np.sort(np.log10(positive))
    median = float(np.median(y))
    guard = np.log10(min_fold)

    high_cut = np.inf
    start = np.searchsorted(
        y, max(float(np.quantile(y, quantile)), median + guard)
    )
    for i in range(max(start, 1), y.size):
        if y[i] - y[i - 1] > gap:
            high_cut = 10.0 ** ((y[i] + y[i - 1]) / 2.0)
            break

    low_cut = -np.inf
    stop = np.searchsorted(
        y, min(float(np.quantile(y, 1 - quantile)), median - guard)
    )
    for i in range(min(stop, y.size - 1), 0, -1):
        if y[i] - y[i - 1] > gap:
            low_cut = 10.0 ** ((y[i] + y[i - 1]) / 2.0)
            break

    return low_cut, high_cut


def robust_log_cut(values: np.ndarray, k: float):
    """Cut at ``k`` robust spreads from the median, in log space.

    The gap rule fails on two counts no threshold of its own can fix: one
    erroneous value between the body and a far one bridges the gap meant to
    separate them, and its guard is a fixed multiple of the median, which is
    many spreads out for a tight feature and inside the normal range for a
    wide one.

    Log space makes the rule scale-free, and taking the spread from the
    feature itself makes the threshold mean the same thing everywhere: a
    temperature 1.2 decades out is absurd, an ALT 2.7 decades out is a real
    acute liver injury, and the two are told apart by how far their own bodies
    spread rather than by a decade count.

    Returns:
        ``(low_cut, high_cut, scale)``, scale being one robust spread in
        decades. Infinite cuts and a NaN scale when there is too little to
        measure; a zero scale when the feature is so discrete that half its
        values sit on the median.
    """
    positive = values[values > 0]
    if positive.size < MIN_VALUES:
        return -np.inf, np.inf, np.nan

    y = np.log10(positive)
    centre = float(np.median(y))
    # 1.4826 puts the MAD on the same footing as a standard deviation for
    # normal data, so k reads as a number of sigmas rather than an arbitrary
    # unit.
    scale = 1.4826 * float(np.median(np.abs(y - centre)))
    if scale <= 0:
        return -np.inf, np.inf, 0.0
    return 10.0 ** (centre - k * scale), 10.0 ** (centre + k * scale), scale


def standardized_scale(body: np.ndarray):
    """The centre and scale the model standardizes with, measured robustly.

    ``standardize_feats`` takes its percentiles of the **L2 norm** of a
    feature's value vector, which for the width-1 numeric features of section
    4.4 is the absolute value -- so the scale is measured over ``|v|``, not over
    the signed values. The two coincide for a strictly positive test and diverge
    for one that goes negative.

    The scale is ``feature_scale``'s, not a bare 5th-to-95th range: that range
    is degenerate for a test reported at a detection limit, and widening it is
    what keeps such a feature from being reduced to its occurrence indicator.
    Measuring against anything else would report a magnitude the loss does not
    see.

    Unlike a MAD, that span covers the clinically observed range rather than
    the regulated core, which is what a rule needs if real pathology is not to
    read as extreme.

    The centre here is the **median**, where the model subtracts the mean. A
    mean is dragged by the very values being looked for, so a diagnostic using
    one would understate exactly the outliers it exists to find. The
    consequence is that a cut from this rule is ``u`` spans from the median
    rather than from the model's centre; the two differ by however far the
    outliers have pulled the mean, which is itself worth reading off the
    report.
    """
    if body.size < MIN_VALUES:
        return np.nan, np.nan
    # Imported rather than reimplemented: this has to be the divisor the loader
    # actually uses, and a second copy of the widening rule is a second thing to
    # keep in step with it.
    scale = feature_scale(np.abs(body))
    return float(np.median(body)), (scale if scale > 0 else np.nan)


def standardized_extent(value: float, centre: float, scale: float) -> float:
    """How large a value is once standardized: what the loss squares."""
    if not np.isfinite(value) or not np.isfinite(scale):
        return np.nan
    return (value - centre) / scale


def standardized_cut(body: np.ndarray, u: float):
    """Cut beyond ``u`` standardized units of the median.

    Returns:
        ``(low_cut, high_cut, centre, scale)`` in original units, the cuts
        infinite when there is too little to measure.
    """
    centre, scale = standardized_scale(body)
    if not np.isfinite(scale):
        return -np.inf, np.inf, centre, scale
    return centre - u * scale, centre + u * scale, centre, scale


# Below this many positive observations a feature is not measured at all:
# every rule here reads a quantile or a median off the tail, and on a handful
# of values that is noise wearing a threshold's clothing.
MIN_VALUES = 100


def collect(base_path: str, block: int, seed: int):
    """One pass over the extracted arrays, accumulating per numeric feature.

    Read memory-mapped and never written. Blocked by episode rather than by
    feature because the indicator array is ``(n, T, n_numeric)``: one
    feature's column is strided across the whole of it, so a column-at-a-time
    read would touch every page once per feature.

    Returns:
        ``(names, accumulators)``, aligned with ``metadata['numeric_feats']``.
    """
    with open(os.path.join(base_path, 'metadata.pkl'), 'rb') as f_in:
        metadata = pickle.load(f_in)

    names = list(metadata['numeric_feats'])
    dims = list(metadata['numeric_feat_dims'])
    wide = [n for n, d in zip(names, dims) if d != 1]
    if wide:
        raise ValueError(
            "Only width-1 numeric features can be calibrated; "
            f"{', '.join(wide)} are wider. A cut on a multi-column feature "
            "would have to say which column it applies to, and section 4.4 "
            "gives every numeric feature width 1."
        )

    indicators = np.load(
        os.path.join(base_path, 'val_numeric_indicators.npy'), mmap_mode='r'
    )
    values = [
        np.load(os.path.join(base_path, f'val_numeric_values_{f}.npy'),
                mmap_mode='r')
        for f in range(len(names))
    ]

    random.seed(seed)
    accumulators = [Accumulator() for _ in names]
    n_episodes = indicators.shape[0]

    for start in range(0, n_episodes, block):
        stop = min(start + block, n_episodes)
        indicator_block = np.asarray(indicators[start:stop])
        for f in range(len(names)):
            mask = indicator_block[:, :, f] == 1.0
            if not mask.any():
                continue
            observed = np.asarray(values[f][start:stop])[..., 0][mask]
            accumulators[f].add(observed)
        print(f"  {stop}/{n_episodes} episodes", end='\r', file=sys.stderr)

    print(file=sys.stderr)
    return names, accumulators


def _number(value: float) -> str:
    """A table cell for a value that may not exist."""
    return 'n/a' if not np.isfinite(value) else f'{value:.4g}'


def _fraction(removed: int, total: int) -> str:
    return '0' if not total else f'{removed / total:.2%}'


HEADER = (f"{'feature':<12}{'low':>12}{'high':>12}{'removed':>10}"
          f"{'share':>9}{'kept low':>12}{'kept high':>12}")


def acting_cuts(names, accumulators, cuts, positive_only):
    """The features a rule actually removes something from.

    A rule that finds nothing signals it with infinite bounds, but the
    standardized-magnitude rule has no such state: its cut is a band at
    ``median +- u * span``, which is finite for every feature whose scale is
    measurable. Reporting or tabulating those would list all 94 features
    whether or not the cut does anything.

    Calibration and application read the same arrays, so a cut that removes
    nothing here provably removes nothing later, and leaving it out of the
    table costs no coverage. That makes "in the table" mean "acts on this
    feature" for all three rules alike.

    Yields:
        ``(name, accumulator, low, high, removed, saturated)`` per feature
        the rule acts on.
    """
    for name, acc, (low, high) in zip(names, accumulators, cuts):
        if not np.isfinite(low) and not np.isfinite(high):
            continue
        removed, saturated = acc.beyond(low, high, positive_only)
        if not removed:
            continue
        yield name, acc, low, high, removed, saturated


def _rows(names, accumulators, cuts, positive_only):
    """One report line per feature the rule acts on, and the totals."""
    lines = [HEADER, '-' * len(HEADER)]
    touched = 0
    total_removed = 0
    for name, acc, low, high, removed, saturated in acting_cuts(
            names, accumulators, cuts, positive_only):
        kept_low, kept_high = acc.surviving(low, high)
        denominator = acc.positive_count if positive_only else acc.count
        touched += 1
        total_removed += removed
        flag = ' (saturated)' if saturated else ''
        lines.append(
            f"{name:<12}{_number(low):>12}{_number(high):>12}"
            f"{removed:>10}{_fraction(removed, denominator):>9}"
            f"{_number(kept_low):>12}{_number(kept_high):>12}{flag}"
        )
    lines.append('-' * len(HEADER))
    lines.append(f"{touched} of {len(names)} feature(s) cut, "
                 f"{total_removed} observation(s) removed in total")
    return lines


def report(names, accumulators, args) -> list:
    """The whole report: a guide, then one section per rule."""
    lines = [
        "Outlier filter calibration",
        "=" * 78,
        "",
        "Nothing here has been applied. Each section says what one rule would",
        "remove at the thresholds this run used. Read them against each other:",
        "",
        "  - The two log rules ask whether a value is clinically possible.",
        "  - The standardized-magnitude rule asks whether it would dominate",
        "    the loss, which is the question the model needs answered. A",
        "    height of 419 cm is impossible and standardizes to single digits;",
        "    a sentinel near 1e6 standardizes to thousands.",
        "",
        "'kept low' and 'kept high' are the extreme values a cut would leave",
        "in place. A high cut far above the largest surviving value is doing",
        "nothing; one just above it is about to cut into real data.",
        "",
        "'share' is of the observations the rule could see -- for the log",
        "rules, the positive ones, since neither can act on a zero or a",
        "negative. The zeros and negatives have their own section.",
        "",
        f"Thresholds: --gap {args.gap} --quantile {args.quantile} "
        f"--min_fold {args.min_fold} --z {args.z} --u {args.u}",
        "",
    ]

    bodies = [acc.body_sample() for acc in accumulators]
    tails = [acc.tail_sample() for acc in accumulators]

    lines += ["", f"Tail gap (--gap {args.gap} decades)", "=" * 78]
    gap_cuts = [
        tail_gap_cut(t, args.gap, args.quantile, args.min_fold) for t in tails
    ]
    lines += _rows(names, accumulators, gap_cuts, True)

    lines += ["", "", f"Robust log spread (--z {args.z})", "=" * 78]
    robust = [robust_log_cut(t, args.z) for t in tails]
    lines += _rows(names, accumulators,
                   [(low, high) for low, high, _ in robust], True)

    lines += ["", "", f"Standardized magnitude (--u {args.u} spans)",
              "=" * 78]
    standardized = [standardized_cut(b, args.u) for b in bodies]
    lines += _rows(names, accumulators,
                   [(low, high) for low, high, _, _ in standardized], False)

    lines += ["", "", "Zeros and negatives", "=" * 78,
              "Invisible to both log rules, and a negative reading is itself a",
              "finding: no unit conversion in prepare_RMT23345_LAB.R produces",
              "one, so a negative is a data defect rather than a scale.", "",
              f"{'feature':<12}{'observed':>12}{'zeros':>10}"
              f"{'negatives':>12}"]
    lines.append('-' * 46)
    for name, acc in zip(names, accumulators):
        if not acc.zeros and not acc.negatives:
            continue
        lines.append(f"{name:<12}{acc.count:>12}{acc.zeros:>10}"
                     f"{acc.negatives:>12}")
    lines.append('-' * 46)

    return lines, standardized, gap_cuts, robust


def emit_cuts(path: str, names, accumulators, cuts, rule: str,
              positive_only: bool) -> int:
    """Write the cut table the loader applies: FEATURE, LOW, HIGH.

    Only features the rule removes something from are written -- see
    ``acting_cuts``. A feature absent from the table is left alone, so an
    empty table and no table mean the same thing.
    """
    written = 0
    with open(path, 'w') as f_out:
        f_out.write(f"# {rule} cuts from calibrate_outlier_filter.py. "
                    f"Regenerate rather than editing by hand.\n")
        f_out.write("FEATURE,LOW,HIGH\n")
        for name, _, low, high, _, _ in acting_cuts(
                names, accumulators, cuts, positive_only):
            f_out.write(f"{name},{low},{high}\n")
            written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('extracted_dir',
                        help="{DATA_DIR}/extracted, the extractor's output.")
    parser.add_argument('--report', default=None,
                        help='Write the report here. Detail goes to a file '
                             'rather than stdout, which is captured by '
                             'whatever ran this. Omit to print it.')
    parser.add_argument('--emit_cuts', default=None,
                        help='Write the cut table here. Put it OUTSIDE '
                             'extracted/, which extract_data.py clears on '
                             'every run.')
    parser.add_argument('--rule', default='standardized',
                        choices=('standardized', 'tail_gap', 'robust'),
                        help='Which rule --emit_cuts writes. Default is the '
                             'one that answers the training-stability '
                             'question.')
    parser.add_argument('--block', type=int, default=BLOCK,
                        help='Episodes per read.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gap', type=float, default=0.5,
                        help='Tail-gap width, in decades.')
    parser.add_argument('--quantile', type=float, default=0.999,
                        help='Tail-gap search starts beyond this quantile.')
    parser.add_argument('--min_fold', type=float, default=3.0,
                        help='Tail-gap guard, as a multiple of the median.')
    parser.add_argument('--z', type=float, default=8.0,
                        help='Robust spreads from the median, in log space.')
    parser.add_argument('--u', type=float, default=20.0,
                        help='Standardized spans from the median.')
    args = parser.parse_args()

    names, accumulators = collect(args.extracted_dir, args.block, args.seed)
    lines, standardized, gap_cuts, robust = report(names, accumulators, args)

    if args.report:
        with open(args.report, 'w') as f_out:
            f_out.write('\n'.join(lines) + '\n')
        print(f"Wrote the report for {len(names)} numeric feature(s) to "
              f"{args.report}")
    else:
        print('\n'.join(lines))

    if args.emit_cuts:
        chosen, positive_only = {
            'standardized': (
                [(low, high) for low, high, _, _ in standardized], False),
            'tail_gap': (gap_cuts, True),
            'robust': ([(low, high) for low, high, _ in robust], True),
        }[args.rule]
        written = emit_cuts(args.emit_cuts, names, accumulators, chosen,
                            args.rule, positive_only)
        print(f"Wrote {args.rule} cuts for {written} feature(s) to "
              f"{args.emit_cuts}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
