"""``*_LADDER_P_MIN`` and ``*_LADDER_P_MAX`` from the extracted timestamps.

Reads ``extracted/{val,event}_{times,masks}.npy`` and prints the frequency
ladder's bounds for each stream, in days. The numbers go into an experiment
config by hand; nothing here writes one.

What sets each bound
--------------------

``build_frequency_ladder`` spaces bands log-uniformly in period. A band
carries usable information about a gap only while its phase change is
neither frozen nor wrapped, ``0.1 <= gap / lambda <= pi``, which in periods
is ``2 * gap <= P <= 20 * pi * gap``. So the shortest gap the model must
resolve sets ``P_MIN`` and the longest span it must represent sets
``P_MAX``.

The span is ``max - min`` over an episode's *observed* timesteps, not
``max``. Attention is all-pairs, so what the encoding has to represent is
the distance between the earliest and latest record, and these timestamps
are non-positive and right-aligned -- the index time is ``0`` in the final
column.

That last point is also the hazard here. Padding is stored as ``0``, which
is a perfectly legitimate timestamp in this layout, so padding can only be
excluded by the mask. Reading the arrays without it would report every
episode as spanning from its earliest record to the index time, which is
wrong in the safe direction for ``P_MAX`` and silently wrong for the step
gaps.

This reads cohort data. It prints aggregates only -- never a record, a
patient or a row.

Usage:
    python scripts/measure_ladder_bounds.py /path/to/data
    python scripts/measure_ladder_bounds.py /path/to/data --chunk 8192
"""

import argparse
import os

import numpy as np

import _path  # noqa: F401  (repository root on sys.path)

from TransEHR2.data.datasets import MINUTES_PER_DAY
from TransEHR2.survival import format_days


# `0.1 <= gap / lambda <= pi` in periods: P_MIN = 2 * gap_min and
# P_MAX = 20 * pi * gap_max. The upper factor is ~62.83; the MIMIC configs
# round it to 63, and the drift is well inside a log-uniform band.
P_MIN_FACTOR = 2.0
P_MAX_FACTOR = 20.0 * np.pi

STREAMS = (('VALUE', 'val'), ('EVENT', 'event'))


def measure_spans(times, masks, chunk: int = 4096) -> dict:
    """Spans and step gaps over observed timesteps, in days.

    Args:
        times: (n_episodes, n_timesteps) int32 minutes <= 0, memory-mapped.
        masks: (n_episodes, n_timesteps), non-zero at a real record.
        chunk: Episodes to bring into memory at a time.

    Returns:
        A dict of ``max_span``, ``median_span``, ``min_step_gap``,
        ``p01_episode_min_gap``, ``max_step_gap`` and ``n_episodes``, all in
        days, plus the episode count the spans were taken over.
    """
    max_span = 0.0
    max_step = 0.0
    min_step = np.inf
    spans = []
    episode_min_gaps = []

    for start in range(0, times.shape[0], chunk):
        block = np.asarray(times[start:start + chunk],
                           dtype=np.float64) / MINUTES_PER_DAY
        valid = np.asarray(masks[start:start + chunk]) > 0.5
        if not valid.any():
            continue

        # Rows with no record at all are dropped before the reduction, not
        # filtered after it. An all-NaN row is what an episode carrying no
        # record in this stream looks like, and nanmin warns on one -- the
        # answer would still be right, but the run would print a page of
        # RuntimeWarnings over an expected case.
        has_record = valid.any(axis=1)
        if not has_record.any():
            continue

        # By the mask, never by value: 0 is the index time, not padding.
        masked = np.where(valid[has_record], block[has_record], np.nan)
        lo = np.nanmin(masked, axis=1)
        hi = np.nanmax(masked, axis=1)
        episode_spans = hi - lo
        if episode_spans.size:
            spans.append(episode_spans)
            max_span = max(max_span, float(episode_spans.max()))

        for row, row_valid in zip(block, valid):
            observed = np.sort(row[row_valid])
            if observed.size < 2:
                continue
            steps = np.diff(observed)
            steps = steps[steps > 0]
            if steps.size:
                max_step = max(max_step, float(steps.max()))
                min_step = min(min_step, float(steps.min()))
                episode_min_gaps.append(float(steps.min()))

    all_spans = np.concatenate(spans) if spans else np.zeros(0)
    gaps = np.asarray(episode_min_gaps)
    return {
        'max_span': max_span,
        'median_span': float(np.median(all_spans)) if all_spans.size else 0.0,
        'min_step_gap': None if not np.isfinite(min_step) else min_step,
        'p01_episode_min_gap': (float(np.quantile(gaps, 0.01))
                                if gaps.size else None),
        'max_step_gap': max_step,
        'n_episodes': int(all_spans.size),
    }


def load_stream(extracted: str, prefix: str):
    """``(times, masks)`` memory-mapped, or ``None`` if absent."""
    paths = [os.path.join(extracted, f'{prefix}_{kind}.npy')
             for kind in ('times', 'masks')]
    if not all(os.path.exists(p) for p in paths):
        return None
    return tuple(np.load(p, mmap_mode='r') for p in paths)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Frequency ladder bounds from the extracted timestamps.'
    )
    parser.add_argument('data_dir',
                        help='Directory holding extracted/')
    parser.add_argument('--chunk', type=int, default=4096,
                        help='Episodes held in memory at a time '
                             '[default: %(default)s]')
    args = parser.parse_args(argv)

    extracted = os.path.join(args.data_dir, 'extracted')
    if not os.path.isdir(extracted):
        print(f'No extracted/ under {args.data_dir}')
        return 1

    print(f'data:   {extracted}')
    print('units:  days (int32 minutes on disk, / 1440 at __getitem__)\n')

    config_lines = []
    for label, prefix in STREAMS:
        stream = load_stream(extracted, prefix)
        if stream is None:
            print(f'{label}: no {prefix}_times.npy / {prefix}_masks.npy; '
                  f'skipped\n')
            continue

        measured = measure_spans(*stream, chunk=args.chunk)
        if not measured['n_episodes'] or measured['min_step_gap'] is None:
            print(f'{label}: no episode carries two observed records, so '
                  f'there is no gap to measure\n')
            continue

        p_min = P_MIN_FACTOR * measured['min_step_gap']
        p_max = P_MAX_FACTOR * measured['max_span']

        print(f'{label} stream  ({measured["n_episodes"]} episodes with '
              f'records)')
        print(f'  span      max {format_days(measured["max_span"]):>10}   '
              f'median {format_days(measured["median_span"]):>10}')
        print(f'  step gap  min {measured["min_step_gap"]:.6f} d   '
              f'max {format_days(measured["max_step_gap"]):>10}')
        print(f'            1st pct of per-episode minima '
              f'{measured["p01_episode_min_gap"]:.6f} d')
        decades = np.log10(p_max / p_min) if p_min > 0 else float('nan')
        print(f'  ladder    P_MIN {p_min:.6f} d    P_MAX {p_max:.2f} d '
              f'({format_days(p_max)})')
        print(f'            spans {decades:.1f} decades\n')

        config_lines += [f'{label}_LADDER_P_MIN: {p_min:.6g}',
                         f'{label}_LADDER_P_MAX: {p_max:.6g}']

    if not config_lines:
        print('Nothing measured.')
        return 1

    print('Paste into the experiment config:\n')
    print('\n'.join(config_lines))
    print('\n  P_MIN follows the shortest observed gap, so one episode with '
          'two\n  records a minute apart sets it for the whole study. Weigh '
          'it against\n  the 1st-percentile figure above before taking it '
          'literally: a ladder\n  stretched over needless decades spends '
          'bands on gaps nothing has.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
