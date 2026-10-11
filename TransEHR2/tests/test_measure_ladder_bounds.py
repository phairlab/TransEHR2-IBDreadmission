"""The ladder bounds, measured rather than carried over.

MIMIC's four constants are in hours and its event bound came from a
window this fork does not have, so both bounds are measured here. These
probes pin the two readings that are easy to get wrong: a span is
``max - min`` rather than ``max``, and padding stored as ``0`` is
indistinguishable from the index time without the mask.
"""

import numpy as np
import pytest

from measure_ladder_bounds import (P_MAX_FACTOR, P_MIN_FACTOR, measure_spans)


DAY = 1440.0  # minutes


def episode(*days_before):
    """One right-aligned row of minutes <= 0, and its mask.

    Padding is written as 0 deliberately: that is what the extractor
    leaves, and 0 is also a real timestamp -- the index time.
    """
    times = np.zeros(6, dtype=np.int32)
    mask = np.zeros(6, dtype=np.float32)
    for i, day in enumerate(days_before):
        times[len(times) - len(days_before) + i] = int(-day * DAY)
        mask[len(mask) - len(days_before) + i] = 1.0
    return times, mask


def spans_of(*episodes, chunk=4096):
    times = np.stack([e[0] for e in episodes])
    masks = np.stack([e[1] for e in episodes])
    return measure_spans(times, masks, chunk=chunk)


def test_the_span_is_earliest_to_latest_not_distance_to_the_index():
    """Attention is all-pairs, so the gap to represent is the full extent.

    An episode running from 30 to 10 days before the index spans 20 days,
    not the 30 the last timestamp would suggest.
    """
    assert spans_of(episode(30, 20, 10))['max_span'] == pytest.approx(20.0)


def test_padding_is_excluded_by_the_mask_not_by_value():
    """0 is the index time in this layout, not a missing value.

    Reading the arrays without the mask would take every padded cell for
    a record at the index and report a span running to 0.
    """
    # Two real records, 30 and 20 days back; four padded cells at 0.
    measured = spans_of(episode(30, 20))
    assert measured['max_span'] == pytest.approx(10.0), "not 30"
    assert measured['min_step_gap'] == pytest.approx(10.0)


def test_an_episode_at_the_index_time_still_counts():
    """A record *at* 0 is real, and the mask is what says so."""
    measured = spans_of(episode(5, 0))
    assert measured['max_span'] == pytest.approx(5.0)
    assert measured['n_episodes'] == 1


def test_the_shortest_gap_anywhere_sets_p_min():
    """One tight pair in one episode sets the bound for the whole study.

    Which is why the script reports the 1st percentile of per-episode
    minima beside it rather than printing the minimum alone.
    """
    measured = spans_of(episode(40, 20), episode(10, 10 - 1 / 1440))
    assert measured['min_step_gap'] == pytest.approx(1 / 1440, rel=1e-6)
    assert P_MIN_FACTOR * measured['min_step_gap'] == pytest.approx(
        2 / 1440, rel=1e-6)


def test_the_longest_span_anywhere_sets_p_max():
    measured = spans_of(episode(3, 1), episode(400, 2))
    assert measured['max_span'] == pytest.approx(398.0)
    assert P_MAX_FACTOR * measured['max_span'] == pytest.approx(
        20 * np.pi * 398.0)


def test_a_single_record_contributes_a_span_but_no_gap():
    """It has an extent of zero and no step to measure."""
    measured = spans_of(episode(7))
    assert measured['max_span'] == pytest.approx(0.0)
    assert measured['min_step_gap'] is None


def test_an_empty_episode_is_skipped_entirely():
    """All padding, so nothing to measure and nothing to crash on."""
    measured = spans_of(episode())
    assert measured['n_episodes'] == 0
    assert measured['min_step_gap'] is None


def test_chunking_does_not_change_the_answer():
    """The arrays are memory-mapped and read in blocks."""
    rows = [episode(40, 20, 5), episode(9, 2), episode(100, 99)]
    assert (spans_of(*rows, chunk=1)['max_span']
            == spans_of(*rows, chunk=64)['max_span'])
    assert (spans_of(*rows, chunk=1)['min_step_gap']
            == pytest.approx(spans_of(*rows, chunk=64)['min_step_gap']))


def test_duplicate_timestamps_do_not_become_a_zero_gap():
    """Two records at one timestamp are one instant, not an infinitely
    fast one -- a zero gap would drive P_MIN to zero and the ladder with
    it."""
    measured = spans_of(episode(12, 12, 4))
    assert measured['min_step_gap'] == pytest.approx(8.0)
