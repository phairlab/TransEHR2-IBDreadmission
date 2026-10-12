"""The outlier filter, from the emitted table to the value the loss squares.

``calibrate_outlier_filter.py`` measures what a rule would remove and writes a
``FEATURE,LOW,HIGH`` table in original units. Nothing used to read it. These
probe the two places that now do: ``standardize_feats``, which must measure the
cut values rather than the stored ones, and ``MixedDataset``, which clips at
load time.

The defect that motivates it: a laboratory sentinel standardizes into the
thousands against a robust scale that does not clip, the generator squares it,
and one value sets an epoch's mean. A cut bounds that at ``u`` spans by
construction.
"""

import pickle
import shutil
import subprocess
import sys

import numpy as np
import pytest

from pathlib import Path

from TransEHR2.data.datasets import MixedDataset
from TransEHR2.data.preprocessing import read_numeric_cuts, standardize_feats

SCRIPT = (Path(__file__).resolve().parents[2]
          / 'scripts' / 'apply_numeric_cuts.py')


# --- the table ------------------------------------------------

def _table(tmp_path, body: str):
    path = tmp_path / 'numeric_cuts.csv'
    path.write_text("# standardized cuts from calibrate_outlier_filter.py\n"
                    "FEATURE,LOW,HIGH\n" + body)
    return str(path)


def test_an_empty_table_cuts_nothing(tmp_path):
    """An empty table and no table have to mean the same thing."""
    low, high = read_numeric_cuts(_table(tmp_path, ''), ['HGB', 'CRP'])
    assert np.all(np.isneginf(low)) and np.all(np.isposinf(high))


def test_cuts_land_on_the_named_feature(tmp_path):
    """Alignment is by name, not by row order: the table holds only the
    features the rule acts on, so its order is not the extraction's."""
    path = _table(tmp_path, 'CRP,0.0,80.0\n')
    low, high = read_numeric_cuts(path, ['HGB', 'CRP', 'ALB'])
    assert (low[1], high[1]) == (0.0, 80.0)
    assert np.isneginf(low[0]) and np.isposinf(high[2])


def test_a_table_from_a_different_extraction_is_refused(tmp_path):
    """Applying the rest of it would cut the wrong features silently."""
    path = _table(tmp_path, 'CRP,0.0,80.0\nNOTAFEATURE,1.0,2.0\n')
    with pytest.raises(ValueError, match='does not have'):
        read_numeric_cuts(path, ['CRP'])


# --- the statistics -------------------------------------------

def _arrays(values):
    """One feature, one timestep per episode, every position observed."""
    values = np.asarray(values, dtype=np.float32).reshape(-1, 1, 1)
    return {
        'val_numeric_indicators': np.ones(
            (values.shape[0], 1, 1), dtype=np.float32),
        'val_numeric_values': [values],
    }


def test_a_sentinel_drags_the_mean_when_it_is_not_cut(tmp_path):
    """The premise. The mean is the one statistic here that is not robust."""
    arrays = _arrays([1.0] * 99 + [1e6])
    path = str(tmp_path / 'summary_statistics_fold0.npz')
    standardize_feats(arrays, 1, np.arange(100), path)
    assert np.load(path)['means'][0] > 100.0


def test_cutting_restores_the_centre(tmp_path):
    """So statistics and cuts have to be measured together."""
    arrays = _arrays([1.0] * 99 + [1e6])
    path = str(tmp_path / 'summary_statistics_fold0.npz')
    cuts = (np.array([0.0], np.float32), np.array([2.0], np.float32))
    standardize_feats(arrays, 1, np.arange(100), path, cuts=cuts)

    stats = np.load(path)
    assert stats['means'][0] == pytest.approx(1.01, abs=1e-5)
    assert stats['cut_low'][0] == 0.0 and stats['cut_high'][0] == 2.0


def test_statistics_measured_without_cuts_carry_none(tmp_path):
    """Which is what makes an older statistics file mean 'cut nothing'."""
    path = str(tmp_path / 'summary_statistics_fold0.npz')
    standardize_feats(_arrays([1.0, 2.0]), 1, np.arange(2), path)
    assert 'cut_low' not in np.load(path)


def test_the_arrays_are_not_modified(tmp_path):
    """The cut is one fold's; the values are the cohort's."""
    arrays = _arrays([1.0] * 99 + [1e6])
    before = arrays['val_numeric_values'][0].copy()
    cuts = (np.array([0.0], np.float32), np.array([2.0], np.float32))
    standardize_feats(arrays, 1, np.arange(100),
                      str(tmp_path / 's.npz'), cuts=cuts)
    assert (arrays['val_numeric_values'][0] == before).all()


# --- the loader -----------------------------------------------

def _dataset(values, indicators, stats):
    """A one-feature, one-episode dataset; every other family empty."""
    n, t = indicators.shape[0], indicators.shape[1]
    empty = np.zeros((n, t, 0), dtype=np.float32)
    return MixedDataset(
        val_times=np.zeros((n, t), dtype=np.int32),
        val_masks=np.ones((n, t), dtype=np.float32),
        val_numeric_indicators=indicators,
        val_numeric_values=[values],
        val_categorical_indicators=empty, val_categorical_values=[],
        val_ordinal_indicators=empty, val_ordinal_values=[],
        val_multilabel_indicators=empty, val_multilabel_values=[],
        categorical_feat_dims=[], ordinal_feat_dims=[],
        lookup_indicators={}, lookup_csr=[], lookup_tables=[],
        lookup_slot_dims=[], lookup_feat_types=[],
        event_indicators=empty, event_times=np.zeros((n, t), dtype=np.int32),
        event_masks=np.ones((n, t), dtype=np.float32),
        time_to_event=np.zeros(n, dtype=np.float32),
        event_type=np.zeros(n, dtype=np.int8),
        max_ts_len=t, numeric_stats=stats,
    )


STATS = {'means': np.array([1.0], np.float32),
         'p5': np.array([0.0], np.float32),
         'p95': np.array([2.0], np.float32),
         'scale': np.array([2.0], np.float32)}


def test_a_sentinel_is_bounded_by_its_cut():
    """1e6 standardizes to 5e5 uncut and to 4.5 clipped at 10."""
    values = np.array([[[1e6]]], dtype=np.float32)
    indicators = np.ones((1, 1, 1), dtype=np.float32)

    uncut = _dataset(values, indicators, dict(STATS))
    assert uncut._numeric_values(0, 0, indicators[0, :, 0]).item() > 1e5

    cut = _dataset(values, indicators, dict(
        STATS, cut_low=np.array([0.0], np.float32),
        cut_high=np.array([10.0], np.float32)))
    assert cut._numeric_values(0, 0, indicators[0, :, 0]).item() == \
        pytest.approx(4.5)


def test_an_unobserved_zero_is_not_raised_to_the_low_cut():
    """The hazard a blind clip would introduce. An unobserved position is
    stored as 0, and any strictly positive assay has a low cut above it, so
    clipping the whole row would invent a reading at every empty timestep."""
    values = np.array([[[0.0], [5.0]]], dtype=np.float32)
    indicators = np.array([[[0.0], [1.0]]], dtype=np.float32)

    dataset = _dataset(values, indicators, dict(
        STATS, cut_low=np.array([2.0], np.float32),
        cut_high=np.array([4.0], np.float32)))
    out = dataset._numeric_values(0, 0, indicators[0, :, 0])

    # Unobserved: 0 standardized, i.e. (0 - 1) / 2. Observed: clipped to 4.
    assert out[0].item() == pytest.approx(-0.5)
    assert out[1].item() == pytest.approx(1.5)


def test_statistics_without_cuts_leave_the_values_as_they_were():
    """The back-compat path, which every fold is on until the table lands."""
    values = np.array([[[9.0]]], dtype=np.float32)
    indicators = np.ones((1, 1, 1), dtype=np.float32)
    out = _dataset(values, indicators, dict(STATS))._numeric_values(
        0, 0, indicators[0, :, 0])
    assert out.item() == pytest.approx(4.0)


# --- the script -----------------------------------------------

def _extraction(tmp_path, values):
    """The smallest data_dir apply_numeric_cuts.py will work over."""
    extracted = tmp_path / 'extracted'
    extracted.mkdir()
    values = np.asarray(values, dtype=np.float32).reshape(-1, 1, 1)
    np.save(extracted / 'val_numeric_values_0.npy', values)
    np.save(extracted / 'val_numeric_indicators.npy',
            np.ones(values.shape, dtype=np.float32))
    with open(extracted / 'metadata.pkl', 'wb') as f_out:
        pickle.dump({'numeric_feats': ['CRP'], 'numeric_feat_dims': [1]},
                    f_out)

    fold = tmp_path / 'fold0'
    fold.mkdir()
    np.save(fold / 'fold0_train_rows.npy', np.arange(values.shape[0]))
    return extracted


def _run(tmp_path, *args):
    """Through the CLI, which is where the import chain is actually tested."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path), *args],
        capture_output=True, text=True
    )


def test_the_script_rewrites_the_fold_statistics_under_the_cuts(tmp_path):
    extracted = _extraction(tmp_path, [1.0] * 99 + [1e6])
    cuts = _table(tmp_path, 'CRP,0.0,2.0\n')

    result = _run(tmp_path, '--cuts', cuts)
    assert result.returncode == 0, result.stderr

    stats = np.load(extracted / 'summary_statistics_fold0.npz')
    assert stats['means'][0] == pytest.approx(1.01, abs=1e-5)
    assert stats['cut_high'][0] == 2.0


def test_running_it_twice_gives_the_same_statistics(tmp_path):
    """Re-tuning a threshold re-runs this, so it has to be idempotent."""
    extracted = _extraction(tmp_path, [1.0] * 99 + [1e6])
    cuts = _table(tmp_path, 'CRP,0.0,2.0\n')

    _run(tmp_path, '--cuts', cuts)
    first = dict(np.load(extracted / 'summary_statistics_fold0.npz'))
    _run(tmp_path, '--cuts', cuts)
    second = dict(np.load(extracted / 'summary_statistics_fold0.npz'))

    assert first.keys() == second.keys()
    assert all((first[k] == second[k]).all() for k in first)


def test_no_table_restores_what_extraction_measured(tmp_path):
    """How a cut table is withdrawn without re-extracting."""
    extracted = _extraction(tmp_path, [1.0] * 99 + [1e6])
    _run(tmp_path, '--cuts', _table(tmp_path, 'CRP,0.0,2.0\n'))
    _run(tmp_path)

    stats = np.load(extracted / 'summary_statistics_fold0.npz')
    assert 'cut_low' not in stats
    assert stats['means'][0] > 100.0


def test_a_cohort_with_no_folds_is_an_error(tmp_path):
    """Statistics are per fold; there is nothing to write without one."""
    _extraction(tmp_path, [1.0, 2.0])
    shutil.rmtree(tmp_path / 'fold0')

    result = _run(tmp_path)
    assert result.returncode == 1
    assert 'no fold' in result.stderr
