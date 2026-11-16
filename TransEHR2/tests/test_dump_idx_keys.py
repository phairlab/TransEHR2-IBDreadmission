"""Every model's dump keys an episode by its ``labels.csv`` row.

A TransEHR2 batch's ``idx`` is the episode's position in the extracted
arrays (``MixedDataset`` behind a ``Subset`` of fold positions); the tabular
baselines' is the ``labels.csv`` row. ``labels_rows`` maps the first to the
second at the dump, so the two must agree episode for episode.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',
                                'scripts'))

from TransEHR2.tabular import partition_rows                  # noqa: E402
from dump_transehr2_deephit_predictions import labels_rows    # noqa: E402


def write_split(tmp_path, selection, test_positions):
    np.save(tmp_path / 'extracted_rows.npy', selection)
    (tmp_path / 'fold1').mkdir()
    np.save(tmp_path / 'fold1' / 'fold1_test_rows.npy', test_positions)


def test_labels_rows_maps_positions_through_the_selection(tmp_path):
    # One episode per patient: labels.csv rows, sparse and ascending.
    selection = np.array([3, 8, 9, 20, 41, 57], dtype=np.int64)
    write_split(tmp_path, selection, np.array([4, 1, 5]))
    np.testing.assert_array_equal(
        labels_rows(str(tmp_path), np.array([4, 1, 5])), [41, 8, 57])


def test_transehr2_and_tabular_keys_agree(tmp_path):
    selection = np.array([0, 5, 6, 12, 13, 30, 31, 44], dtype=np.int64)
    positions = np.array([7, 2, 0, 5])
    write_split(tmp_path, selection, positions)
    # What a TransEHR2 loader yields as idx for the fold's test split ...
    transehr2 = labels_rows(str(tmp_path), positions)
    # ... and what the tabular baseline keys the same split by.
    tabular = partition_rows(str(tmp_path), 'fold1', 'test')
    np.testing.assert_array_equal(transehr2, tabular)
