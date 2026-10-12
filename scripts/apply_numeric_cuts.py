#!/usr/bin/env python
"""Re-measure each fold's standardization statistics under a cut table.

``extract_data.py`` writes ``summary_statistics_fold{i}.npz`` on its way past,
before any cut table exists -- the table is calibrated *from* the finished
arrays, so it cannot inform the pass that produced them. This is the second
pass. It reads the extracted arrays memory-mapped, applies the cuts, and
rewrites each fold's statistics with ``cut_low`` and ``cut_high`` stored
beside them, which is what makes the loader clip.

**Nothing in ``extracted/`` is modified** except the per-fold statistics files.
That is the whole point of a load-time filter: re-tuning a threshold re-runs
this script, not the ~75 GB extraction.

Statistics and cuts travel in one file deliberately. The mean is the one
statistic here that a sentinel can move -- ``feature_scale`` reads percentiles,
which it cannot -- so statistics measured over uncut values would centre a
feature the loader then cuts on a mean the cuts would have changed. Keeping
them together means neither can be applied without the other.

Running this with no ``--cuts`` restores the statistics to what extraction
wrote, which is how a cut table is withdrawn.

Usage:
    python scripts/apply_numeric_cuts.py <data_dir> \\
        --cuts <data_dir>/numeric_cuts.csv
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

import _path  # noqa: F401  (repository root on sys.path)

from TransEHR2.data.preprocessing import read_numeric_cuts, standardize_feats

from extract_data import find_fold_train_rows


def open_arrays(extracted_dir: str, n_feats: int) -> dict:
    """The two arrays ``standardize_feats`` reads, memory-mapped.

    Read-only and never written: one fold's statistics are a reduction over
    the cohort's values, and the cohort's values are not this script's to
    change.
    """
    return {
        'val_numeric_indicators': np.load(
            os.path.join(extracted_dir, 'val_numeric_indicators.npy'),
            mmap_mode='r'
        ),
        'val_numeric_values': [
            np.load(
                os.path.join(extracted_dir, f'val_numeric_values_{f}.npy'),
                mmap_mode='r'
            )
            for f in range(n_feats)
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('data_dir',
                        help="The directory holding extracted/ and fold*/.")
    parser.add_argument('--cuts', default=None,
                        help='The table calibrate_outlier_filter.py '
                             '--emit_cuts wrote. Omit to restore the '
                             'statistics extraction wrote.')
    args = parser.parse_args(argv)

    extracted_dir = os.path.join(args.data_dir, 'extracted')
    with open(os.path.join(extracted_dir, 'metadata.pkl'), 'rb') as f_in:
        metadata = pickle.load(f_in)
    names = list(metadata['numeric_feats'])

    cuts = None
    if args.cuts:
        cuts = read_numeric_cuts(args.cuts, names)
        acting = int(np.isfinite(cuts[0]).sum() + np.isfinite(cuts[1]).sum())
        print(f"{args.cuts} cuts {acting} bound(s) over "
              f"{len(names)} numeric feature(s).")
    else:
        print("No cut table given; the statistics will be restored to what "
              "extraction measured.")

    fold_train_rows = find_fold_train_rows(args.data_dir)
    if not fold_train_rows:
        print(f"ERROR: no fold*/fold*_train_rows.npy under {args.data_dir}. "
              f"Statistics are per fold, over that fold's training rows "
              f"alone.", file=sys.stderr)
        return 1

    arrays = open_arrays(extracted_dir, len(names))
    for fold_name, rows in fold_train_rows.items():
        stats_path = os.path.join(
            extracted_dir, f'summary_statistics_{fold_name}.npz'
        )
        standardize_feats(arrays, len(names), np.asarray(rows), stats_path,
                          cuts=cuts)
        print(f"Wrote {stats_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
