#!/usr/bin/env python
"""Count how many unique text strings the cohort contains.

``text_embeddings.npy`` has one row per unique ``TEXT_SUPERFEATURE``
string, so that count is what sizes it -- the difference between a table
of roughly 10^6 rows and one of 10^7 is the difference between a 65 GB
artifact and a 650 GB one, and between a few GPU-hours and something that
does not finish.

The count is not knowable from the source tables alone. Records from
different sources that land on the same minute have their text fields
merged into one string, and a merged string is a new string, so the sum
of the sources' own unique counts is a floor rather than an estimate.
This walks the prepared per-patient timeseries and counts what is
actually there.

It reads real patient records and so is **run by the operator, not by an
assistant**; it prints counts and sizes only, never a string, a patient
identifier or a row.

Strings are held as 16-byte BLAKE2b digests rather than as themselves: at
a few million merged code descriptions the set would otherwise be several
GB, and a digest collision at that scale is far below the precision the
answer is wanted to.

Usage:
    python text_dedup_report.py --root /path/to/data/root
"""

import argparse
import hashlib
import os
import pandas as pd
import sys


COLUMN = 'TEXT_SUPERFEATURE'


def scan(root: str, column: str = COLUMN):
    """Return ``(n_patients, n_rows, n_nonblank, unique_digests)``."""
    digests = set()
    n_patients = n_rows = n_nonblank = 0

    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry, 'timeseries.csv')
        if not os.path.isfile(path):
            continue
        n_patients += 1
        frame = pd.read_csv(path, usecols=lambda c: c == column,
                            dtype=str, keep_default_na=False)
        if column not in frame.columns:
            continue
        n_rows += len(frame)
        for value in frame[column]:
            value = value.strip()
            if not value:
                continue
            n_nonblank += 1
            digests.add(
                hashlib.blake2b(value.encode('utf-8'),
                                digest_size=16).digest()
            )
        if n_patients % 5000 == 0:
            print(f"  {n_patients} patients, {len(digests)} unique so far",
                  flush=True)

    return n_patients, n_rows, n_nonblank, digests


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Count the cohort's unique text strings"
    )
    parser.add_argument('--root', '-r', required=True,
                        help="Stage B's data/root/ directory")
    parser.add_argument('--column', default=COLUMN,
                        help=f"Text column to count (default: {COLUMN})")
    args = parser.parse_args(argv)

    n_patients, n_rows, n_nonblank, digests = scan(args.root, args.column)
    unique = len(digests)

    print(f"\nPatients scanned      : {n_patients}")
    print(f"Timeseries rows       : {n_rows}")
    print(f"Non-blank {args.column:<12}: {n_nonblank}")
    print(f"Unique strings (U)    : {unique}")
    if unique:
        print(f"Dedup ratio           : {n_nonblank / unique:.2f}x")
        print(f"Text-bearing timesteps: "
              f"{100 * n_nonblank / max(n_rows, 1):.1f}% of rows")

    # What the measured U costs, per stored artifact.
    gb = 1024 ** 3
    print(f"\nProjected table sizes at U = {unique}:")
    print(f"  text_tokens.npy      int32 x 1024 : "
          f"{unique * 1024 * 4 / gb:8.1f} GB")
    for width in (4096, 8192):
        print(f"  text_embeddings.npy  fp32  x {width:<4} : "
              f"{unique * width * 4 / gb:8.1f} GB")
    print(f"\nThe pre-merge floor for this cohort is 3,950,000 unique "
          f"strings; this run is {unique / 3_950_000:.2f}x that.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
