#!/usr/bin/env python
"""Measure the post-merge text dedup ratio (section 8, item 2).

Section 4.5 sizes ``text_embeddings.npy`` from a *pre-merge* count --
AMB 1,784,100 + DAD 269,234 + CLM 1,900,036, about 3.95 M unique strings
-- and says that number is a **floor**: section 2.3 merges the three
sources' ``TEXT_SUPERFEATURE`` fields at a shared timestamp, and a merged
string is a new string. How far above the floor the real figure sits
decides whether the table is 10^6 or 10^7 rows, which is the difference
between a 65 GB artifact and a 650 GB one.

This walks Stage B's ``root/`` and counts. It reads real patient records
and so is **run by the user, not by Claude**; it prints counts and sizes
only, never a string, a PATID or a row.

Strings are held as 16-byte BLAKE2b digests rather than as themselves:
at 4 M merged code descriptions the set would otherwise be several GB,
and a digest collision at that scale is far below the precision the
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
        description="Post-merge text dedup ratio (section 8, item 2)"
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

    # Section 4.5's table, at the measured U rather than at its floor.
    gb = 1024 ** 3
    print(f"\nProjected table sizes at U = {unique}:")
    print(f"  text_tokens.npy      int32 x 1024 : "
          f"{unique * 1024 * 4 / gb:8.1f} GB")
    for width in (4096, 8192):
        print(f"  text_embeddings.npy  fp32  x {width:<4} : "
              f"{unique * width * 4 / gb:8.1f} GB")
    print(f"\nSection 4.5's pre-merge floor is 3,950,000 unique strings; "
          f"this run is {unique / 3_950_000:.2f}x that.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
