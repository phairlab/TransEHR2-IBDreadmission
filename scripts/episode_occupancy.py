#!/usr/bin/env python3
"""How many (timestep, feature) cells an episode actually observes.

TSR costs ``1 + T + (observed cells)`` per episode, because deleting a cell that is zero in
both its value and its indicator cannot move the saliency map (see `TransEHR2.tsr._occupied`).
That makes occupancy the dominant term in the whole explanation budget, and it is a property of
the data rather than of the model -- so no benchmark against a synthetic batch can supply it.
This reads it off the extracted arrays.

PRINTS AGGREGATES ONLY. It opens the indicator arrays, which say which feature was observed
for which patient at which minute, and emits nothing but counts, means and quantiles over the
cohort. No episode, timestep or feature value is written to stdout. Operator-run, by the same
rule the dedup report follows.

    python scripts/episode_occupancy.py --data-dir /path/to/extracted
    python scripts/episode_occupancy.py --data-dir ... --max-steps 300

Padding is excluded: `val_masks` says which timesteps are real, and an episode is explained at
its own length rather than at the padded width. ``--max-steps`` reports the same numbers for a
truncated window, which is the lever T gives you.
"""

import argparse
import glob
import os
import sys

import numpy as np

import _path  # noqa: F401  (repository root on sys.path)


def indicator_paths(data_dir):
    """Every value-associated indicator array, which is one per family.

    The event branch is excluded: TSR never perturbs it, so its occupancy does not enter the
    pass count.
    """
    paths = sorted(glob.glob(os.path.join(data_dir, '*_indicators.npy')))
    return [p for p in paths if 'event' not in os.path.basename(p)]


def occupancy(data_dir, max_steps=None, block=256):
    """``(lengths, observed per episode, n_features)`` over the cohort.

    Episodes are read in blocks so that a cohort whose dense arrays run to tens of gigabytes
    never has more than a block of them resident.
    """
    masks = np.load(os.path.join(data_dir, 'val_masks.npy'), mmap_mode='r')
    arrays = [np.load(p, mmap_mode='r') for p in indicator_paths(data_dir)]
    if not arrays:
        raise FileNotFoundError(f'no *_indicators.npy in {data_dir}')

    n_episodes = masks.shape[0]
    n_features = sum(a.shape[-1] for a in arrays)
    lengths = np.zeros(n_episodes, dtype=np.int64)
    observed = np.zeros(n_episodes, dtype=np.int64)

    for start in range(0, n_episodes, block):
        stop = min(start + block, n_episodes)
        mask = np.asarray(masks[start:stop]) > 0
        if max_steps is not None:
            # Extraction left-pads -- `save_extracted` writes `val_masks[row, start:] = 1`
            # and leaves the left of the row zero -- so the most recent timesteps are at the
            # right end of the row and a recent window is a suffix, not a prefix.
            keep = np.zeros_like(mask)
            keep[:, -max_steps:] = True
            mask &= keep
        lengths[start:stop] = mask.sum(axis=1)
        for array in arrays:
            counts = (np.asarray(array[start:stop]) != 0).sum(axis=-1)
            observed[start:stop] += (counts * mask).sum(axis=1)
    return lengths, observed, n_features


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', required=True,
                        help='Directory holding the extracted *.npy arrays.')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Truncate each episode to its most recent this-many timesteps.')
    parser.add_argument('--block', type=int, default=256,
                        help='Episodes held in memory at once.')
    parser.add_argument('--ig-steps', type=int, default=0,
                        help='Report the pass count for integrated gradients at this many '
                             'path points instead of grad x input.')
    args = parser.parse_args(argv)

    lengths, observed, n_features = occupancy(args.data_dir, args.max_steps, args.block)
    real = lengths > 0
    if not real.any():
        print('No episodes with an unpadded timestep.')
        return 1

    lengths, observed = lengths[real], observed[real]
    cells = lengths.astype(np.int64) * n_features
    density = observed / np.maximum(cells, 1)
    multiplier = args.ig_steps or 1
    passes = multiplier * (1 + lengths + observed)
    dense = multiplier * (1 + lengths + cells)

    def quantiles(values, fmt='{:.0f}'):
        qs = np.quantile(values, [0.5, 0.9, 0.99])
        return '  '.join(fmt.format(q) for q in qs)

    print(f"episodes          {real.sum()}")
    print(f"features          {n_features}")
    print(f"timesteps         mean {lengths.mean():.1f}   "
          f"median/p90/p99 {quantiles(lengths)}")
    print(f"density           mean {density.mean():.4f}   "
          f"median/p90/p99 {quantiles(density, '{:.4f}')}")
    print()
    print(f"TSR passes per episode ({'IG x%d' % args.ig_steps if args.ig_steps else 'grad x input'})")
    print(f"  measured        mean {passes.mean():.0f}   median/p90/p99 {quantiles(passes)}")
    print(f"  if fully dense  mean {dense.mean():.0f}")
    print(f"  skipped         {100 * (1 - passes.sum() / dense.sum()):.1f}%")
    print()
    print(f"cohort total      {passes.sum():,} passes")
    print("  multiply by the measured ms/pass/item for the wall time of one horizon")
    return 0


if __name__ == '__main__':
    sys.exit(main())
