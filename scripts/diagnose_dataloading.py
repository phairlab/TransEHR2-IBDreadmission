#!/usr/bin/env python3
"""Where a pretraining step's wall time goes, when it is not going to the GPU.

A head-to-head of 1 against 2 encoder blocks came back with the same epoch time,
while the attribution benchmark measured a second block at +40% of the pass. Those
only reconcile if pretraining is bound by something other than the model, and a
step has three candidates: reading the dense arrays off disk, building batches out
of them on the host, and copying them to the device. This times each.

PRINTS AGGREGATES ONLY. It opens the extracted arrays through the real dataset, so
it touches patient data, but it emits nothing but byte counts, rates and seconds.
No episode, timestep or feature value reaches stdout. Operator-run, by the same
rule the dedup report follows.

    python scripts/diagnose_dataloading.py --data-dir .../data --fold fold0

Stages, each answerable on its own:

``raw``      Reads episode rows straight off the memmapped arrays, no Dataset and
             no collation. The filesystem's ceiling for this access pattern, and
             the number every other stage should be read against.
``loader``   The real `prepare_dataloaders` path, swept over worker counts. If it
             lands near ``raw`` the storage is the limit and more workers will not
             help; if it lands far below, the cost is host-side and they will.
``device``   Host-to-device copy for one batch, which pinning and prefetch hide
             only if there is compute to hide them behind.
``batch``    The same loader swept over batch size instead of workers. Epoch time
             flat in batch size is an independent signature of an I/O bound, and
             it does not share the worker sweep's failure modes.
``profile``  cProfile over ``Dataset.__getitem__`` alone. The worker sweep says
             whether the cost is host-side; this says which part of it is, which
             matters because the one-hot expansion for categorical and ordinal
             features builds dense rows per item across the whole feature set.
``padding``  How much of what was read is padding. Every dense array is T wide for
             every episode regardless of length, so this bounds what truncating
             the extraction can win before anyone writes code for it.
"""

import argparse
import os
import statistics
import sys
import time

import numpy as np
import torch

import _path  # noqa: F401  (repository root on sys.path)

from TransEHR2.data.preprocessing import load_dataset, prepare_dataloaders
from TransEHR2.utils import move_batch_to_device

STAGES = ('raw', 'loader', 'batch', 'device', 'padding', 'profile')
GIB = 1024 ** 3


def nbytes(obj) -> int:
    """Bytes in a batch, walking the nested dicts and per-feature lists."""
    if torch.is_tensor(obj):
        return obj.element_size() * obj.nelement()
    if isinstance(obj, dict):
        return sum(nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(nbytes(v) for v in obj)
    return 0


def pick_device(requested):
    if requested != 'auto':
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def dense_arrays(extracted):
    """The per-episode arrays a batch is built from, largest first.

    The lookup family is stored sparse and the event branch is small, so the
    dense value arrays are what the read volume is made of.
    """
    paths = []
    for name in sorted(os.listdir(extracted)):
        if not name.endswith('.npy') or not name.startswith('val_'):
            continue
        path = os.path.join(extracted, name)
        paths.append((os.path.getsize(path), path))
    return [p for _, p in sorted(paths, reverse=True)]


def report_raw(args, rows):
    """Read `--batches` worth of episode rows off the memmaps and nothing else.

    Same row indices the sampler would draw, so the access pattern is the real
    one -- scattered rows of a large array, not a sequential scan. `np.asarray`
    on the slice is what forces the pages in; without it the memmap would be
    timed doing nothing.
    """
    arrays = [np.load(p, mmap_mode='r') for p in dense_arrays(args.extracted)]
    if not arrays:
        raise FileNotFoundError(f'no val_*.npy in {args.extracted}')

    print()
    print(f"raw: {len(arrays)} dense arrays, episode rows straight off the memmap")
    header = f"{'batches':>8}  {'GiB':>8}  {'s':>8}  {'GiB/s':>8}"
    print(header)
    print('-' * len(header))

    total_bytes, start = 0, time.perf_counter()
    for b in range(args.batches):
        picked = rows[b * args.batch_size:(b + 1) * args.batch_size]
        if picked.size == 0:
            break
        for array in arrays:
            block = np.asarray(array[picked])
            total_bytes += block.nbytes
    seconds = time.perf_counter() - start
    rate = total_bytes / seconds
    print(f"{args.batches:>8}  {total_bytes / GIB:8.2f}  {seconds:8.2f}  "
          f"{rate / GIB:8.2f}")
    # Pages written by the extraction, or by an earlier stage, are still resident,
    # and a read that never leaves RAM is not a storage ceiling. There is no way to
    # drop caches without root, so the guard is the number itself: past roughly
    # 8 GiB/s this is measuring memory.
    if rate / GIB > 8:
        print("  NOTE: above ~8 GiB/s this is page cache, not storage. Re-run with")
        print("        --batches large enough to read well past node RAM, or treat")
        print("        this as a floor on the real cold-read cost rather than a")
        print("        ceiling on throughput.")
    print(f"  read {total_bytes / GIB:.1f} GiB; compare against the node's RAM")
    return rate


def report_loader(args, n_train, raw_rate):
    """The real dataloader, swept over worker counts.

    Timed from the iterator rather than around it, so the first batch's worker
    startup is excluded: that cost is paid once per epoch and would otherwise
    dominate a short run and flatter the high-worker settings.
    """
    print()
    print(f"loader: prepare_dataloaders, batch {args.batch_size}, "
          f"{n_train} training episodes")
    header = (f"{'workers':>8}  {'s/batch':>9}  {'GiB/batch':>10}  {'GiB/s':>8}  "
              f"{'epoch min':>10}")
    print(header)
    print('-' * len(header))

    for workers in [int(w) for w in args.workers.split(',')]:
        loaders = prepare_dataloaders(args.data_dir, args.fold, args.batch_size,
                                      num_workers=workers,
                                      prefetch_factor=args.prefetch_factor)
        iterator = iter(loaders[0])
        batch = next(iterator)                       # startup, not timed
        per_batch = nbytes(batch)
        timings = []
        for _ in range(args.batches):
            start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            timings.append(time.perf_counter() - start)
        del iterator, loaders
        if not timings:
            continue
        seconds = statistics.median(timings)
        steps = max(1, -(-n_train // args.batch_size))
        print(f"{workers:>8}  {seconds:9.3f}  {per_batch / GIB:10.3f}  "
              f"{per_batch / GIB / seconds:8.2f}  "
              f"{steps * seconds / 60:10.1f}")
    if raw_rate:
        print(f"  raw ceiling for the same bytes: {raw_rate / GIB:.2f} GiB/s")


def report_batch(args, n_train):
    """The loader swept over batch size at one worker setting.

    A second, independent read on the same question the worker sweep asks. If
    seconds-per-episode is flat across batch sizes the step is paying per byte,
    which is storage; if it falls, there is per-batch overhead to amortize.
    """
    print()
    print(f"batch: prepare_dataloaders at {args.workers_for_device} workers")
    header = (f"{'batch':>8}  {'s/batch':>9}  {'ms/episode':>11}  {'GiB/s':>8}  "
              f"{'epoch min':>10}")
    print(header)
    print('-' * len(header))

    for batch_size in [int(b) for b in args.batch_sizes.split(',')]:
        loaders = prepare_dataloaders(args.data_dir, args.fold, batch_size,
                                      num_workers=args.workers_for_device,
                                      prefetch_factor=args.prefetch_factor)
        iterator = iter(loaders[0])
        batch = next(iterator)
        per_batch = nbytes(batch)
        timings = []
        for _ in range(args.batches):
            start = time.perf_counter()
            try:
                next(iterator)
            except StopIteration:
                break
            timings.append(time.perf_counter() - start)
        del iterator, loaders
        if not timings:
            continue
        seconds = statistics.median(timings)
        steps = max(1, -(-n_train // batch_size))
        print(f"{batch_size:>8}  {seconds:9.3f}  {1000 * seconds / batch_size:11.2f}  "
              f"{per_batch / GIB / seconds:8.2f}  {steps * seconds / 60:10.1f}")


def report_profile(args):
    """Where the time inside one ``__getitem__`` goes.

    Profiled on the main process with no workers, because a profile across worker
    processes reports the parent doing nothing. The ranking is what matters, not
    the absolute times -- profiling overhead inflates both.
    """
    import cProfile
    import pstats

    dataset = load_dataset(args.extracted, fold=None)
    rng = np.random.default_rng(args.seed)
    picks = rng.permutation(dataset.n_episodes)[:args.profile_items]

    profiler = cProfile.Profile()
    profiler.enable()
    for i in picks:
        dataset[int(i)]
    profiler.disable()

    print()
    print(f"profile: {len(picks)} __getitem__ calls, cumulative time")
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(15)


def report_device(args, device):
    """Host-to-device copy for one batch, pinned as the real loader pins it."""
    if device.type == 'cpu':
        print()
        print("device: skipped, no accelerator")
        return
    loaders = prepare_dataloaders(args.data_dir, args.fold, args.batch_size,
                                  num_workers=args.workers_for_device,
                                  prefetch_factor=args.prefetch_factor)
    batch = next(iter(loaders[0]))
    per_batch = nbytes(batch)

    timings = []
    for run in range(args.repeats + 1):
        start = time.perf_counter()
        move_batch_to_device(batch, device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        if run:
            timings.append(time.perf_counter() - start)
    seconds = statistics.median(timings)
    print()
    print(f"device: {per_batch / GIB:.3f} GiB to {device} in {1000 * seconds:.1f} ms "
          f"({per_batch / GIB / seconds:.2f} GiB/s)")


def report_padding(args, extracted):
    """How much of every batch is padding, from the masks alone.

    Dense arrays are T wide for every episode, so an episode of 51 timesteps in a
    300-wide array carries 83% padding and pays full price to read it. This is the
    ceiling on what a shorter extraction, or variable-length storage, could win.
    """
    masks = np.load(os.path.join(extracted, 'val_masks.npy'), mmap_mode='r')
    n_episodes, width = masks.shape
    sampled = min(args.padding_sample, n_episodes)
    lengths = (np.asarray(masks[:sampled]) > 0).sum(axis=1)

    print()
    print(f"padding: {sampled} episodes sampled, arrays {width} timesteps wide")
    print(f"  real timesteps  mean {lengths.mean():.1f}  "
          f"median {np.median(lengths):.0f}  p90 {np.quantile(lengths, 0.9):.0f}")
    print(f"  padding         {100 * (1 - lengths.mean() / width):.1f}% of every "
          f"byte read")
    for cut in (300, 200, 150, 100):
        if cut < width:
            kept = np.minimum(lengths, cut)
            print(f"  at T={cut:<4} {100 * (1 - cut / width):.0f}% fewer bytes, "
                  f"{100 * (1 - kept.mean() / lengths.mean()):.1f}% of real "
                  f"timesteps lost")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', required=True,
                        help="DATA_DIR from the dataset config; extracted/ sits under it.")
    parser.add_argument('--fold', default='fold0')
    parser.add_argument('--batch-size', type=int, default=200,
                        help="BATCH_SIZE from the experiment config.")
    parser.add_argument('--batches', type=int, default=20,
                        help='Batches to time per setting.')
    parser.add_argument('--workers', default='0,4,8,16',
                        help='num_workers settings to sweep.')
    parser.add_argument('--workers-for-device', type=int, default=4)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--batch-sizes', default='50,100,200,400',
                        help='Batch sizes for the batch stage.')
    parser.add_argument('--profile-items', type=int, default=200,
                        help='__getitem__ calls to profile.')
    parser.add_argument('--padding-sample', type=int, default=20000)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--stages', default='all',
                        help=f"Comma-separated from {', '.join(STAGES)}, or all.")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)

    args.extracted = os.path.join(args.data_dir, 'extracted')
    stages = STAGES if args.stages == 'all' else tuple(args.stages.split(','))
    device = pick_device(args.device)

    dataset = load_dataset(args.extracted, fold=None)
    n_episodes = dataset.n_episodes
    rng = np.random.default_rng(args.seed)
    rows = rng.permutation(n_episodes)[:args.batches * args.batch_size]

    print(f"data      {args.extracted}")
    print(f"episodes  {n_episodes}   fold {args.fold}   device {device}")

    raw_rate = None
    if 'raw' in stages:
        raw_rate = report_raw(args, rows)
    if 'loader' in stages:
        report_loader(args, n_episodes, raw_rate)
    if 'batch' in stages:
        report_batch(args, n_episodes)
    if 'device' in stages:
        report_device(args, device)
    if 'profile' in stages:
        report_profile(args)
    if 'padding' in stages:
        report_padding(args, args.extracted)
    return 0


if __name__ == '__main__':
    sys.exit(main())
