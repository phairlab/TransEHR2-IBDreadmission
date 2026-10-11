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
import pickle
import statistics
import sys
import time

import numpy as np
import torch

import _path  # noqa: F401  (repository root on sys.path)

from TransEHR2.data.preprocessing import load_dataset, prepare_dataloaders
from TransEHR2.utils import move_batch_to_device

STAGES = ('warm', 'raw', 'loader', 'batch', 'bytes', 'pin', 'device',
          'padding', 'profile')
GIB = 1024 ** 3
WARM_ROWS = 4096


def nbytes(obj) -> int:
    """Bytes in a batch, walking the nested dicts and per-feature lists."""
    if torch.is_tensor(obj):
        return obj.element_size() * obj.nelement()
    if isinstance(obj, dict):
        return sum(nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(nbytes(v) for v in obj)
    return 0


def byte_tree(obj, path=''):
    """``(path, bytes)`` per tensor, keeping the nesting in the name.

    The same walk ``nbytes`` does, reported leaf by leaf. A per-feature list
    collapses to one row with its length, because 94 rows of one number each
    says less than one row saying there are 94 of them.
    """
    if torch.is_tensor(obj):
        return [(path or '<tensor>', obj.element_size() * obj.nelement())]
    if isinstance(obj, dict):
        return [row for key, value in obj.items()
                for row in byte_tree(value, f'{path}.{key}' if path else str(key))]
    if isinstance(obj, (list, tuple)):
        # A long per-feature list collapses: 94 rows of one number each says
        # less than one row saying there are 94 of them. A short one does not,
        # because a two-element lookup list is text and drug -- different
        # answers to what to do about it, pooled into one number.
        if 0 < len(obj) <= 4:
            return [row for i, value in enumerate(obj)
                    for row in byte_tree(value, f'{path}[{i}]')]
        total = sum(nbytes(v) for v in obj)
        return [(f'{path}[{len(obj)}]', total)] if total else []
    return []


def report_bytes(args, n_episodes):
    """What the collated batch is made of, against what it was read from.

    The loader moves every one of these bytes from a worker to the main
    process, so a batch that is some multiple of what it was read from is
    paying that multiple in IPC. This says which tensors the multiple is in.
    """
    on_disk = sum(
        os.path.getsize(os.path.join(args.extracted, name))
        for name in os.listdir(args.extracted) if name.endswith('.npy')
    )
    loaders = prepare_dataloaders(args.data_dir, args.fold, args.batch_size,
                                  num_workers=0,
                                  prefetch_factor=args.prefetch_factor)
    batch = next(iter(loaders[0]))
    rows = sorted(byte_tree(batch), key=lambda row: -row[1])
    total = sum(size for _, size in rows)
    del loaders

    print()
    print(f"bytes: one collated batch of {args.batch_size}, largest first")
    header = f"{'tensor':<44}  {'MiB':>9}  {'share':>6}  {'KiB/episode':>12}"
    print(header)
    print('-' * len(header))
    for name, size in rows:
        if size / total < 0.005:
            continue
        print(f"{name:<44}  {size / 2**20:9.1f}  {size / total:6.1%}  "
              f"{size / args.batch_size / 1024:12.1f}")
    stored = on_disk / n_episodes
    collated = total / args.batch_size
    print(f"{'TOTAL':<44}  {total / 2**20:9.1f}  {1.0:6.1%}  "
          f"{collated / 1024:12.1f}")
    print(f"  {stored / 1024:.1f} KiB/episode on disk -> "
          f"{collated / 1024:.1f} KiB/episode collated ({collated / stored:.1f}x)")
    print("  every byte of the collated figure crosses worker IPC once per epoch")

    # A one-hot is its feature's level count wide in float32 where the stored
    # index is 2 bytes, so the expansion is levels * 2. Whether that is spread
    # over every feature or carried by a few decides whether the fix is to move
    # the expansion to the GPU or to look at a handful of columns.
    with open(os.path.join(args.extracted, 'metadata.pkl'), 'rb') as handle:
        metadata = pickle.load(handle)
    for family in ('categorical', 'ordinal'):
        dims = metadata.get(f'{family}_feat_dims') or []
        feats = metadata.get(f'{family}_feats') or []
        if not dims:
            continue
        widest = sorted(zip(feats, dims), key=lambda pair: -pair[1])
        total_levels = sum(dims)
        print(f"  {family}: {len(dims)} feature(s), {total_levels} level(s) "
              f"in all, widest " +
              ', '.join(f'{name} {width}' for name, width in widest[:5]))


def report_pin(args, n_train):
    """The same loader with pinning on and off.

    Pinning copies every byte the workers produced into page-locked memory, in
    the main process, once per batch. The worker sweep going flat past four
    workers says the main process is the limit; this says how much of that limit
    is the pin. If the gap is large the bytes are the problem and sending fewer
    of them is the fix; if it is small the cost is the handoff itself, and the
    same conclusion follows for a different reason -- which is why both arms are
    worth one run.
    """
    print()
    print(f"pin: prepare_dataloaders at {args.workers_for_device} workers")
    header = (f"{'pin_memory':>10}  {'s/batch':>9}  {'GiB/s':>8}  "
              f"{'epoch min':>10}")
    print(header)
    print('-' * len(header))

    for pin in (True, False):
        loaders = prepare_dataloaders(args.data_dir, args.fold, args.batch_size,
                                      num_workers=args.workers_for_device,
                                      pin_memory=pin,
                                      prefetch_factor=args.prefetch_factor)
        seconds, taken, per_batch = time_batches(loaders[0], args.batches)
        del loaders
        if seconds is None:
            continue
        steps = max(1, -(-n_train // args.batch_size))
        print(f"{str(pin):>10}  {seconds:9.3f}  "
              f"{per_batch / GIB / seconds:8.2f}  {steps * seconds / 60:10.1f}")
    print("  pinning is what makes the host-to-device copy fast, so off is a")
    print("  measurement and not a setting to adopt on its own")


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


def report_warm(args):
    """Read the whole cohort once, so every stage below starts from one cache state.

    Without it the arms of a sweep are not exchangeable. The first arm reads cold
    and leaves what it touched resident for the rest, which flatters whichever ran
    last -- and at 200 batches each arm touches a third of this cohort, so by the
    third arm most of it is in RAM. Two runs of the same sweep on a warm and a cold
    node differed 2.3x at 8 workers for no other reason.

    It is also the honest cold number: the whole cohort rather than the sample the
    raw stage draws, which is what a training run pays once in its first epoch
    before every later epoch runs warm.
    """
    names = sorted(n for n in os.listdir(args.extracted) if n.endswith('.npy'))
    total, start = 0, time.perf_counter()
    for name in names:
        array = np.load(os.path.join(args.extracted, name), mmap_mode='r')
        for begin in range(0, array.shape[0], WARM_ROWS):
            # np.array and not np.asarray: a basic slice of a memmap is a view,
            # so asarray returns the mapping itself and nothing is paged in. The
            # copy is what touches the pages, which is the entire point here --
            # the version that did not read 34.4 GiB in 0.1 s.
            total += int(np.array(array[begin:begin + WARM_ROWS]).nbytes)
        del array
    seconds = time.perf_counter() - start
    print()
    print(f"warm: {len(names)} array(s), {total / GIB:.1f} GiB read in {seconds:.1f} s "
          f"({total / GIB / seconds:.2f} GiB/s)")
    print("  the cohort is now resident, so every stage below measures the warm")
    print("  path -- the one a training run is on after its first epoch")
    # Above main memory's own bandwidth nothing was read: that is what a version
    # of this loop that produced views rather than copies reported, and the
    # arithmetic looked entirely reasonable until the rate did not.
    if total / GIB / seconds > 50:
        print("  WARNING: that is faster than memory. Nothing was paged in and")
        print("           the stages below are NOT warm.")


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
    on_disk = sum(
        os.path.getsize(os.path.join(args.extracted, name))
        for name in os.listdir(args.extracted) if name.endswith('.npy')
    )
    print(f"  read {total_bytes / GIB:.1f} GiB of {on_disk / GIB:.1f} GiB on disk")
    # Whatever this pass leaves resident, the loader stages read from RAM. So this
    # is a floor on the cold cost and they are an estimate of the warm one -- and
    # when the cohort fits in the job's memory, every epoch after the first is the
    # warm case regardless, which makes the warm number the one that matters.
    print("  the loader stages re-read this cohort, so they run against whatever")
    print("  this pass left in page cache: warm, not cold. Compare the size above")
    print("  against --mem to see which case a steady-state epoch is in.")
    return rate


def time_batches(loader, batches):
    """Seconds per batch, timed over the whole run rather than per call.

    A DataLoader with ``w`` workers and ``prefetch_factor`` p fills a queue of
    ``w * p`` batches before the first ``next`` returns, so the first ``w * p``
    gaps measure a dequeue and not the production being asked about. The median
    of the gaps lands inside that region whenever the run is short, which is how
    a 16-worker sweep over 20 batches once reported 588 GiB/s -- three orders
    above the storage under it. Total elapsed over the batches actually taken
    converges on the production rate instead.

    Returns ``(seconds per batch, batches taken, bytes per batch)``, the count so
    the caller can mark a row the queue was still covering.
    """
    iterator = iter(loader)
    batch = next(iterator)                           # startup, not timed
    per_batch = nbytes(batch)
    start = time.perf_counter()
    taken = 0
    for _ in range(batches):
        try:
            next(iterator)
        except StopIteration:
            break
        taken += 1
    elapsed = time.perf_counter() - start
    del iterator
    return (elapsed / taken if taken else None), taken, per_batch


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

    covered = False
    for workers in [int(w) for w in args.workers.split(',')]:
        loaders = prepare_dataloaders(args.data_dir, args.fold, args.batch_size,
                                      num_workers=workers,
                                      prefetch_factor=args.prefetch_factor)
        seconds, taken, per_batch = time_batches(loaders[0], args.batches)
        del loaders
        if seconds is None:
            continue
        queued = max(1, workers) * args.prefetch_factor
        covered = covered or taken <= 2 * queued
        steps = max(1, -(-n_train // args.batch_size))
        print(f"{workers:>8}  {seconds:9.3f}  {per_batch / GIB:10.3f}  "
              f"{per_batch / GIB / seconds:8.2f}  "
              f"{steps * seconds / 60:10.1f}"
              f"{'  *' if taken <= 2 * queued else ''}")
    if covered:
        print("  * fewer batches than twice the prefetch queue, so the run never")
        print("    outran what was queued before timing began. Raise --batches.")
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
        seconds, taken, per_batch = time_batches(loaders[0], args.batches)
        del loaders
        if seconds is None:
            continue
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

    if 'warm' in stages:
        report_warm(args)
    raw_rate = None
    if 'raw' in stages:
        raw_rate = report_raw(args, rows)
    if 'loader' in stages:
        report_loader(args, n_episodes, raw_rate)
    if 'batch' in stages:
        report_batch(args, n_episodes)
    if 'bytes' in stages:
        report_bytes(args, n_episodes)
    if 'pin' in stages:
        report_pin(args, n_episodes)
    if 'device' in stages:
        report_device(args, device)
    if 'profile' in stages:
        report_profile(args)
    if 'padding' in stages:
        report_padding(args, args.extracted)
    return 0


if __name__ == '__main__':
    sys.exit(main())
