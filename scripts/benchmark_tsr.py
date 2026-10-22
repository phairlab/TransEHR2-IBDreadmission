"""What TSR costs per episode, against the real model and the real arrays.

Replaces the synthetic stage-1 path in `benchmark_grad_attribution.py` for the
question "what will an attribution run cost". That one builds a model by hand at
hardcoded widths and feeds it generated tensors at an assumed density, which is
why its numbers went stale the moment the model changed: nothing in it reads the
experiment config, so nothing in it noticed. This builds the classifier with
`render_token_attributions.build_classifier` -- the same construction the
production attribution path uses, config-driven -- and reads real episodes
through `prepare_dataloaders`. It cannot drift from the thing it measures.

Weights are not loaded by default. A forward and a backward cost the same
whatever the weights say, and the cell list TSR walks comes from the data
(`tsr._occupied`), not from the parameters. Pass `--weights` to time a specific
checkpoint; the number should not move.

THE THREE BUCKETS

  warmup   the first `--warmup` batches, reported and then discarded. They carry
           CUDA context creation, cuDNN algorithm selection, dataloader worker
           spawn, and whatever of the cohort was not yet in page cache. None of
           it repeats, so none of it belongs in a per-episode rate.
  load     the loader's own production rate, measured with no compute in the
           loop. This is what the GPU would wait on if it were infinitely fast.
  compute  `tsr_scores` on a batch already resident on the device, with a
           synchronize either side so the backward is actually finished.

The pipeline column is the interleaved loop -- load, move, attribute -- and is
what a real run takes. It is at most load + compute and at best max(load,
compute), the difference being how much the prefetch hides.

BALANCE IS NOT A FREE PARAMETER

A batch of B episodes does not cost B times a batch of 1. `tsr_scores` perturbs
the *union* of the batch's occupied cells and runs every item through every
perturbation, so an item pays for its neighbours' cells as well as its own, and
per-episode compute grows with B while per-episode load stays flat. Balancing
the two therefore means finding where a rising curve meets a flat one, and if
compute already exceeds load at B=1 there is nothing to find: the answer is
B=1 and the loader has headroom to spare. The sweep reports both so the
question is settled by the numbers rather than assumed either way.

    sbatch SLURM/slurm_benchmark_tsr.sh
"""

import argparse
import os
import statistics
import time

import torch

import _path  # noqa: F401  (repository root on sys.path)

from TransEHR2.cli import resolve_device
from TransEHR2.data.preprocessing import prepare_dataloaders
from TransEHR2.survival import TimeGrid, cif_logit_target
from TransEHR2.tsr import tsr_scores
from TransEHR2.utils import move_batch_to_device

from render_token_attributions import build_classifier, load_config


def cell(value, width, digits=3):
    """A number, or a dash where there was nothing to measure.

    `--warmup 0` and a split shorter than one batch both leave a column with no
    samples, and a dash says that where a 0.000 would claim a measurement.
    """
    if value is None:
        return f"{'-':>{width}}"
    return f"{value:{width}.{digits}f}"


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def split_loader(loaders, split):
    """The requested partition, whether or not a validation split exists."""
    names = ('train', 'val', 'test') if len(loaders) == 3 else ('train', 'test')
    return dict(zip(names, loaders))[split]


def attribute(model, batch, target, device, args):
    """One batch attributed, timed with the backward actually finished."""
    synchronize(device)
    start = time.perf_counter()
    out = tsr_scores(model, batch, target=target, quantile=args.quantile,
                     gated_value=args.gated_value, chunk=args.chunk,
                     ig_steps=args.ig_steps)
    synchronize(device)
    return time.perf_counter() - start, int(out['passes']), int(out['calls'])


def time_load_only(loader, batches, warmup):
    """The loader's production rate, with nothing consuming the GPU.

    Timed over the whole run rather than per call: a loader with workers hands
    back `workers * prefetch_factor` batches before it has produced anything, so
    the first gaps measure a queue draining rather than the rate being asked
    about.
    """
    iterator = iter(loader)
    warm_start = time.perf_counter()
    taken = 0
    for _ in range(warmup):
        try:
            next(iterator)
        except StopIteration:
            break
        taken += 1
    warm = time.perf_counter() - warm_start

    start, counted = time.perf_counter(), 0
    for _ in range(batches):
        try:
            next(iterator)
        except StopIteration:
            break
        counted += 1
    elapsed = time.perf_counter() - start
    return (elapsed / counted if counted else None), (warm / taken if taken else None)


def time_pipeline(model, loader, target, device, args):
    """The interleaved loop, split into the wait, the copy, and the attribution."""
    iterator = iter(loader)
    warm_start = time.perf_counter()
    warmed = 0
    for _ in range(args.warmup):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        attribute(model, move_batch_to_device(batch, device), target, device, args)
        warmed += 1
    warm = time.perf_counter() - warm_start

    waits, copies, computes, passes, calls = [], [], [], [], []
    for _ in range(args.batches):
        start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        waited = time.perf_counter() - start

        start = time.perf_counter()
        batch = move_batch_to_device(batch, device)
        synchronize(device)
        copied = time.perf_counter() - start

        computed, n_passes, n_calls = attribute(model, batch, target, device, args)
        waits.append(waited)
        copies.append(copied)
        computes.append(computed)
        passes.append(n_passes)
        calls.append(n_calls)
    return {
        'warm': warm / warmed if warmed else None,
        'wait': statistics.median(waits) if waits else None,
        'copy': statistics.median(copies) if copies else None,
        'compute': statistics.median(computes) if computes else None,
        'passes': statistics.median(passes) if passes else None,
        'calls': statistics.median(calls) if calls else None,
        'batches': len(waits),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset_config')
    parser.add_argument('experiment_config')
    parser.add_argument('--weights', default=None,
                        help='Checkpoint to load. Omitted: random weights, which cost the same.')
    parser.add_argument('--fold', default='fold0')
    parser.add_argument('--split', default='test', choices=('train', 'val', 'test'))
    parser.add_argument('--batch-sizes', default='1,2,4,8',
                        help='Episodes per attribution call, swept.')
    parser.add_argument('--batches', type=int, default=8,
                        help='Batches timed per setting, after the warmup.')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--chunk', type=int, default=1024,
                        help='Perturbation replicas per model call.')
    parser.add_argument('--ig-steps', type=int, default=0,
                        help='0 is grad x input; >0 integrates over that many alphas.')
    parser.add_argument('--quantile', type=float, default=0.0)
    parser.add_argument('--gated-value', type=float, default=0.0)
    parser.add_argument('--cause', type=int, default=0)
    parser.add_argument('--horizon-bin', type=int, default=0)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    dataset_config = load_config(args.dataset_config)
    experiment_config = load_config(args.experiment_config)
    data_dir = dataset_config['DATA_DIR']

    # Only the keys the config actually carries. `.get` returning None would
    # override TimeGrid's own defaults with None rather than fall back to them,
    # which fails inside the constructor rather than where it was caused.
    grid_keys = {'TIME_GRID_CUTS_DAYS': 'cuts_days',
                 'MODELLED_CAUSES': 'causes',
                 'BRIER_INTEGRATION_DAYS': 'brier_integration_days'}
    grid = TimeGrid(**{argument: experiment_config[key]
                       for key, argument in grid_keys.items()
                       if experiment_config.get(key) is not None})
    target = cif_logit_target(args.cause, args.horizon_bin, grid.n_causes, grid.n_bins)

    model = build_classifier(dataset_config, experiment_config, grid, data_dir)
    if args.weights:
        model.load_state_dict(
            torch.load(args.weights, map_location='cpu', weights_only=False))
    model = model.to(device).eval()

    method = 'grad x input' if args.ig_steps == 0 else f'IG x{args.ig_steps}'
    print(f"device {device}   split {args.split}   fold {args.fold}")
    print(f"method {method}   chunk {args.chunk}   quantile {args.quantile}   "
          f"workers {args.workers}")
    print(f"weights {args.weights or 'random (timing is weight-independent)'}")
    print(f"horizon bin {args.horizon_bin} of {grid.n_bins}, cause "
          f"{args.cause} of {grid.n_causes}")

    header = (f"{'batch':>6}  {'warmup/b':>9}  {'load/b':>8}  {'wait/b':>8}  "
              f"{'h2d/b':>7}  {'tsr/b':>9}  {'tsr/ep':>9}  {'passes/ep':>10}  "
              f"{'calls':>6}  {'bound':>8}")
    print()
    print(header)
    print('-' * len(header))

    for batch_size in [int(b) for b in args.batch_sizes.split(',')]:
        loaders = prepare_dataloaders(data_dir, args.fold, batch_size,
                                      num_workers=args.workers,
                                      prefetch_factor=args.prefetch_factor)
        load_rate, _ = time_load_only(split_loader(loaders, args.split),
                                      args.batches, args.warmup)
        del loaders

        loaders = prepare_dataloaders(data_dir, args.fold, batch_size,
                                      num_workers=args.workers,
                                      prefetch_factor=args.prefetch_factor)
        timings = time_pipeline(model, split_loader(loaders, args.split),
                                target, device, args)
        del loaders
        if timings['compute'] is None:
            print(f"{batch_size:>6}  no batches; is {args.split} smaller than one batch?")
            continue

        bound = 'compute' if timings['compute'] >= (load_rate or 0) else 'load'
        print(f"{batch_size:>6}  {cell(timings['warm'], 9, 2)}  "
              f"{cell(load_rate, 8)}  {cell(timings['wait'], 8)}  "
              f"{cell(timings['copy'], 7)}  {cell(timings['compute'], 9)}  "
              f"{cell(timings['compute'] / batch_size, 9)}  "
              f"{cell(timings['passes'] / batch_size, 10, 0)}  "
              f"{cell(timings['calls'], 6, 0)}  {bound:>8}")

    print()
    print("load/b is the loader alone; wait/b is what the pipeline actually blocked")
    print("for, so wait << load means the prefetch hid it. Balance is load/b against")
    print("tsr/b in the same row, and tsr/ep is the number an attribution run scales by.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
