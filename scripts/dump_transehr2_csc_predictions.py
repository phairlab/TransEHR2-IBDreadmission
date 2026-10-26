"""Dump a Cox-head TransEHR2 model in the cause-specific Cox baseline's form.

A model fitted with ``SURVIVAL_HEAD: cox`` (``fit_transehr2_deephit.py``)
outputs one log relative hazard per cause, eta_k. Per fold this writes what
``fit_baseline_csc.R`` writes for the CSC baseline (``TransEHR2.csc``), so
every later step treats the two alike:

* ``competing_risks_{split}_csc_lp.csv`` for train, val and test: idx,
  time_to_event, event_type and ``lp_<cause>``, the uncentred eta_k;
* ``competing_risks_csc_baseline_hazard.csv``: each cause's Breslow
  cumulative baseline hazard on the train partition (:func:`breslow`), at
  every train event time;
* the grid dumps, through ``dump_baseline_csc_predictions.write_fold_dumps``.

``recalibrate_predictions.py`` and ``evaluate_finetuned_predictions.py``
then recalibrate it with dense offsets and score it on its exact curve,
because the baseline file sits beside the dumps.

Usage:
    python scripts/dump_transehr2_csc_predictions.py <dataset_config> \\
        <experiment_config> <experiment_name> [--model_dir ./models] \\
        [--folds fold1] [--device cuda:0] [--num_workers 0] [--batch_size 750]
"""

import gc
import os

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

import _path  # noqa: F401  (repository root on sys.path)

from TransEHR2.cli import TASK, resolve_device
from TransEHR2.csc import BASELINE_FILE, LP_FILE
from TransEHR2.data.preprocessing import prepare_dataloaders
from TransEHR2.survival import TimeGrid, event_times_days
from TransEHR2.utils import move_batch_to_device

from dump_baseline_csc_predictions import write_fold_dumps
from dump_transehr2_deephit_predictions import (
    SPLITS, install_nan_hooks, parse_args, setup
)


def run_inference(model, loader: DataLoader, device: torch.device
                  ) -> Dict[str, np.ndarray]:
    """``{'idx', 'time_to_event', 'event_type', 'lp'}`` for one split."""
    model.eval()
    idx, times, types, lps = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='    Inference', leave=False):
            batch = move_batch_to_device(batch, device=device)
            eta = model(batch).double()
            if torch.isnan(eta).any():
                raise ValueError('NaN log hazards; the encoders produced NaN '
                                 'that the hooks did not catch')
            lps.append(eta.cpu().numpy())
            targets = batch['targets']
            idx.append(batch['idx'].cpu().numpy())
            times.append(targets['time_to_event'].reshape(-1).cpu().numpy())
            types.append(targets['event_type'].reshape(-1).cpu().numpy())
    return {'idx': np.concatenate(idx), 'time_to_event': np.concatenate(times),
            'event_type': np.concatenate(types),
            'lp': np.concatenate(lps, axis=0)}


def breslow(lp: np.ndarray, time_to_event: np.ndarray,
            event_type: np.ndarray, grid: TimeGrid
            ) -> Tuple[np.ndarray, np.ndarray]:
    """Each cause's Breslow cumulative baseline hazard, ``(times, cumhaz)``.

    .. math::
        \\Lambda_{0k}(t) = \\sum_{t_i \\le t,\\ \\delta_i = k}
            \\frac{1}{\\sum_{j: t_j \\ge t_i} \\exp \\eta_k(x_j)}

    ``times`` are the distinct event times of any modelled cause, in days
    truncated at the horizon as the evaluation reads them, and ``cumhaz``
    (n_causes, m) is uncentred: it pairs with exp(lp) as the lp were
    written. Ties are Breslow's, every tied event over the full tied risk
    set, as the training loss treats them.
    """
    _, cause_index, is_event = grid.discretize(
        torch.as_tensor(time_to_event, dtype=torch.float64),
        torch.as_tensor(event_type))
    cause_index, is_event = cause_index.numpy(), is_event.numpy()
    days = event_times_days(time_to_event, grid)

    times = np.unique(days[is_event])
    # Risk sums at each event time, from the latest backwards. Shifted by
    # each cause's largest lp so exp cannot overflow; the shift comes back
    # out of the hazard, which is then uncentred.
    shift = lp.max(axis=0)
    weights = np.exp(lp - shift)
    order = np.argsort(-days, kind='stable')
    cum = np.cumsum(weights[order], axis=0)
    # Everyone with days >= t: the prefix of the descending order up to the
    # last such episode.
    n_at_risk = np.searchsorted(-days[order], -times, side='right')
    risk = cum[n_at_risk - 1]                       # (m, n_causes)

    cumhaz = np.empty((grid.n_causes, times.size))
    for k in range(grid.n_causes):
        failed = is_event & (cause_index == k)
        deaths = np.bincount(np.searchsorted(times, days[failed]),
                             minlength=times.size)
        cumhaz[k] = np.cumsum(deaths / risk[:, k]) * np.exp(-shift[k])
    return times, cumhaz


def save_lp(path: str, data: Dict[str, np.ndarray], grid: TimeGrid) -> None:
    columns = {'idx': data['idx'], 'time_to_event': data['time_to_event'],
               'event_type': data['event_type']}
    for k, cause in enumerate(grid.cause_names):
        columns[f'lp_{cause}'] = data['lp'][:, k]
    order = np.argsort(data['idx'], kind='stable')
    pd.DataFrame(columns).iloc[order].to_csv(path, index=False)
    print(f"    Wrote {path} ({len(order):,} rows)")


def save_baseline(path: str, times: np.ndarray, cumhaz: np.ndarray,
                  grid: TimeGrid) -> None:
    columns = {'time_days': times}
    for k, cause in enumerate(grid.cause_names):
        columns[f'cumhaz_{cause}'] = cumhaz[k]
    pd.DataFrame(columns).to_csv(path, index=False)
    print(f"    Wrote {path} ({len(times):,} event times)")


def main():
    args = parse_args('Dump a Cox-head TransEHR2 model in the CSC form')
    device = resolve_device(args.device)
    print(f"Running inference on {device}")
    s = setup(args)
    if s['head'] != 'cox':
        raise ValueError(f"SURVIVAL_HEAD is {s['head']!r}; this dumps a Cox "
                         f"head. A DeepHit head's predictions come from "
                         f"dump_transehr2_deephit_predictions.py")
    grid = s['grid']

    for fold_name in s['folds']:
        print(f"\n=== {fold_name} ===")
        weights_path = (f'{args.model_dir}/{args.experiment_name}/'
                        f'{fold_name}/pretrained/finetuned_{TASK}.pt')
        if not os.path.exists(weights_path):
            print(f"  No finetuned weights at {weights_path}, skipping.")
            continue

        model = s['build_classifier']()
        # strict: a head of another width would be a different model.
        model.load_state_dict(
            torch.load(weights_path, map_location='cpu', weights_only=False))
        model = model.to(device)
        hooks = install_nan_hooks(model)

        loaders = prepare_dataloaders(
            s['data_dir'], fold_name, s['batch_size'],
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2 if args.num_workers > 0 else 1)
        if len(loaders) != 3:
            raise RuntimeError(f'{fold_name} has no validation split, which '
                               f'the recalibration fits on')

        folder = f'{args.model_dir}/{args.experiment_name}/{fold_name}/{TASK}'
        os.makedirs(folder, exist_ok=True)
        splits = {}
        for split, loader in zip(SPLITS, loaders):
            print(f"  {split}: {len(loader.dataset):,} episodes")
            splits[split] = run_inference(model, loader, device)
            save_lp(os.path.join(folder,
                                 LP_FILE.format(task=TASK, split=split)),
                    splits[split], grid)

        train = splits['train']
        save_baseline(os.path.join(folder, BASELINE_FILE.format(task=TASK)),
                      *breslow(train['lp'], train['time_to_event'],
                               train['event_type'], grid), grid)
        write_fold_dumps(folder, fold_name, grid)

        counts = hooks[0].nan_counts
        if any(counts.values()):
            print(f"  NaN encoder outputs zeroed: {counts}")
        for h in hooks:
            h.remove()
        del model, loaders, splits
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
