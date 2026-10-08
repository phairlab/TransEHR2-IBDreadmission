"""Write each finetuned model's predictions to CSV, one file per split.

What a competing-risks model predicts is not a number per episode but a
surface: for each cause, the cumulative incidence at every bin on the time
grid. That whole surface is written, because the questions asked of it
downstream ("who is in the top decile of 90-day readmission risk?", "how
does the readmission curve bend once death competes?") each need a
different slice of it and none of them can be recovered from a single
column.

Columns, one row per episode:

    idx                     row of labels.csv, the canonical episode key
    time_to_event           minutes from INDEX_TIME, as labelled
    event_type              0 censored, 1 readmission, 2 death, 3 out-migration
    cif_<cause>_<cut>       F_k(t) at each cut, per cause
    p_survive               probability of reaching the horizon event-free

One process, one GPU: fold-level parallelism is the caller's, as in
run_experiment.py.

Usage:
    python scripts/dump_finetuned_predictions.py <dataset_config> <experiment_config> \\
        <experiment_name> [--model_dir ./models] [--folds fold0] \\
        [--device cuda:0] [--num_workers 0] [--batch_size 750]
"""

import argparse
import gc
import numpy as np
import os
import pandas as pd
import torch
import yaml

from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple

import _path  # noqa: F401  (repository root on sys.path)

from TransEHR2.cli import (
    TASK, TUNING_FOLD, get_fold_names, resolve_device
)
from TransEHR2.data.preprocessing import (
    compute_static_feat_dims, lookup_feat_widths, prepare_dataloaders,
    value_encoder_dims
)
from TransEHR2.models import MixedClassifier
from TransEHR2.modules import DeepHitHead, EventDataEncoder, ValueDataEncoder
from TransEHR2.survival import (
    DEFAULT_CAUSES, DEFAULT_CUTS_DAYS, TimeGrid, cif_from_pmf,
    deephit_distribution, format_days
)
from TransEHR2.utils import move_batch_to_device



SPLITS = ('train', 'val', 'test')


def install_nan_hooks(model: MixedClassifier) -> List:
    """Install forward hooks that replace NaN encoder output with zeros.

    A query row with no attendable key softmaxes over nothing, and
    ``torch.nn.MultiheadAttention`` returns NaN for it.  Those NaN values
    survive the subsequent ``val_enc * mask`` operation because
    ``NaN * 0 = NaN`` in IEEE 754, and then ``torch.sum`` propagates the
    NaN to every prediction in the batch.

    Two changes have narrowed what can still reach this hook.  The permute
    that made each *timestep* a batch item is gone, so a timestep padded
    across the whole batch no longer masks every key; and the encoders now
    own their attention, giving fully masked rows the zero vector rather
    than NaN.  The remaining route is ``norm='BatchNorm'``, which still
    delegates to ``torch.nn.MultiheadAttention`` -- no shipped config
    selects it.  The hook is kept as a backstop, and its counter says
    whether anything is still arriving.

    Args:
        model: A MixedClassifier whose encoders may produce NaN at
            fully-padded timesteps.

    Returns:
        List of hook handles (call ``.remove()`` on each to uninstall).
    """
    nan_counts: Dict[str, int] = {'val_encoder': 0, 'event_encoder': 0}

    def _make_hook(name: str):
        def hook(module, inp, output):
            if isinstance(output, torch.Tensor):
                n = torch.isnan(output).sum().item()
                if n > 0:
                    nan_counts[name] += n
                    return torch.nan_to_num(output, nan=0.0)
            return output
        return hook

    handles = [
        model.val_encoder.register_forward_hook(_make_hook('val_encoder')),
        model.event_encoder.register_forward_hook(_make_hook('event_encoder')),
    ]
    # Expose the counter dict so callers can inspect it later.
    for h in handles:
        h.nan_counts = nan_counts  # type: ignore[attr-defined]
    return handles


def run_inference(
    model: MixedClassifier,
    loader: DataLoader,
    grid: TimeGrid,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score one split.

    Returns:
        ``(idx, time_to_event, event_type, cif, p_survive)`` with ``cif``
        of shape (n, n_causes, n_bins) and the rest one-dimensional.
    """
    model.eval()
    idx, times, types, cifs, survives = [], [], [], [], []
    nan_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc='    Inference',
                                       leave=False)):
            batch = move_batch_to_device(batch, device=device)
            logits = model(batch)

            n_nan = int(torch.isnan(logits).sum().item())
            if n_nan > 0:
                nan_batches += 1
                if i == 0:
                    print(f"    WARNING: {n_nan} NaN values in logits "
                          f"(batch 0, shape {tuple(logits.shape)})")
            elif i == 0:
                print(f"    logits OK (batch 0): "
                      f"min={logits.min().item():.4f}, "
                      f"max={logits.max().item():.4f}")

            pmf, p_survive = deephit_distribution(
                logits.double(), grid.n_causes, grid.n_bins)
            cifs.append(cif_from_pmf(pmf).cpu().numpy())
            survives.append(p_survive.cpu().numpy())

            targets = batch['targets']
            idx.append(batch['idx'].cpu().numpy())
            times.append(targets['time_to_event'].reshape(-1).cpu().numpy())
            types.append(targets['event_type'].reshape(-1).cpu().numpy())

            del batch, logits

    if nan_batches:
        print(f"    WARNING: {nan_batches} batches produced NaN logits")

    return (np.concatenate(idx), np.concatenate(times),
            np.concatenate(types), np.concatenate(cifs, axis=0),
            np.concatenate(survives))


def save_predictions_csv(
    idx: np.ndarray,
    time_to_event: np.ndarray,
    event_type: np.ndarray,
    cif: np.ndarray,
    p_survive: np.ndarray,
    grid: TimeGrid,
    output_path: str
) -> None:
    """Write one split's predictions.

    Rows come out in ``idx`` order, which is ``labels.csv`` order, so two
    splits of the same fold concatenate into a cohort-wide frame without a
    join.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    order = np.argsort(idx, kind='stable')
    columns = {
        'idx': idx[order],
        'time_to_event': time_to_event[order],
        'event_type': event_type[order],
    }
    for k, cause in enumerate(grid.cause_names):
        for b, cut in enumerate(grid.cuts):
            name = f"cif_{cause}_{format_days(float(cut)).replace(' ', '')}"
            columns[name] = cif[order, k, b]
    columns['p_survive'] = p_survive[order]

    pd.DataFrame(columns).to_csv(output_path, index=False)
    print(f"    Wrote {output_path} ({len(order):,} rows)")


def main():
    parser = argparse.ArgumentParser(
        description='Extract predictions from finetuned TransEHR2 models'
    )
    parser.add_argument('dataset_config', type=str,
                        help='YAML file specifying dataset parameters')
    parser.add_argument('experiment_config', type=str,
                        help='YAML file specifying experiment parameters')
    parser.add_argument('experiment_name', type=str,
                        help='Name of the experiment (locates model weights '
                             'under model_dir)')
    parser.add_argument('--model_dir', type=str, default='./models',
                        help='Root directory containing saved model weights')
    parser.add_argument('--folds', type=str, nargs='+', default=None,
                        help='Which folds to score. Default is all of them.')
    parser.add_argument('--device', type=str, default=None,
                        help='Device for inference, e.g. cuda:0')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader worker processes (default: 0)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Inference batch size. Defaults to the '
                             'experiment config\'s BATCH_SIZE.')
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Running inference on {device}")

    with open(args.dataset_config, 'r') as f_in:
        dataset_config = yaml.safe_load(f_in)
    DATA_DIR = dataset_config['DATA_DIR']
    VARIABLE_PROPERTIES_PATH = dataset_config['VARIABLE_PROPERTIES_PATH']
    VALUED_FEATS = dataset_config['VALUED_FEATS']
    EVENT_FEATS = dataset_config['EVENT_FEATS']
    STATIC_FEATS = dataset_config['STATIC_FEATS']

    with open(args.experiment_config, 'r') as f_in:
        experiment_config = yaml.safe_load(f_in)
    BATCH_SIZE = args.batch_size or experiment_config['BATCH_SIZE']
    DISCRIMINATOR_ENCODER_D_MODEL = experiment_config['DISCRIMINATOR_ENCODER_D_MODEL']
    DISCRIMINATOR_ENCODER_N_HEADS = experiment_config['DISCRIMINATOR_ENCODER_N_HEADS']
    DISCRIMINATOR_ENCODER_N_ENCODER_BLOCKS = experiment_config['DISCRIMINATOR_ENCODER_N_ENCODER_BLOCKS']
    DISCRIMINATOR_ENCODER_DIM_FEEDFORWARD = experiment_config['DISCRIMINATOR_ENCODER_DIM_FEEDFORWARD']
    DISCRIMINATOR_ENCODER_DROPOUT = experiment_config['DISCRIMINATOR_ENCODER_DROPOUT']
    DISCRIMINATOR_ENCODER_ACTIVATION = experiment_config['DISCRIMINATOR_ENCODER_ACTIVATION']
    DISCRIMINATOR_ENCODER_NORM = experiment_config['DISCRIMINATOR_ENCODER_NORM']
    DISCRIMINATOR_ENCODER_NORM_FIRST = experiment_config.get('DISCRIMINATOR_ENCODER_NORM_FIRST', True)
    THP_ENCODER_D_MODEL = experiment_config['THP_ENCODER_D_MODEL']
    THP_ENCODER_D_INNER = experiment_config['THP_ENCODER_D_INNER']
    THP_ENCODER_N_LAYERS = experiment_config['THP_ENCODER_N_LAYERS']
    THP_ENCODER_N_HEADS = experiment_config['THP_ENCODER_N_HEADS']
    THP_ENCODER_D_K = experiment_config['THP_ENCODER_D_K']
    THP_ENCODER_D_V = experiment_config['THP_ENCODER_D_V']
    THP_ENCODER_DROPOUT = experiment_config['THP_ENCODER_DROPOUT']
    THP_ENCODER_NORM_FIRST = experiment_config.get('THP_ENCODER_NORM_FIRST', True)
    PREDICTOR_AGGREGATION_METHOD = experiment_config['PREDICTOR_AGGREGATION_METHOD']
    DEEPHIT_HEAD_D_SHARED = experiment_config.get('DEEPHIT_HEAD_D_SHARED', [256, 128])
    DEEPHIT_HEAD_D_CAUSE = experiment_config.get('DEEPHIT_HEAD_D_CAUSE', [128, 64])
    DEEPHIT_HEAD_DROPOUT = experiment_config.get('DEEPHIT_HEAD_DROPOUT', 0.1)

    grid = TimeGrid(
        experiment_config.get('TIME_GRID_CUTS_DAYS',
                              list(DEFAULT_CUTS_DAYS)),
        experiment_config.get('MODELLED_CAUSES', list(DEFAULT_CAUSES)))

    with open(VARIABLE_PROPERTIES_PATH, 'r') as f_in:
        variable_properties = yaml.safe_load(f_in)
    static_dim = sum(
        compute_static_feat_dims(variable_properties, STATIC_FEATS))

    # Excluded for the same reason run_experiment.py excludes it: these
    # predictions are what the reported metrics are computed from, and the
    # tuning fold has no place in them. --folds reaches it when wanted.
    fold_name_list = args.folds or get_fold_names(DATA_DIR,
                                                  exclude=[TUNING_FOLD])

    # The whole lookup family, whenever the extraction carries it: see
    # run_experiment.py. The weights being loaded here were built from
    # the same two numbers, so a short encoder fails the load rather
    # than predicting wrongly -- but it fails on every fold, which is
    # not a useful way to find out.
    lookup_dims = lookup_feat_widths(os.path.join(DATA_DIR, 'extracted'))
    use_lookup = bool(lookup_dims)
    n_val_feats, tot_val_feat_dim = value_encoder_dims(
        variable_properties, VALUED_FEATS, lookup_dims)
    n_event_types = len(EVENT_FEATS)

    def build_classifier() -> MixedClassifier:
        return MixedClassifier(
            event_encoder=EventDataEncoder(
                num_types=n_event_types, d_model=THP_ENCODER_D_MODEL,
                d_inner=THP_ENCODER_D_INNER, n_layers=THP_ENCODER_N_LAYERS,
                n_head=THP_ENCODER_N_HEADS, d_k=THP_ENCODER_D_K,
                d_v=THP_ENCODER_D_V, dropout=THP_ENCODER_DROPOUT,
                normalize_before=THP_ENCODER_NORM_FIRST),
            val_encoder=ValueDataEncoder(
                n_features=n_val_feats, feat_dim=tot_val_feat_dim,
                d_model=DISCRIMINATOR_ENCODER_D_MODEL,
                n_heads=DISCRIMINATOR_ENCODER_N_HEADS,
                n_encoder_blocks=DISCRIMINATOR_ENCODER_N_ENCODER_BLOCKS,
                dim_feedforward=DISCRIMINATOR_ENCODER_DIM_FEEDFORWARD,
                dropout=DISCRIMINATOR_ENCODER_DROPOUT,
                activation=DISCRIMINATOR_ENCODER_ACTIVATION,
                norm=DISCRIMINATOR_ENCODER_NORM,
                normalize_before=DISCRIMINATOR_ENCODER_NORM_FIRST),
            d_event_enc=THP_ENCODER_D_MODEL,
            d_val_enc=DISCRIMINATOR_ENCODER_D_MODEL,
            d_statics=static_dim,
            num_classes=grid.n_causes * grid.n_bins,
            aggr=PREDICTOR_AGGREGATION_METHOD,
            use_lookup=use_lookup,
            head=DeepHitHead(
                d_in=MixedClassifier.encoding_width(
                    THP_ENCODER_D_MODEL, DISCRIMINATOR_ENCODER_D_MODEL,
                    static_dim),
                n_causes=grid.n_causes, n_bins=grid.n_bins,
                d_shared=DEEPHIT_HEAD_D_SHARED, d_cause=DEEPHIT_HEAD_D_CAUSE,
                dropout=DEEPHIT_HEAD_DROPOUT),
        )

    for fold_name in fold_name_list:
        print(f"\n=== {fold_name} ===")
        weights_path = (f'{args.model_dir}/{args.experiment_name}/'
                        f'{fold_name}/pretrained/finetuned_{TASK}.pt')
        if not os.path.exists(weights_path):
            print(f"  No finetuned weights at {weights_path}, skipping.")
            continue

        model = build_classifier()
        # strict: a grid edited since finetuning changes the head's width,
        # and a loose load would score a randomly initialized head.
        model.load_state_dict(
            torch.load(weights_path, map_location='cpu', weights_only=False))
        model = model.to(device)
        hooks = install_nan_hooks(model)

        loaders = prepare_dataloaders(
            DATA_DIR, fold_name, BATCH_SIZE, num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2 if args.num_workers > 0 else 1)
        # prepare_dataloaders drops the validation split when a fold has
        # none, so pair loaders with names from the end.
        names = SPLITS if len(loaders) == 3 else ('train', 'test')

        for split, loader in zip(names, loaders):
            print(f"  {split}: {len(loader.dataset):,} episodes")
            idx, times, types, cif, p_survive = run_inference(
                model, loader, grid, device)
            save_predictions_csv(
                idx, times, types, cif, p_survive, grid,
                f'{args.model_dir}/{args.experiment_name}/{fold_name}/'
                f'{TASK}/{TASK}_{split}_finetuned_output.csv')

        counts = hooks[0].nan_counts
        if any(counts.values()):
            print(f"  NaN encoder outputs zeroed: {counts}")
        for h in hooks:
            h.remove()

        del model, loaders
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
