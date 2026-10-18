"""Fit the tabular DeepHit baseline on each fold's train partition.

TransEHR2's DeepHit head, read straight off the baseline features
(``baseline_features.csv``, from IBDdataprep's ``lace.py``) instead of the
EHR encoder's output. Everything else is TransEHR2's finetuning, unchanged:
the same ``finetune_model`` loop, so the same loss (sigma, rank weight),
Adam, gradient clipping, learning-rate half-life, selection metric and
early stopping on val, on the same ``TimeGrid``. The settings are the
experiment config's (``configs/experiments/part1_tabdh.yaml``).

Per fold, under ``{MODEL_DIR}/{EXPERIMENT_NAME}/{fold}/competing_risks/``:

    baseline_encoding.json              the encoding fitted on train
                                        (TransEHR2.tabular.BaselineEncoding)
    finetuned_competing_risks.pt        the best epoch's weights
    evaluation/evaluation_competing_risks.yaml
                                        the settings and that epoch's train
                                        and val scores

``dump_baseline_deephit_predictions.py`` reads the first two. A fold whose
weights exist is skipped.

``--lr_check`` is the plan's small learning-rate check, and the only tuning
this model gets: one fit on the tuning fold per ``LR_CHECK_RATES`` value,
each under ``{MODEL_DIR}/{EXPERIMENT_NAME}_lr_check/lr<rate>/``, with the
best validation selection metric per rate in ``lr_check.yaml`` there. The
chosen rate goes into the config by hand.

Usage:
    python scripts/fit_baseline_deephit.py <dataset_config> <experiment_config> \\
        [--folds fold1 fold2] [--device cuda:0] [--lr_check]
"""

import argparse
import os
import pandas as pd
import torch
import yaml

from torch.utils.tensorboard import SummaryWriter
from typing import Dict

import _path  # noqa: F401  (repository root on sys.path)

from TransEHR2.cli import TASK, TUNING_FOLD, get_fold_names, resolve_device
from TransEHR2.routines import SELECTION_METRICS, finetune_model
from TransEHR2.tabular import (
    ENCODING_FILE, LABELS_FILE, WEIGHTS_FILE, BaselineEncoding, build_model,
    config_grid, load_split, make_loader, partition_rows,
    read_baseline_features
)
from TransEHR2.utils import convert_to_python_types


def fit_fold(cfg: Dict, data_dir: str, features: pd.DataFrame,
             labels: pd.DataFrame, fold_name: str, experiment_name: str,
             learning_rate: float, device: torch.device) -> str:
    """Fit one fold, unless its weights exist; the path of its evaluation."""
    grid = config_grid(cfg)
    out_dir = os.path.join(cfg['MODEL_DIR'], experiment_name, fold_name, TASK)
    weights_path = os.path.join(out_dir, WEIGHTS_FILE)
    evaluation_path = os.path.join(out_dir, 'evaluation',
                                   f'evaluation_{TASK}.yaml')
    if os.path.exists(weights_path):
        print(f"{fold_name}: weights found at {weights_path}, skipping.")
        return evaluation_path
    if not os.path.exists(os.path.join(data_dir, fold_name,
                                       f'{fold_name}_val_rows.npy')):
        raise RuntimeError(
            f'{fold_name} has no validation split, which early stopping '
            f"selects on; rebuild the folds with IBDdataprep's split.py.")

    train_rows = partition_rows(data_dir, fold_name, 'train')
    encoding = BaselineEncoding.fit(
        features.loc[train_rows], cfg['LINEAR_FEATURES'],
        cfg['MEDIAN_FILL_FEATURES'], cfg['CATEGORICAL_FEATURES'])
    encoding.save(os.path.join(out_dir, ENCODING_FILE))
    train, val = (load_split(data_dir, features, labels, encoding, fold_name,
                             split) for split in ('train', 'val'))
    print(f"{fold_name}: {len(train):,} train and {len(val):,} val episodes, "
          f"{len(encoding.columns())} encoded columns")

    torch.manual_seed(cfg['SEED'])
    model = build_model(len(encoding.columns()), grid, cfg)
    loaders = (make_loader(train, cfg['BATCH_SIZE'], shuffle_seed=cfg['SEED']),
               make_loader(val, cfg['BATCH_SIZE']))

    writer = SummaryWriter(
        f'./log/{experiment_name}/{fold_name}/finetuned_{TASK}')
    train_scores, val_scores = finetune_model(
        model=model, save_path=weights_path, loaders=loaders, grid=grid,
        writer=writer, learning_rate=learning_rate, device=device,
        rank_weight=cfg['DEEPHIT_RANK_WEIGHT'], sigma=cfg['DEEPHIT_SIGMA'],
        cause_weights=cfg['DEEPHIT_CAUSE_WEIGHTS'],
        selection_metric=cfg['FINETUNE_SELECTION_METRIC'],
        lr_half_life=cfg['FINETUNE_LR_HALF_LIFE'],
        total_epoch=cfg['FINETUNE_TOTAL_EPOCH'],
        checkpoint_dir=f'./checkpoints/{experiment_name}/{fold_name}/finetuned')
    writer.close()

    os.makedirs(os.path.dirname(evaluation_path), exist_ok=True)
    with open(evaluation_path, 'w') as f_out:
        yaml.dump({
            'task': TASK,
            'fold': fold_name,
            'experiment': experiment_name,
            'time_grid_cuts_days': list(map(float, grid.cuts)),
            'modelled_causes': list(grid.cause_names),
            'brier_integration_days': grid.brier_integration_days,
            'selection_metric': cfg['FINETUNE_SELECTION_METRIC'],
            'learning_rate': learning_rate,
            'seed': cfg['SEED'],
            'encoded_columns': encoding.columns(),
            'train_scores': convert_to_python_types(train_scores or {}),
            'validation_scores': convert_to_python_types(val_scores or {}),
        }, f_out, default_flow_style=False, indent=2)
    return evaluation_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Fit the tabular DeepHit baseline, per fold')
    parser.add_argument('dataset_config', type=str,
                        help='YAML file giving DATA_DIR')
    parser.add_argument('experiment_config', type=str,
                        help='YAML file of the experiment, part1_tabdh.yaml')
    parser.add_argument(
        '--folds', type=str, nargs='+', default=None,
        help='Which folds to fit. Default is every fold in DATA_DIR but '
             'the tuning fold.')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to train on, e.g. cuda:0')
    parser.add_argument(
        '--lr_check', action='store_true',
        help='Fit the tuning fold once per LR_CHECK_RATES value instead.')
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    print(f"Training on {device}")

    with open(args.dataset_config, 'r') as f_in:
        data_dir = yaml.safe_load(f_in)['DATA_DIR']
    with open(args.experiment_config, 'r') as f_in:
        cfg = yaml.safe_load(f_in)

    features = read_baseline_features(
        os.path.join(data_dir, cfg['BASELINE_FEATURES_FILE']))
    labels = pd.read_csv(os.path.join(data_dir, LABELS_FILE),
                         usecols=['TIME_TO_EVENT', 'EVENT_TYPE'])

    if args.lr_check:
        name = f"{cfg['EXPERIMENT_NAME']}_lr_check"
        metric_key = SELECTION_METRICS[cfg['FINETUNE_SELECTION_METRIC']][0]
        results = {}
        for rate in cfg['LR_CHECK_RATES']:
            print(f"\n=== {TUNING_FOLD}, learning rate {rate:g} ===")
            evaluation_path = fit_fold(cfg, data_dir, features, labels,
                                       TUNING_FOLD, f'{name}/lr{rate:g}',
                                       float(rate), device)
            with open(evaluation_path, 'r') as f_in:
                scores = yaml.safe_load(f_in)['validation_scores']
            results[float(rate)] = float(scores.get(metric_key, float('nan')))
        path = os.path.join(cfg['MODEL_DIR'], name, 'lr_check.yaml')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f_out:
            yaml.safe_dump({'fold': TUNING_FOLD, 'metric': metric_key,
                            'val_by_learning_rate': results}, f_out,
                           sort_keys=False)
        print(f"\nBest val {metric_key} by learning rate: {results}")
        print(f"Wrote {path}")
        return

    fold_names = args.folds or get_fold_names(data_dir,
                                              exclude=[TUNING_FOLD])
    if not fold_names:
        raise FileNotFoundError(f'No fold directories found in {data_dir}')
    for fold_name in fold_names:
        print(f"\n=== {fold_name} ===")
        fit_fold(cfg, data_dir, features, labels, fold_name,
                 cfg['EXPERIMENT_NAME'], cfg['FINETUNE_LEARNING_RATE'],
                 device)


if __name__ == '__main__':
    main()
