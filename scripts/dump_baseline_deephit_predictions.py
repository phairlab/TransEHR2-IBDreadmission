"""Write the tabular DeepHit baseline's predictions to CSV, one file per split.

Loads what ``fit_baseline_deephit.py`` left per fold -- the weights and the
encoding fitted on train -- and scores train, val and test with
``dump_transehr2_deephit_predictions.py``'s own ``run_inference`` and
``save_predictions_csv``. The columns are that script's, written to
``{model_dir}/{experiment}/{fold}/competing_risks/
competing_risks_{split}_finetuned_output.csv``, where
``recalibrate_predictions.py`` and ``evaluate_finetuned_predictions.py``
read them.

``idx`` is the ``labels.csv`` row, the key of ``baseline_features.csv``.
The encoding is applied, never refitted, so a re-dump is identical.

Usage:
    python scripts/dump_baseline_deephit_predictions.py <dataset_config> \\
        <experiment_config> <experiment_name> [--model_dir ./models] \\
        [--folds fold1] [--device cuda:0] [--batch_size 200]
"""

import argparse
import os
import pandas as pd
import torch
import yaml

import _path  # noqa: F401  (repository root on sys.path)

from TransEHR2.cli import TASK, TUNING_FOLD, get_fold_names, resolve_device
from TransEHR2.tabular import (
    ENCODING_FILE, LABELS_FILE, WEIGHTS_FILE, BaselineEncoding, build_model,
    config_grid, load_split, make_loader, read_baseline_features
)

from dump_transehr2_deephit_predictions import (
    SPLITS, run_inference, save_predictions_csv
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Extract predictions from fitted tabular DeepHit models')
    parser.add_argument('dataset_config', type=str,
                        help='YAML file giving DATA_DIR')
    parser.add_argument('experiment_config', type=str,
                        help='YAML file of the experiment, part1_tabdh.yaml')
    parser.add_argument('experiment_name', type=str,
                        help='Name of the experiment (locates model weights '
                             'under model_dir)')
    parser.add_argument('--model_dir', type=str, default='./models',
                        help='Root directory containing saved model weights')
    parser.add_argument('--folds', type=str, nargs='+', default=None,
                        help='Which folds to score. Default is every fold '
                             'in DATA_DIR but the tuning fold.')
    parser.add_argument('--device', type=str, default=None,
                        help='Device for inference, e.g. cuda:0')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Inference batch size. Defaults to the '
                             'experiment config\'s BATCH_SIZE.')
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    print(f"Running inference on {device}")

    with open(args.dataset_config, 'r') as f_in:
        data_dir = yaml.safe_load(f_in)['DATA_DIR']
    with open(args.experiment_config, 'r') as f_in:
        cfg = yaml.safe_load(f_in)
    batch_size = args.batch_size or cfg['BATCH_SIZE']
    grid = config_grid(cfg)

    features = read_baseline_features(
        os.path.join(data_dir, cfg['BASELINE_FEATURES_FILE']))
    labels = pd.read_csv(os.path.join(data_dir, LABELS_FILE),
                         usecols=['TIME_TO_EVENT', 'EVENT_TYPE'])

    for fold_name in args.folds or get_fold_names(data_dir,
                                                  exclude=[TUNING_FOLD]):
        print(f"\n=== {fold_name} ===")
        folder = os.path.join(args.model_dir, args.experiment_name,
                              fold_name, TASK)
        weights_path = os.path.join(folder, WEIGHTS_FILE)
        if not os.path.exists(weights_path):
            print(f"  No fitted weights at {weights_path}, skipping.")
            continue

        encoding = BaselineEncoding.load(os.path.join(folder, ENCODING_FILE))
        model = build_model(len(encoding.columns()), grid, cfg)
        # strict: a grid or head edited since fitting changes a shape, and a
        # loose load would score a randomly initialized head.
        model.load_state_dict(
            torch.load(weights_path, map_location='cpu', weights_only=True))
        model = model.to(device)

        for split in SPLITS:
            dataset = load_split(data_dir, features, labels, encoding,
                                 fold_name, split)
            print(f"  {split}: {len(dataset):,} episodes")
            idx, times, types, cif, pmf, p_survive = run_inference(
                model, make_loader(dataset, batch_size), grid, device)
            save_predictions_csv(
                idx, times, types, cif, pmf, p_survive, grid,
                os.path.join(folder, f'{TASK}_{split}_finetuned_output.csv'))


if __name__ == '__main__':
    main()
