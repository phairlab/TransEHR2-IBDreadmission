#!/usr/bin/env python
"""Score the covariate-free reference a fold's integrated Brier scores are read against.

An IBS is dominated by how often the event happens: a rare outcome scores well under a
prediction of nearly zero for everyone, so the number means nothing until you know what a
model that used none of its inputs would have got. This computes that -- the Aalen-Johansen
marginal cumulative incidence, handed to every episode alike, scored with the same Brier code,
the same time grid and the same censoring weights as the model.

Given a model's score it also reports the Index of Prediction Accuracy,

    IPA = 1 - IBS_model / IBS_null

the fraction of the reference's error the model removes.

**No model is loaded and no GPU is used.** The reference depends only on the fold's outcomes,
so one run of this serves every cell of every sweep scored on that fold.

Usage:
    python scripts/marginal_reference.py <dataset_config> --fold fold0
    python scripts/marginal_reference.py <dataset_config> --fold fold0 \\
        --experiment_config <a tuning cell's config> --model_ibs 0.060461
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

import _path  # noqa: F401  (repository root on sys.path)

from TransEHR2.survival import (DEFAULT_BRIER_INTEGRATION_DAYS, DEFAULT_CAUSES,
                                DEFAULT_CUTS_DAYS, TimeGrid, metric_label,
                                marginal_reference_metrics)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('dataset_config',
                        help='The dataset config, for DATA_DIR.')
    parser.add_argument('--fold', default='fold0',
                        help='Which fold, e.g. fold0.')
    parser.add_argument('--split', default='val',
                        choices=('train', 'val', 'test'),
                        help='Which partition to score. val is the one the '
                             'tuning sweeps select on.')
    parser.add_argument('--experiment_config', default=None,
                        help='A config carrying TIME_GRID_CUTS_DAYS, '
                             'MODELLED_CAUSES and BRIER_INTEGRATION_DAYS. The '
                             'grid has to be the one the model was scored on, '
                             'or the two numbers are not comparable.')
    parser.add_argument('--model_ibs', type=float, default=None,
                        help="A model's integrated Brier score for the first "
                             'modelled cause. Given one, the IPA is reported.')
    parser.add_argument('--data_dir', default=None,
                        help="Override the dataset config's DATA_DIR.")
    args = parser.parse_args(argv)

    with open(args.dataset_config) as f_in:
        data_dir = args.data_dir or yaml.safe_load(f_in)['DATA_DIR']

    cuts, causes = list(DEFAULT_CUTS_DAYS), list(DEFAULT_CAUSES)
    brier_days = DEFAULT_BRIER_INTEGRATION_DAYS
    if args.experiment_config:
        with open(args.experiment_config) as f_in:
            cfg = yaml.safe_load(f_in)
        cuts = cfg.get('TIME_GRID_CUTS_DAYS', cuts)
        causes = cfg.get('MODELLED_CAUSES', causes)
        brier_days = cfg.get('BRIER_INTEGRATION_DAYS', brier_days)
    grid = TimeGrid(cuts, causes, brier_days)

    extracted = os.path.join(data_dir, 'extracted')
    rows_path = os.path.join(data_dir, args.fold,
                             f'{args.fold}_{args.split}_rows.npy')
    if not os.path.exists(rows_path):
        print(f"ERROR: no {rows_path}. The fold arrays are written by "
              f"IBDdataprep's split.py.", file=sys.stderr)
        return 1

    rows = np.load(rows_path)
    time_to_event = np.load(os.path.join(extracted, 'time_to_event.npy'),
                            mmap_mode='r')[rows]
    event_type = np.load(os.path.join(extracted, 'event_type.npy'),
                         mmap_mode='r')[rows]

    out = marginal_reference_metrics(np.asarray(time_to_event),
                                     np.asarray(event_type), grid)

    print(f"fold:     {args.fold} ({args.split})")
    print(f"episodes: {rows.size:,}")
    print(f"censored: {out['Censored']:,.0f}")
    print(f"grid:     cuts {list(grid.cuts)}, integrated to "
          f"{grid.brier_integration_days} days")
    print("")
    for name in grid.cause_names:
        key = metric_label(name)
        print(f"  {key}")
        print(f"    events:            {out[f'{key}_Events']:,.0f}")
        print(f"    IBS, no covariates: "
              f"{out[f'{key}_Integrated_Brier_Null']:.6f}")

    if args.model_ibs is not None:
        key = metric_label(grid.cause_names[0])
        null = out[f'{key}_Integrated_Brier_Null']
        if null > 0:
            print("")
            print(f"  model IBS {args.model_ibs:.6f} against a reference of "
                  f"{null:.6f}")
            print(f"  IPA       {1.0 - args.model_ibs / null:+.4f}  "
                  f"(the fraction of the reference's error removed)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
