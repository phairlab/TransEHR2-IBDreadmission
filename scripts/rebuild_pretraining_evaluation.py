#!/usr/bin/env python
"""Rebuild a sweep's ``evaluation_pretrained.yaml`` files from its TensorBoard logs.

``fit_transehr2_deephit.py`` writes each epoch's losses twice: to
``{MODEL_DIR}/{EXPERIMENT_NAME}/{fold}/pretrained/evaluation/`` once at the end,
and to ``./log/{EXPERIMENT_NAME}/{fold}/pretrained/`` every epoch as TensorBoard
scalars. The second copy is complete and lives in a different tree, so a sweep
whose model directories were lost can still be ranked.

**It reconstructs the artifact rather than the ranking.** The files written here
are the ones ``hp_tuning.results`` already reads, so
``report_tuning_results.py`` and ``select_tuned_cell.py`` work unmodified. A
parallel ranking path would be a second implementation of the selection rule to
keep in step with the first.

The best epoch is found by replaying ``is_improvement`` over the epochs in
order, not by ``argmin``. The two differ: the training loop only accepts an
epoch that improves on the incumbent by more than a relative threshold, so a
later epoch better by a hair is not the one the loop recorded, and the losses
written here have to be the ones that were selected rather than the ones that
were smallest.

``settings`` is read from each trial's own config rather than from TensorBoard,
which carries no hyperparameters.

Usage:
    python scripts/rebuild_pretraining_evaluation.py <manifest> [--log_root ./log]
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

import _path  # noqa: F401  (repository root on sys.path)

from TransEHR2.hp_tuning.results import evaluation_path
from TransEHR2.hp_tuning.spec import load_manifest
from TransEHR2.routines import is_improvement

# The seven terms `_mean_losses` records, which is what the evaluation file
# carries. Named here rather than imported as `_EMPTY_PRETRAIN_LOSSES`' keys so
# that a term absent from an older run's logs is a reported gap rather than a
# KeyError at ranking time.
LOSS_TERMS = (
    'Optimization_Loss', 'Generator_Loss', 'Discriminator_Loss',
    'THP_Loss', 'THP_NLL_Loss', 'THP_Type_Loss', 'THP_Time_Loss',
)


def read_scalars(log_dir: str):
    """Every ``<term>/<split>`` series in one run's event files, keyed by epoch.

    Returns:
        ``{split: {term: {epoch: value}}}`` for the splits present.
    """
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator
    )

    # size_guidance 0 for scalars means "keep them all": the default downsamples
    # to 1000 points, and a 200-epoch run would survive that while a longer one
    # would silently lose the epoch the loop actually selected.
    accumulator = EventAccumulator(log_dir, size_guidance={'scalars': 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get('scalars', []))

    out = {'train': {}, 'val': {}}
    for split in out:
        for term in LOSS_TERMS:
            tag = f'{term}/{split}'
            if tag not in available:
                continue
            # Last write wins, so a cell that resumed from a checkpoint and
            # rewrote an epoch reports the value the resumed run produced.
            out[split][term] = {e.step: float(e.value) for e in
                                accumulator.Scalars(tag)}
    return out


def best_epoch_losses(scalars):
    """Replay the training loop's selection over the recorded epochs.

    Returns:
        ``(epoch, train_losses, val_losses)``, or ``(None, None, None)`` when
        there is no validation ``Optimization_Loss`` to select on.
    """
    val_optimization = scalars['val'].get('Optimization_Loss')
    if not val_optimization:
        return None, None, None

    best_epoch, best_value = None, float('inf')
    for epoch in sorted(val_optimization):
        if is_improvement(val_optimization[epoch], best_value):
            best_epoch, best_value = epoch, val_optimization[epoch]

    if best_epoch is None:
        return None, None, None

    def at(split):
        return {term: scalars[split][term][best_epoch]
                for term in LOSS_TERMS
                if best_epoch in scalars[split].get(term, {})}

    return best_epoch, at('train'), at('val')


def settings_from_config(config_path: str):
    """The settings block `write_pretraining_evaluation` records, from the config.

    Keys and spelling follow its call site in ``fit_transehr2_deephit.py``:
    the file is read by eye as often as by machine, and two spellings of the
    same run would be worse than none.
    """
    with open(config_path) as f_in:
        config = yaml.safe_load(f_in)
    return {
        'learning_rate': config['PRETRAIN_LEARNING_RATE'],
        'lr_half_life': config['PRETRAIN_LR_HALF_LIFE'],
        'total_epoch': config['PRETRAIN_TOTAL_EPOCH'],
        'disc_loss_weight': config['DISC_LOSS_WEIGHT'],
        'record_mask_ratio': config['RECORD_MASK_RATIO'],
        'obs_unobs_sample_ratio': config['OBS_UNOBS_SAMPLE_RATIO'],
        'cmpnt_mask_ratio': config['CMPNT_MASK_RATIO'],
        'generator_encoder_blocks':
            config['GENERATOR_ENCODER_N_ENCODER_BLOCKS'],
        'discriminator_encoder_blocks':
            config['DISCRIMINATOR_ENCODER_N_ENCODER_BLOCKS'],
        'use_thp': config['USE_THP'],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('manifest', help="The sweep's manifest.")
    parser.add_argument('--log_root', default='./log',
                        help="Where fit_transehr2_deephit.py wrote its TensorBoard "
                             "logs. Relative to the repository root, which is "
                             "the directory the job ran from.")
    parser.add_argument('--overwrite', action='store_true',
                        help='Rewrite an evaluation file that is already '
                             'there. Off by default: a rebuilt file must '
                             'never quietly replace one a real run wrote.')
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    fold = manifest['fold']

    written, skipped, missing = 0, 0, []
    for trial in manifest['trials']:
        path = evaluation_path(manifest, trial, 'pretrain')
        if os.path.exists(path) and not args.overwrite:
            print(f"  {trial['name']}: already has {path}; left alone.")
            skipped += 1
            continue

        log_dir = os.path.join(args.log_root, trial['name'], fold,
                               'pretrained')
        if not os.path.isdir(log_dir):
            missing.append(f"{trial['name']}: no {log_dir}")
            continue

        epoch, train_losses, val_losses = best_epoch_losses(
            read_scalars(log_dir))
        if epoch is None:
            missing.append(f"{trial['name']}: {log_dir} has no "
                           f"Optimization_Loss/val")
            continue

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f_out:
            yaml.safe_dump({
                'task': 'pretrain',
                'fold': fold,
                'experiment': trial['name'],
                'settings': settings_from_config(trial['config']),
                'train_losses': train_losses,
                'val_losses': val_losses,
                # Not written by fit_transehr2_deephit.py. A reader comparing these
                # against a later run's needs to know these were recovered
                # from the logs rather than written by the run itself.
                'rebuilt_from_tensorboard': {
                    'log_dir': os.path.abspath(log_dir),
                    'best_epoch': int(epoch),
                },
            }, f_out, sort_keys=False)
        print(f"  {trial['name']}: epoch {epoch}, "
              f"val Optimization_Loss {val_losses['Optimization_Loss']:.4f}")
        written += 1

    print(f"\nRebuilt {written}, left {skipped} alone, "
          f"{len(missing)} without logs.")
    for line in missing:
        print(f"  MISSING {line}", file=sys.stderr)
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
