"""Recovering a sweep's ranking from its TensorBoard logs.

``fit_transehr2_deephit.py`` writes each epoch's losses to two trees: the
evaluation YAML beside the weights, and TensorBoard scalars under ``./log``.
The second is complete, so losing the first costs the encoders but not the
ranking.

What these pin is that the recovery rebuilds *the artifact the ranking machinery
already reads*, with the epoch the training loop would have selected -- which is
not the smallest validation loss, because ``is_improvement`` only accepts an
epoch that beats the incumbent by more than a relative threshold.
"""

import subprocess
import sys

import numpy as np
import pytest
import yaml

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

SCRIPT = (Path(__file__).resolve().parents[2]
          / 'scripts' / 'rebuild_pretraining_evaluation.py')

TERMS = ('Optimization_Loss', 'Generator_Loss', 'Discriminator_Loss',
         'THP_Loss', 'THP_NLL_Loss', 'THP_Type_Loss', 'THP_Time_Loss')


def _config(path: Path, lr=0.002, half_life=480):
    path.write_text(yaml.safe_dump({
        'PRETRAIN_LEARNING_RATE': lr, 'PRETRAIN_LR_HALF_LIFE': half_life,
        'PRETRAIN_TOTAL_EPOCH': 200, 'DISC_LOSS_WEIGHT': 1.0,
        'RECORD_MASK_RATIO': 0.15, 'OBS_UNOBS_SAMPLE_RATIO': 10.0,
        'CMPNT_MASK_RATIO': 0.25,
        'GENERATOR_ENCODER_N_ENCODER_BLOCKS': 1,
        'DISCRIMINATOR_ENCODER_N_ENCODER_BLOCKS': 1, 'USE_THP': False,
    }))
    return str(path)


def _write_logs(log_root: Path, name: str, fold: str, val_curve):
    """One trial's TensorBoard run, as fit_transehr2_deephit.py lays it out."""
    log_dir = log_root / name / fold / 'pretrained'
    log_dir.mkdir(parents=True)
    writer = SummaryWriter(str(log_dir))
    for epoch, value in enumerate(val_curve):
        for term in TERMS:
            writer.add_scalar(f'{term}/val', value, epoch)
            writer.add_scalar(f'{term}/train', value * 2, epoch)
    writer.close()


def _manifest(tmp_path: Path, trials, model_dir: Path, fold='fold0'):
    path = tmp_path / 'manifest.yaml'
    path.write_text(yaml.safe_dump({
        'spec_name': 'phase1', 'design': 'factorial', 'fold': fold,
        'model_dir': str(model_dir), 'arms': ['additive'],
        'grid': {}, 'trials': trials,
    }))
    return path


def _trial(name, config, arm='additive'):
    return {'name': name, 'config': config, 'arm': arm, 'is_centre': True,
            'hyperparameter': None, 'value': None, 'cell': {}, 'covers': [],
            'needs_finetune': False, 'is_extra': False,
            'select_on': 'pretrain', 'description': ''}


def _run(manifest, log_root, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(manifest),
         '--log_root', str(log_root), *args],
        capture_output=True, text=True
    )


@pytest.fixture
def sweep(tmp_path):
    """One trial, a descending validation curve, nothing in the model tree."""
    model_dir = tmp_path / 'models'
    log_root = tmp_path / 'log'
    config = _config(tmp_path / 'cell.yaml')
    manifest = _manifest(tmp_path, [_trial('phase1_additive_lr_0p002_hl_480',
                                           config)], model_dir)
    return {'tmp_path': tmp_path, 'model_dir': model_dir,
            'log_root': log_root, 'manifest': manifest,
            'name': 'phase1_additive_lr_0p002_hl_480'}


def _evaluation(sweep):
    path = (sweep['model_dir'] / sweep['name'] / 'fold0' / 'pretrained'
            / 'evaluation' / 'evaluation_pretrained.yaml')
    return yaml.safe_load(path.read_text()) if path.exists() else None


def test_the_rebuilt_file_is_the_one_the_ranking_reads(sweep):
    """Schema, not a parallel format: hp_tuning.results reads val_losses."""
    _write_logs(sweep['log_root'], sweep['name'], 'fold0', [9.0, 7.0, 5.0])
    result = _run(sweep['manifest'], sweep['log_root'])
    assert result.returncode == 0, result.stderr

    data = _evaluation(sweep)
    assert data['task'] == 'pretrain'
    assert data['fold'] == 'fold0'
    assert data['val_losses']['Optimization_Loss'] == pytest.approx(5.0)
    assert data['train_losses']['Optimization_Loss'] == pytest.approx(10.0)
    assert data['settings']['lr_half_life'] == 480


def test_the_selected_epoch_is_the_loops_not_the_smallest(sweep):
    """``is_improvement`` wants a relative margin, so epoch 2 here -- better by
    a hair -- is not the epoch the training loop would have kept."""
    _write_logs(sweep['log_root'], sweep['name'], 'fold0',
                [9.0, 5.0, 5.0 * (1 - 1e-6)])
    _run(sweep['manifest'], sweep['log_root'])

    data = _evaluation(sweep)
    assert data['rebuilt_from_tensorboard']['best_epoch'] == 1
    assert data['val_losses']['Optimization_Loss'] == pytest.approx(5.0)


def test_a_rebuilt_file_says_so(sweep):
    """A reader comparing these against a later run has to be able to tell."""
    _write_logs(sweep['log_root'], sweep['name'], 'fold0', [9.0, 5.0])
    _run(sweep['manifest'], sweep['log_root'])
    assert 'rebuilt_from_tensorboard' in _evaluation(sweep)


def test_a_real_runs_file_is_not_replaced(sweep):
    """The rerun writes its own. Clobbering it would be the same mistake twice."""
    _write_logs(sweep['log_root'], sweep['name'], 'fold0', [9.0, 5.0])
    path = (sweep['model_dir'] / sweep['name'] / 'fold0' / 'pretrained'
            / 'evaluation' / 'evaluation_pretrained.yaml')
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump({'task': 'pretrain', 'val_losses':
                                    {'Optimization_Loss': 1.0}}))

    _run(sweep['manifest'], sweep['log_root'])
    assert _evaluation(sweep)['val_losses']['Optimization_Loss'] == 1.0

    _run(sweep['manifest'], sweep['log_root'], '--overwrite')
    assert _evaluation(sweep)['val_losses']['Optimization_Loss'] == 5.0


def test_a_trial_with_no_logs_is_reported_and_fails_the_run(sweep):
    """Silence here would rank a sweep on whichever cells happened to survive."""
    result = _run(sweep['manifest'], sweep['log_root'])
    assert result.returncode == 1
    assert 'MISSING' in result.stderr
    assert _evaluation(sweep) is None
