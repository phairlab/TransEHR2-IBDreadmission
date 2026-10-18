"""Pretraining has to leave behind something a sweep can rank.

``pretrain_model`` returns the best epoch's losses and
``fit_transehr2_deephit`` used to throw them away: they reached the console
and TensorBoard, and the weights reached disk, but no file carried the
number a learning-rate sweep is decided on. These probes pin the file's
path and the one key the ranking reads, because both are a contract with
``hp_tuning`` rather than an internal detail.
"""

import numpy as np
import pytest
import torch
import yaml

from fit_transehr2_deephit import write_pretraining_evaluation


LOSSES = {'Optimization_Loss': 4.79, 'Generator_Loss': 3.1,
          'Discriminator_Loss': 1.69}


def write(tmp_path, **overrides):
    """Write one evaluation file and return (path, parsed)."""
    kwargs = {
        'pretrained_dir': str(tmp_path / 'trial' / 'fold0' / 'pretrained'),
        'experiment_name': 'trial',
        'fold_name': 'fold0',
        'settings': {'learning_rate': 0.002, 'lr_half_life': 160},
        'train_losses': dict(LOSSES),
        'val_losses': dict(LOSSES),
    }
    kwargs.update(overrides)
    path = write_pretraining_evaluation(**kwargs)
    with open(path) as f_in:
        return path, yaml.safe_load(f_in)


def test_the_file_lands_where_the_ranking_looks_for_it(tmp_path):
    """``{...}/{fold}/pretrained/evaluation/evaluation_pretrained.yaml``.

    The retired sweep script wrote a path with no fold in it. A trial is
    one (experiment, fold) pair, and the reader addresses it that way, so
    the fold segment is part of the contract rather than decoration.
    """
    path, _ = write(tmp_path)
    assert path.endswith(
        'trial/fold0/pretrained/evaluation/evaluation_pretrained.yaml')


def test_the_selection_metric_is_readable_as_a_float(tmp_path):
    """``val_losses['Optimization_Loss']`` is what ranks a schedule."""
    _, data = write(tmp_path)
    value = data['val_losses']['Optimization_Loss']
    assert isinstance(value, float) and value == pytest.approx(4.79)


def test_validation_and_training_losses_are_kept_apart(tmp_path):
    """Selection reads validation; training is recorded for diagnosis."""
    _, data = write(tmp_path,
                    train_losses={'Optimization_Loss': 1.0},
                    val_losses={'Optimization_Loss': 2.0})
    assert data['train_losses']['Optimization_Loss'] == 1.0
    assert data['val_losses']['Optimization_Loss'] == 2.0


def test_tensor_and_numpy_losses_survive_the_dump(tmp_path):
    """The losses arrive as whatever the training loop accumulated.

    ``yaml.safe_dump`` refuses a tensor or a numpy scalar outright, so a
    run that had trained for hours would fail at the last step. The
    writer converts first.
    """
    _, data = write(tmp_path, val_losses={
        'Optimization_Loss': torch.tensor(4.79),
        'Generator_Loss': np.float32(3.1),
    })
    assert data['val_losses']['Optimization_Loss'] == pytest.approx(4.79)
    assert data['val_losses']['Generator_Loss'] == pytest.approx(3.1)


def test_a_flat_schedule_records_a_null_half_life(tmp_path):
    """``lr_half_life: None`` is the no-decay arm, not a missing value.

    It is one cell of the sweep, so the record has to distinguish it from
    a schedule nobody wrote down.
    """
    _, data = write(tmp_path, settings={'learning_rate': 0.002,
                                        'lr_half_life': None})
    assert 'lr_half_life' in data['settings']
    assert data['settings']['lr_half_life'] is None


def test_the_settings_that_produced_the_losses_are_recorded(tmp_path):
    """The losses alone do not say which schedule they belong to."""
    _, data = write(tmp_path)
    assert data['settings']['learning_rate'] == 0.002
    assert data['settings']['lr_half_life'] == 160
    assert data['experiment'] == 'trial'
    assert data['fold'] == 'fold0'
    assert data['task'] == 'pretrain'


def test_pretrain_only_is_an_option_on_the_entry_point(capsys):
    """A phase ranked on the pretraining loss must be able to stop there.

    Without it the 24-cell schedule sweep also fits 24 heads that no
    selection ever reads.
    """
    import sys

    import fit_transehr2_deephit

    argv = sys.argv
    sys.argv = ['fit_transehr2_deephit.py', '--help']
    try:
        with pytest.raises(SystemExit):
            fit_transehr2_deephit.main()
    finally:
        sys.argv = argv

    assert '--pretrain_only' in capsys.readouterr().out
