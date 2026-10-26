"""cr_dcal_curves.py writes the curves the evaluation's D_k was computed on."""

import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from TransEHR2.survival import TimeGrid

from .cr_cohort import simulate
from .test_csc import CAUSES as CSC_CAUSES, CUTS as CSC_CUTS, write_csc_fold
from .test_evaluate_finetuned_predictions import (CUTS, TWO_CAUSES,
                                                   write_split)


def prepare(tmp_path, monkeypatch, name, config, modules):
    """Write ``config`` as ``<name>.yaml`` and run ``modules`` over ``name``."""
    (tmp_path / f'{name}.yaml').write_text(yaml.dump(config))
    for module in modules:
        monkeypatch.setattr(sys, 'argv', [
            'script.py', name, '--model_dir', str(tmp_path),
            '--experiment_config', str(tmp_path / f'{name}.yaml')])
        module.main()


def curves(tmp_path, *experiments):
    import cr_dcal_curves

    out = tmp_path / 'tables' / 'curves.csv'
    assert cr_dcal_curves.main([
        '--out', str(out), '--experiments', *experiments,
        '--model_dir', str(tmp_path), '--config_dir', str(tmp_path)]) == 0
    return pd.read_csv(out)


@pytest.fixture
def grid_experiment(tmp_path, monkeypatch):
    import evaluate_finetuned_predictions
    import recalibrate_predictions

    grid = TimeGrid(CUTS, TWO_CAUSES)
    seed = 70
    for fold in ('fold1', 'fold2'):
        for split in ('train', 'val', 'test'):
            seed += 1
            write_split(str(tmp_path / 'exp' / fold / 'competing_risks'
                            / f'competing_risks_{split}_finetuned_output.csv'),
                        simulate(grid, n=1200, seed=seed), grid)
    prepare(tmp_path, monkeypatch, 'exp', {
        'TIME_GRID_CUTS_DAYS': CUTS, 'MODELLED_CAUSES': TWO_CAUSES,
        'BRIER_INTEGRATION_DAYS': 365, 'CR_DCAL_N_RHO': 250,
        'EVAL_HORIZONS_DAYS': [30, 365]},
        (recalibrate_predictions, evaluate_finetuned_predictions))
    return tmp_path


def test_grid_model_curves_reproduce_its_d_k(grid_experiment):
    frame = curves(grid_experiment, 'exp')
    # experiment x variants x folds x causes x rhos
    assert len(frame) == 1 * 2 * 2 * 2 * 250
    with open(grid_experiment / 'exp' / 'test_evaluation.yaml') as f_in:
        evaluation = yaml.safe_load(f_in)
    one = frame[(frame['variant'] == 'raw') & (frame['fold'] == 'fold2')
                & (frame['cause'] == 'death')]
    d_k = np.sqrt(np.mean((one['b'] - one['rho']) ** 2))
    assert d_k == pytest.approx(evaluation['Death_CR_DCal'][1], abs=1e-12)


def test_a_curve_that_disagrees_with_the_evaluation_stops_the_run(
        grid_experiment):
    import cr_dcal_curves

    path = grid_experiment / 'exp' / 'test_evaluation.yaml'
    evaluation = yaml.safe_load(path.read_text())
    evaluation['Readmission_CR_DCal'][0] += 0.01
    path.write_text(yaml.dump(evaluation))
    out = grid_experiment / 'tables' / 'curves.csv'
    with pytest.raises(ValueError, match='not the curve that was scored'):
        cr_dcal_curves.main([
            '--out', str(out), '--experiments', 'exp', '--model_dir',
            str(grid_experiment), '--config_dir', str(grid_experiment)])
    assert not out.exists()


def test_the_csc_is_read_on_its_exact_curve(tmp_path, monkeypatch):
    import dump_baseline_csc_predictions
    import evaluate_finetuned_predictions
    import recalibrate_predictions

    grid = TimeGrid(CSC_CUTS, CSC_CAUSES)
    write_csc_fold(tmp_path / 'csc' / 'fold1' / 'competing_risks', grid,
                   seed=30)
    prepare(tmp_path, monkeypatch, 'csc', {
        'TIME_GRID_CUTS_DAYS': CSC_CUTS, 'MODELLED_CAUSES': CSC_CAUSES,
        'BRIER_INTEGRATION_DAYS': 365, 'CR_DCAL_N_RHO': 200,
        'EVAL_HORIZONS_DAYS': [30, 90, 365]},
        (dump_baseline_csc_predictions, recalibrate_predictions,
         evaluate_finetuned_predictions))
    # The run itself checks every curve's D_k against the evaluation, which
    # read the exact curve; a grid read would not match.
    frame = curves(tmp_path, 'csc')
    assert set(frame['variant']) == {'raw', 'recal'}
    assert set(frame['cause']) == set(CSC_CAUSES)
