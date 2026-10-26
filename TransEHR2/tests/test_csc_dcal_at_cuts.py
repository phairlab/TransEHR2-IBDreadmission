"""csc_dcal_at_cuts.py reproduces the DeepHit models' D-calibration path."""

import sys

import numpy as np
import pandas as pd
import yaml

from TransEHR2.survival import TimeGrid

from .cr_cohort import simulate
from .test_evaluate_finetuned_predictions import (CUTS, TWO_CAUSES,
                                                   write_split)


def test_at_cut_rows_match_the_evaluation_of_a_grid_model(tmp_path,
                                                          monkeypatch):
    """A model with no CSC parameters is scored on the grid by the
    evaluation itself, so there the two reads must agree exactly."""
    import csc_dcal_at_cuts
    import evaluate_finetuned_predictions
    import recalibrate_predictions

    grid = TimeGrid(CUTS, TWO_CAUSES)
    config = tmp_path / 'experiment.yaml'
    config.write_text(yaml.dump({
        'TIME_GRID_CUTS_DAYS': CUTS, 'MODELLED_CAUSES': TWO_CAUSES,
        'BRIER_INTEGRATION_DAYS': 365, 'CR_DCAL_CHI2_BINS': 5,
        'EVAL_HORIZONS_DAYS': [30, 365]}))
    seed = 40
    for fold in ('fold1', 'fold2'):
        for split in ('train', 'val', 'test'):
            seed += 1
            write_split(str(tmp_path / 'exp' / fold / 'competing_risks'
                            / f'competing_risks_{split}_finetuned_output.csv'),
                        simulate(grid, n=1500, seed=seed), grid)
    common = ['--model_dir', str(tmp_path), '--experiment_config',
              str(config)]
    for module, name in ((recalibrate_predictions, 'recalibrate'),
                         (evaluate_finetuned_predictions, 'evaluate')):
        monkeypatch.setattr(sys, 'argv', [name, 'exp', *common])
        module.main()

    out = tmp_path / 'tables' / 'csc_dcal_at_cuts.csv'
    assert csc_dcal_at_cuts.main(['--out', str(out), '--experiment_name',
                                  'exp', *common]) == 0
    frame = pd.read_csv(out)
    assert len(frame) == 2 * 2 * 2 * 2      # folds, causes, variants, reads
    key = ['fold', 'cause', 'variant']
    cuts = frame[frame['read'] == 'cuts'].set_index(key).sort_index()
    exact = frame[frame['read'] == 'exact'].set_index(key).sort_index()
    for column in ('CR_DCal', 'chi2', 'p'):
        np.testing.assert_allclose(cuts[column], exact[column], rtol=1e-12)
    # Recalibration changed something, so the recal rows test it.
    raw = cuts.xs('raw', level='variant')['chi2']
    recal = cuts.xs('recal', level='variant')['chi2']
    assert not np.allclose(raw, recal)


def test_nothing_is_written_under_the_model_dir(tmp_path):
    import csc_dcal_at_cuts

    grid = TimeGrid(CUTS, TWO_CAUSES)
    folder = tmp_path / 'exp' / 'fold1' / 'competing_risks'
    for i, split in enumerate(('val', 'test')):
        write_split(str(folder / f'competing_risks_{split}_finetuned_output.csv'),
                    simulate(grid, n=800, seed=60 + i), grid)
    before = sorted((tmp_path / 'exp').rglob('*'))
    config = tmp_path / 'experiment.yaml'
    config.write_text(yaml.dump({'TIME_GRID_CUTS_DAYS': CUTS,
                                 'MODELLED_CAUSES': TWO_CAUSES,
                                 'CR_DCAL_CHI2_BINS': 5}))
    out = tmp_path / 'out' / 'rows.csv'
    assert csc_dcal_at_cuts.main([
        '--out', str(out), '--experiment_name', 'exp', '--model_dir',
        str(tmp_path), '--experiment_config', str(config)]) == 0
    assert sorted((tmp_path / 'exp').rglob('*')) == before
    # No evaluation YAMLs, so no exact rows.
    assert set(pd.read_csv(out)['read']) == {'cuts'}
