"""The prediction dump and the scoring script, end to end on a synthetic cohort."""

import os
import sys

import numpy as np
import pytest
import yaml

from TransEHR2.survival import TimeGrid

from .cr_cohort import simulate


CUTS = [30.0, 90.0, 365.0]
TWO_CAUSES = ["readmission", "death"]


def write_split(path, data, grid):
    from dump_finetuned_predictions import save_predictions_csv

    n = data['cif'].shape[0]
    save_predictions_csv(np.arange(n), data['time_to_event'],
                         data['event_type'], data['cif'], data['pmf'],
                         data['p_survive'], grid, path)


def test_the_dump_reads_back_bit_for_bit(tmp_path):
    from evaluate_finetuned_predictions import read_predictions

    grid = TimeGrid(CUTS, TWO_CAUSES)
    data = simulate(grid, n=200, seed=2)
    path = str(tmp_path / 'out.csv')
    write_split(path, data, grid)
    back = read_predictions(path, grid)
    for key in ('cif', 'pmf', 'p_survive', 'time_to_event', 'event_type'):
        assert np.array_equal(back[key], data[key]), key


def test_a_dump_without_the_pmf_is_refused(tmp_path):
    import pandas as pd
    from evaluate_finetuned_predictions import read_predictions

    grid = TimeGrid(CUTS, TWO_CAUSES)
    path = str(tmp_path / 'out.csv')
    write_split(path, simulate(grid, n=20, seed=3), grid)
    frame = pd.read_csv(path)
    frame.drop(columns=[c for c in frame.columns if c.startswith('pmf_')]
               ).to_csv(path, index=False)
    with pytest.raises(ValueError, match='re-dump'):
        read_predictions(path, grid)


def test_recalibration_dumps_beside_and_scoring_reads_it_as_a_split(
        tmp_path, monkeypatch):
    import evaluate_finetuned_predictions
    import recalibrate_predictions
    from evaluate_finetuned_predictions import read_predictions
    from recalibrate_predictions import recalibrate

    grid = TimeGrid(CUTS, TWO_CAUSES)
    config = tmp_path / 'experiment.yaml'
    config.write_text(yaml.dump({
        'TIME_GRID_CUTS_DAYS': CUTS, 'MODELLED_CAUSES': TWO_CAUSES,
        'BRIER_INTEGRATION_DAYS': 365, 'CR_DCAL_CHI2_BINS': 5}))

    seed = 10
    for fold in ('fold1', 'fold2'):
        for split in ('train', 'val', 'test'):
            seed += 1
            write_split(str(tmp_path / 'exp' / fold / 'competing_risks'
                            / f'competing_risks_{split}_finetuned_output.csv'),
                        simulate(grid, n=1500, seed=seed), grid)

    exp = tmp_path / 'exp'
    folder = exp / 'fold1' / 'competing_risks'
    dump = folder / 'competing_risks_test_finetuned_output.csv'
    recal_dump = folder / 'competing_risks_test_recal_aj_finetuned_output.csv'

    def run(module, name):
        monkeypatch.setattr(sys, 'argv', [
            name, 'exp', '--model_dir', str(tmp_path),
            '--experiment_config', str(config)])
        module.main()

    # Recalibration is its own step: scoring alone writes no recalibrated
    # anything.
    run(evaluate_finetuned_predictions, 'evaluate_finetuned_predictions.py')
    assert not (exp / 'test_recal_aj_evaluation.yaml').exists()
    assert not recal_dump.exists()
    plain = (exp / 'test_evaluation.yaml').read_text()
    original = dump.read_bytes()

    run(recalibrate_predictions, 'recalibrate_predictions.py')
    # The corrected dump goes beside the original, which is untouched.
    assert dump.read_bytes() == original
    expected = recalibrate(
        read_predictions(str(folder / 'competing_risks_val_finetuned_output.csv'),
                         grid),
        read_predictions(str(dump), grid), grid)
    written = read_predictions(str(recal_dump), grid)
    for key in ('idx', 'cif', 'pmf', 'p_survive', 'time_to_event',
                'event_type'):
        assert np.array_equal(written[key], expected[key]), key

    with open(exp / 'test_recal_aj_record.yaml') as f_in:
        record = yaml.safe_load(f_in)
    assert record['calibration_split'] == 'val'
    assert list(record['folds']) == ['fold1', 'fold2']
    fold1 = record['folds']['fold1']
    assert fold1['repaired_fraction'] == expected['repaired_fraction']
    assert np.array_equal(
        [fold1['offsets'][c] for c in TWO_CAUSES], expected['offsets'])

    run(evaluate_finetuned_predictions, 'evaluate_finetuned_predictions.py')
    assert (exp / 'test_evaluation.yaml').read_text() == plain

    results = {}
    for name in ('train', 'val', 'test', 'test_recal_aj'):
        with open(exp / f'{name}_evaluation.yaml') as f_in:
            results[name] = yaml.safe_load(f_in)
        assert results[name]['folds'] == ['fold1', 'fold2']
        assert results[name]['cr_dcal']['chi2_bins'] == 5
        for key in ('Readmission_CR_DCal', 'Death_CR_DCal_Chi2_P', 'CR_DCal',
                    'CR_DCal_Chi2_Pass', 'Readmission_Integrated_Brier'):
            assert len(results[name][key]) == 2
            assert all(np.isfinite(results[name][key]))
    # The same routines, so the same keys: the two files compare key for key.
    assert set(results['test']) == set(results['test_recal_aj'])
