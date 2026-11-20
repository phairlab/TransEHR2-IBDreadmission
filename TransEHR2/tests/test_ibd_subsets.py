"""Scoring within one IBD type (``--subset``), on synthetic cohorts.

The reference for every subset run is the plain run over dumps cut to the
subset's rows beforehand: the two must agree metric for metric, so that a
subset evaluation is exactly the whole-cohort routine on fewer patients, with
Lee's censoring curve fitted on the subset's own train rows.
"""

import math
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from TransEHR2.survival import TimeGrid

from .cr_cohort import simulate
from .test_compare_models import (EXPERIMENTS, SPLITS as VARIANT_SPLITS,
                                  fabricate as fabricate_evaluations)
from .test_cr_dcal_curves import prepare
from .test_csc import CAUSES as CSC_CAUSES, CUTS as CSC_CUTS, write_csc_fold
from .test_evaluate_finetuned_predictions import CUTS, TWO_CAUSES


TASK = 'competing_risks'
SCORED = ('test', 'test_recal_aj')
# Header fields that name the run rather than describe its patients.
HEADER = {'experiment', 'split', 'task', 'folds', 'time_grid_cuts_days',
          'modelled_causes', 'brier_integration_days', 'eval_horizons_days',
          'horizon_convention',
          'csc_curve', 'cr_dcal', 'subset', 'n_test', 'n_train'}


def dump_path(root, experiment, fold, split):
    return (root / experiment / fold / TASK
            / f'{TASK}_{split}_finetuned_output.csv')


def write_types(tmp_path, n_rows, seed, drop=None):
    """labels.csv and ibd_eligibility.csv for ``n_rows`` one-episode
    patients; ``drop`` leaves that row's patient out of the eligibility."""
    rng = np.random.default_rng(seed)
    patids = [f'P{i:05d}' for i in range(n_rows)]
    types = rng.choice(['CD', 'UC', 'IBDU'], size=n_rows, p=[0.5, 0.4, 0.1])
    labels = tmp_path / 'labels.csv'
    pd.DataFrame({'PATID': patids, 'EVENT_TYPE': 'x'}).to_csv(labels,
                                                              index=False)
    keep = np.ones(n_rows, dtype=bool)
    if drop is not None:
        keep[drop] = False
    eligibility = tmp_path / 'ibd_eligibility.csv'
    pd.DataFrame({'PATID': np.array(patids)[keep],
                  'ELIGIBILITY_DATE': '2010-01-01',
                  'IBD_TYPE': types[keep]}).to_csv(eligibility, index=False)
    return labels, eligibility, types


def write_grid_dumps(root, experiment, grid, seed):
    """Two folds of train, val and test dumps, every row its own patient."""
    from dump_transehr2_deephit_predictions import save_predictions_csv

    start = 0
    for fold in ('fold1', 'fold2'):
        for split in ('train', 'val', 'test'):
            seed += 1
            data = simulate(grid, n=1500, seed=seed)
            n = data['cif'].shape[0]
            save_predictions_csv(np.arange(start, start + n),
                                 data['time_to_event'], data['event_type'],
                                 data['cif'], data['pmf'], data['p_survive'],
                                 grid, str(dump_path(root, experiment, fold,
                                                     split)))
            start += n
    return start


def score(root, experiment, monkeypatch, subset=None, labels=None,
          eligibility=None):
    import evaluate_finetuned_predictions
    argv = ['evaluate_finetuned_predictions.py', experiment, '--model_dir',
            str(root), '--experiment_config', str(root / f'{experiment}.yaml')]
    if subset:
        argv += ['--subset', subset, '--labels', str(labels),
                 '--eligibility', str(eligibility)]
    monkeypatch.setattr(sys, 'argv', argv)
    evaluate_finetuned_predictions.main()


def cut_copy(root, experiment, reference, types, subset):
    """``reference``: ``experiment`` with every dump cut to ``subset``."""
    shutil.copytree(root / experiment, root / reference)
    shutil.copy(root / f'{experiment}.yaml', root / f'{reference}.yaml')
    for path in (root / reference).glob(f'fold*/{TASK}/*_finetuned_output.csv'):
        frame = pd.read_csv(path, float_precision='round_trip')
        frame[types[frame['idx'].to_numpy()] == subset].to_csv(path,
                                                               index=False)


def read(root, experiment, name):
    with open(root / experiment / name) as f_in:
        return yaml.safe_load(f_in)


def assert_same_metrics(subset_eval, reference_eval):
    metrics = set(reference_eval) - HEADER
    assert metrics == set(subset_eval) - HEADER
    for key in sorted(metrics):
        np.testing.assert_array_equal(subset_eval[key], reference_eval[key],
                                      err_msg=key)


@pytest.fixture
def grid_run(tmp_path, monkeypatch):
    import recalibrate_predictions

    grid = TimeGrid(CUTS, TWO_CAUSES)
    n_rows = write_grid_dumps(tmp_path, 'exp', grid, seed=40)
    labels, eligibility, types = write_types(tmp_path, n_rows, seed=5)
    prepare(tmp_path, monkeypatch, 'exp', {
        'TIME_GRID_CUTS_DAYS': CUTS, 'MODELLED_CAUSES': TWO_CAUSES,
        'BRIER_INTEGRATION_DAYS': 365, 'CR_DCAL_CHI2_BINS': 5,
        'CR_DCAL_N_RHO': 250, 'EVAL_HORIZONS_DAYS': [30, 365]},
        (recalibrate_predictions,))
    return tmp_path, labels, eligibility, types


def test_a_grid_subset_is_the_plain_run_on_its_rows(grid_run, monkeypatch):
    root, labels, eligibility, types = grid_run
    score(root, 'exp', monkeypatch, 'CD', labels, eligibility)
    # Only the two test splits, beside (not over) the whole-cohort files.
    written = {p.name for p in (root / 'exp').glob('*_evaluation*.yaml')}
    assert written == {'test_evaluation_CD.yaml',
                       'test_recal_aj_evaluation_CD.yaml'}

    cut_copy(root, 'exp', 'ref', types, 'CD')
    score(root, 'ref', monkeypatch)
    for split in SCORED:
        subset_eval = read(root, 'exp', f'{split}_evaluation_CD.yaml')
        assert_same_metrics(subset_eval,
                            read(root, 'ref', f'{split}_evaluation.yaml'))
        assert subset_eval['subset'] == 'CD'
        n_test, n_train = [], []
        for fold in ('fold1', 'fold2'):
            test = pd.read_csv(dump_path(root, 'exp', fold, 'test'))
            n_test.append(int(np.sum(types[test['idx']] == 'CD')))
            n_train.append(len(pd.read_csv(dump_path(root, 'exp', fold,
                                                     'train'))))
        assert subset_eval['n_test'] == n_test
        assert subset_eval['n_train'] == n_train
        assert subset_eval['Episodes'] == n_test


def test_the_subset_curves_match_the_subset_evaluation(grid_run,
                                                       monkeypatch):
    import cr_dcal_curves

    root, labels, eligibility, _ = grid_run
    score(root, 'exp', monkeypatch, 'UC', labels, eligibility)
    out = root / 'tables' / 'UC' / 'curves.csv'
    # The run stops unless every curve's D_k is the subset evaluation's.
    assert cr_dcal_curves.main([
        '--out', str(out), '--experiments', 'exp', '--model_dir', str(root),
        '--config_dir', str(root), '--subset', 'UC', '--labels', str(labels),
        '--eligibility', str(eligibility)]) == 0
    evaluation = read(root, 'exp', 'test_evaluation_UC.yaml')
    frame = pd.read_csv(out)
    one = frame[(frame['variant'] == 'raw') & (frame['fold'] == 'fold1')
                & (frame['cause'] == 'death')]
    d_k = np.sqrt(np.mean((one['b'] - one['rho']) ** 2))
    assert d_k == pytest.approx(evaluation['Death_CR_DCal'][0], abs=1e-12)


def test_a_csc_subset_reads_the_exact_curve_of_its_rows(tmp_path,
                                                        monkeypatch):
    import dump_baseline_csc_predictions
    import recalibrate_predictions

    grid = TimeGrid(CSC_CUTS, CSC_CAUSES)
    write_csc_fold(tmp_path / 'csc' / 'fold1' / TASK, grid, seed=30)
    labels, eligibility, types = write_types(tmp_path, 1100, seed=6)
    prepare(tmp_path, monkeypatch, 'csc', {
        'TIME_GRID_CUTS_DAYS': CSC_CUTS, 'MODELLED_CAUSES': CSC_CAUSES,
        'BRIER_INTEGRATION_DAYS': 365, 'EVAL_HORIZONS_DAYS': [30, 90, 365]},
        (dump_baseline_csc_predictions, recalibrate_predictions))
    score(tmp_path, 'csc', monkeypatch, 'UC', labels, eligibility)
    # The linear predictors and offsets stay whole: the curve is read by
    # idx, so only the dumps need cutting.
    cut_copy(tmp_path, 'csc', 'ref', types, 'UC')
    score(tmp_path, 'ref', monkeypatch)
    for split in SCORED:
        subset_eval = read(tmp_path, 'csc', f'{split}_evaluation_UC.yaml')
        assert subset_eval['csc_curve'] is True
        assert_same_metrics(subset_eval,
                            read(tmp_path, 'ref', f'{split}_evaluation.yaml'))


def test_a_patient_with_no_type_is_refused(tmp_path, monkeypatch):
    grid = TimeGrid(CUTS, TWO_CAUSES)
    n_rows = write_grid_dumps(tmp_path, 'exp', grid, seed=50)
    labels, eligibility, _ = write_types(tmp_path, n_rows, seed=7,
                                         drop=n_rows - 1)
    (tmp_path / 'exp.yaml').write_text(yaml.dump({
        'TIME_GRID_CUTS_DAYS': CUTS, 'MODELLED_CAUSES': TWO_CAUSES,
        'BRIER_INTEGRATION_DAYS': 365, 'EVAL_HORIZONS_DAYS': [30]}))
    with pytest.raises(ValueError, match='no IBD_TYPE'):
        score(tmp_path, 'exp', monkeypatch, 'CD', labels, eligibility)


def test_the_held_out_set_is_not_split(tmp_path, monkeypatch):
    grid = TimeGrid(CUTS, TWO_CAUSES)
    n_rows = write_grid_dumps(tmp_path, 'exp', grid, seed=60)
    (tmp_path / 'exp' / 'fold2').rename(tmp_path / 'exp' / 'full')
    labels, eligibility, _ = write_types(tmp_path, n_rows, seed=8)
    (tmp_path / 'exp.yaml').write_text(yaml.dump({
        'TIME_GRID_CUTS_DAYS': CUTS, 'MODELLED_CAUSES': TWO_CAUSES,
        'BRIER_INTEGRATION_DAYS': 365, 'EVAL_HORIZONS_DAYS': [30]}))
    with pytest.raises(SystemExit, match='held-out'):
        score(tmp_path, 'exp', monkeypatch, 'CD', labels, eligibility)


N_TEST = [300, 310, 290, 305, 295]
N_TRAIN = [1200, 1210, 1190, 1205, 1195]


def fabricate_subset(model_dir, n_test=N_TEST, odd=None):
    """compare_models' fabricated evaluations, as a CD run's files;
    ``odd`` gives (model, variant) its own n_test."""
    subset = {'subset': 'CD', 'n_test': list(n_test),
              'n_train': list(N_TRAIN)}
    override = {(m, v): dict(subset) for m in EXPERIMENTS
                for v in VARIANT_SPLITS}
    if odd:
        override[odd]['n_test'] = [n + 1 for n in n_test]
    fabricate_evaluations(model_dir, seed=3, override=override)
    for path in model_dir.glob('*/*_evaluation.yaml'):
        path.rename(path.with_name(path.stem + '_CD.yaml'))


def compare(tmp_path, monkeypatch):
    from compare_models import main
    out_dir = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', [
        'compare_models.py', '--model_dir', str(tmp_path / 'models'),
        '--out_dir', str(out_dir), '--subset', 'CD'])
    main()
    return out_dir


def test_a_subset_comparison_takes_its_ratio_from_the_fold_sizes(
        tmp_path, monkeypatch):
    model_dir = tmp_path / 'models'
    fabricate_subset(model_dir)
    table = pd.read_csv(compare(tmp_path, monkeypatch)
                        / 'readmission_comparisons.csv')
    row = table[(table['metric'] == 'Readmission_IBS_to_1y')
                & (table['comparison'] == 'TransEHR2 raw vs CSC raw')]
    with open(model_dir / 'part1_transehr2' / 'test_evaluation_CD.yaml') as f:
        a = np.array(yaml.safe_load(f)['Readmission_IBS_to_1y'])
    ratio = np.mean(np.array(N_TEST) / np.array(N_TRAIN))
    assert row['se_a'].item() == pytest.approx(
        math.sqrt((1 / 5 + ratio) * a.var(ddof=1)))


def test_a_subset_scored_on_other_patients_is_refused(tmp_path,
                                                      monkeypatch):
    fabricate_subset(tmp_path / 'models', odd=('CSC', 'recal'))
    with pytest.raises(SystemExit, match='n_test'):
        compare(tmp_path, monkeypatch)
