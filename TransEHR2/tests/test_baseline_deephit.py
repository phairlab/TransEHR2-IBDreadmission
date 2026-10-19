"""The tabular DeepHit baseline on a synthetic fold: encode, fit, dump.

The cohort is fabricated here -- ``labels.csv``, ``extracted_rows.npy``, the
fold arrays and ``baseline_features.csv`` in ``lace.py``'s columns -- so the
row mapping is checked as well as the model: a partition's rows are
``extracted_rows[positions]``, and the dump keys them by ``labels.csv`` row.
"""

import numpy as np
import pandas as pd
import pytest
import yaml

from pathlib import Path

from TransEHR2.cli import TASK
from TransEHR2.tabular import (
    ENCODING_FILE, MISSING_LEVEL, WEIGHTS_FILE, BaselineEncoding,
    config_grid
)

from dump_baseline_deephit_predictions import main as dump_main
from evaluate_finetuned_predictions import read_predictions
from fit_baseline_deephit import main as fit_main


CONFIG = (Path(__file__).resolve().parents[1] / 'configs' / 'experiments'
          / 'part1_tabdh.yaml')
FOLD = 'fold1'
SPLITS = ('train', 'val', 'test')
N_LABELS, N_DRAWN = 900, 600
LINEAR = ['AGE', 'L', 'A', 'E', 'CCI', 'QUINTMAT', 'QUINTSOC']
FILLED = ['AGE', 'QUINTMAT', 'QUINTSOC']
CATEGORICAL = ['SEX', 'Rural', 'INDEX_TYPE']   # ZONE_NAME is in the file, unused


def blank(rng, values, fraction):
    """``values`` as strings, with ``fraction`` of them emptied."""
    out = np.array([str(v) for v in values], dtype=object)
    out[rng.random(len(out)) < fraction] = ''
    return out


def write_cohort(data_dir: Path, rng) -> None:
    times_days = rng.uniform(1.0, 6000.0, N_LABELS)
    labels = pd.DataFrame({
        'PATID': [f'P{i:04d}' for i in range(N_LABELS)],
        'STAY_INDEX': 0,
        'TIME_TO_EVENT': np.round(times_days * 1440).astype(int),
        'EVENT_TYPE': rng.choice(4, N_LABELS, p=[0.4, 0.35, 0.15, 0.1]),
        'POST_IBD_ONSET': 1,
    })
    labels.to_csv(data_dir / 'labels.csv', index=False)

    drawn = np.sort(rng.choice(N_LABELS, N_DRAWN, replace=False))
    np.save(data_dir / 'extracted_rows.npy', drawn)
    positions = rng.permutation(N_DRAWN)
    for fold in ('fold0', FOLD):
        (data_dir / fold).mkdir()
        for split, part in zip(SPLITS, np.split(positions, [360, 480])):
            np.save(data_dir / fold / f'{fold}_{split}_rows.npy',
                    np.sort(part))

    n = N_DRAWN
    quint = rng.integers(1, 6, (2, n))
    features = pd.DataFrame({
        'idx': drawn,
        'PATID': labels['PATID'].to_numpy()[drawn],
        'STAY_INDEX': 0,
        'INDEX_TYPE': rng.choice(['DAD', 'AMB'], n),
        'L': rng.choice([0, 1, 2, 3, 4, 5, 7], n),
        'A': rng.choice([0, 3], n),
        'E': rng.integers(0, 5, n),
        'CCI': rng.poisson(1.0, n),
        'AGE': blank(rng, rng.integers(18, 90, n), 0.05),
        'SEX': blank(rng, rng.choice(['F', 'M'], n), 0.05),
        'Rural': blank(rng, rng.choice(['0', '1'], n), 0.05),
        'ZONE_NAME': blank(rng, rng.choice(['North', 'South', 'Calgary'], n),
                           0.05),
        'QUINTMAT': blank(rng, quint[0], 0.15),
        'QUINTSOC': blank(rng, quint[1], 0.15),
    })
    features.to_csv(data_dir / 'baseline_features.csv', index=False,
                    na_rep='')


@pytest.fixture(scope='module')
def fitted(tmp_path_factory):
    """One synthetic cohort, fitted on FOLD and dumped once."""
    root = tmp_path_factory.mktemp('tabdh')
    data_dir = root / 'data'
    data_dir.mkdir()
    write_cohort(data_dir, np.random.default_rng(7))

    dataset_config = root / 'dataset.yaml'
    dataset_config.write_text(yaml.safe_dump({'DATA_DIR': str(data_dir)}))
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg.update({'MODEL_DIR': str(root / 'models'), 'FINETUNE_TOTAL_EPOCH': 3,
                'BATCH_SIZE': 64, 'LR_CHECK_RATES': [0.0002, 0.002]})
    experiment_config = root / 'experiment.yaml'
    experiment_config.write_text(yaml.safe_dump(cfg))

    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(root)                      # ./log and ./checkpoints
        fit_main([str(dataset_config), str(experiment_config),
                  '--folds', FOLD, '--device', 'cpu'])
        dump_args = [str(dataset_config), str(experiment_config),
                     cfg['EXPERIMENT_NAME'], '--model_dir', cfg['MODEL_DIR'],
                     '--folds', FOLD, '--device', 'cpu']
        dump_main(dump_args)

    return {'root': root, 'data_dir': data_dir, 'cfg': cfg,
            'dataset_config': dataset_config,
            'experiment_config': experiment_config, 'dump_args': dump_args,
            'folder': Path(cfg['MODEL_DIR']) / cfg['EXPERIMENT_NAME'] / FOLD
            / TASK}


def dump_path(folder: Path, split: str) -> Path:
    return folder / f'{TASK}_{split}_finetuned_output.csv'


def test_fit_leaves_weights_encoding_and_evaluation(fitted):
    folder = fitted['folder']
    assert (folder / WEIGHTS_FILE).exists()
    assert (folder / ENCODING_FILE).exists()
    evaluation = yaml.safe_load(
        (folder / 'evaluation' / f'evaluation_{TASK}.yaml').read_text())
    assert evaluation['selection_metric'] == 'brier'
    assert 'Readmission_Integrated_Brier' in evaluation['validation_scores']


def test_dumps_load_and_are_distributions(fitted):
    grid = config_grid(fitted['cfg'])
    data_dir = fitted['data_dir']
    labels = pd.read_csv(data_dir / 'labels.csv')
    drawn = np.load(data_dir / 'extracted_rows.npy')

    for split in SPLITS:
        data = read_predictions(str(dump_path(fitted['folder'], split)), grid)
        rows = drawn[np.load(data_dir / FOLD / f'{FOLD}_{split}_rows.npy')]
        np.testing.assert_array_equal(data['idx'], np.sort(rows))
        np.testing.assert_array_equal(
            data['time_to_event'], labels['TIME_TO_EVENT'][data['idx']])
        np.testing.assert_array_equal(
            data['event_type'], labels['EVENT_TYPE'][data['idx']])

        cif, pmf = data['cif'], data['pmf']
        assert cif.shape == (len(rows), grid.n_causes, grid.n_bins)
        assert np.all(pmf >= 0)
        assert np.all(np.diff(cif, axis=2) >= -1e-12)
        np.testing.assert_allclose(cif, np.cumsum(pmf, axis=2), atol=1e-12)
        assert np.all(cif.sum(axis=1) <= 1 + 1e-12)
        np.testing.assert_allclose(
            data['p_survive'], 1 - cif[:, :, -1].sum(axis=1), atol=1e-12)
        assert np.all((data['p_survive'] >= 0) & (data['p_survive'] <= 1))


def test_redump_is_identical(fitted):
    before = {s: dump_path(fitted['folder'], s).read_bytes() for s in SPLITS}
    dump_main(fitted['dump_args'])
    for split in SPLITS:
        assert dump_path(fitted['folder'], split).read_bytes() == \
            before[split]


def test_saved_encoding_is_the_train_partition_fit(fitted):
    data_dir = fitted['data_dir']
    features = pd.read_csv(data_dir / 'baseline_features.csv', dtype=str,
                           keep_default_na=False)
    features.index = features['idx'].astype(int)
    drawn = np.load(data_dir / 'extracted_rows.npy')
    train = features.loc[drawn[np.load(
        data_dir / FOLD / f'{FOLD}_train_rows.npy')]]

    saved = BaselineEncoding.load(str(fitted['folder'] / ENCODING_FILE))
    assert saved == BaselineEncoding.fit(train, LINEAR, FILLED,
                                         CATEGORICAL)
    assert saved.indicated == FILLED
    for q in FILLED:
        observed = pd.to_numeric(train[q][train[q] != ''])
        assert saved.medians[q] == observed.median()

    x = saved.transform(train, 'train')
    columns = saved.columns()
    linear = x[:, :len(LINEAR)]
    np.testing.assert_allclose(linear.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(linear.std(axis=0), 1, atol=1e-5)
    for q in FILLED:
        np.testing.assert_array_equal(
            x[:, columns.index(f'{q}_missing')], (train[q] == '').to_numpy())
    for f in CATEGORICAL:
        block = [i for i, c in enumerate(columns) if c.startswith(f'{f}=')]
        np.testing.assert_array_equal(x[:, block].sum(axis=1), 1)
        if f == 'INDEX_TYPE':           # never missing, so no missing level
            assert f'{f}={MISSING_LEVEL}' not in columns
            continue
        np.testing.assert_array_equal(
            x[:, columns.index(f'{f}={MISSING_LEVEL}')],
            (train[f] == '').to_numpy())


def test_encoding_refuses_what_the_plan_has_no_rule_for():
    train = pd.DataFrame({'L': ['0', '2', '7'],
                          'QUINTMAT': ['1', '', '4'], 'SEX': ['F', 'M', 'F']})
    encoding = BaselineEncoding.fit(train, ['L', 'QUINTMAT'], ['QUINTMAT'],
                                    ['SEX'])
    with pytest.raises(ValueError, match="'L'"):
        encoding.transform(train.assign(L=['0', '', '7']), 'test')
    with pytest.raises(ValueError, match='never did'):
        encoding.transform(train.assign(SEX=['F', 'X', 'M']), 'test')
    # Missing where train never was: no indicator or level was trained.
    complete = train.assign(QUINTMAT=['1', '2', '4'])
    encoding = BaselineEncoding.fit(complete, ['L', 'QUINTMAT'],
                                    ['QUINTMAT'], ['SEX'])
    assert encoding.columns() == ['L', 'QUINTMAT', 'SEX=F', 'SEX=M']
    with pytest.raises(ValueError, match='QUINTMAT'):
        encoding.transform(train, 'test')
    with pytest.raises(ValueError, match='never did'):
        encoding.transform(complete.assign(SEX=['F', '', 'M']), 'test')


def test_lr_check_fits_the_tuning_fold_per_rate(fitted):
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(fitted['root'])
        fit_main([str(fitted['dataset_config']),
                  str(fitted['experiment_config']), '--lr_check',
                  '--device', 'cpu'])
    check = (Path(fitted['cfg']['MODEL_DIR'])
             / f"{fitted['cfg']['EXPERIMENT_NAME']}_lr_check")
    result = yaml.safe_load((check / 'lr_check.yaml').read_text())
    assert result['fold'] == 'fold0'
    assert sorted(result['val_by_learning_rate']) == [0.0002, 0.002]
    for rate in ('0.0002', '0.002'):
        assert (check / f'lr{rate}' / 'fold0' / TASK / WEIGHTS_FILE).exists()
