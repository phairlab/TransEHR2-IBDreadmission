"""The dataloading diagnostic, end to end against a real extracted dataset.

The script exists to be run once on the cluster against the re-extracted cohort,
which is the worst place to discover that a stage raises. The point of this test
is not the numbers it prints -- a two-patient fixture says nothing about
throughput -- but that every stage walks the real `load_dataset` and
`prepare_dataloaders` path without falling over, and that the byte accounting the
rates are derived from is right.
"""

import numpy as np
import pickle
import pytest

from diagnose_dataloading import main, nbytes
from extract_data import main as extract_main

from .conftest import CLINVEC_DIM, CLINVEC_ROWS, MiniRoot  # noqa: F401

TEXT_EMBED_DIM = 5


@pytest.fixture
def extracted(tmp_path):
    """A two-patient cohort, extracted, with the lookup tables beside it."""
    mini = MiniRoot(tmp_path)
    mini.add_patient(
        1001,
        timeseries=[
            ['2019-01-01T00:00:00Z', '', '', '', '', '', ''],
            ['2019-01-02T00:00:00Z', 1.5, 'L', '0', '', 'a note', ''],
            ['2019-01-03T00:00:00Z', 2.5, 'U', '25-50', '', '', 1],
            ['2019-01-04T00:00:00Z', 3.5, 'L', '1-24', '', 'other note', ''],
        ],
        stays=[('DAD', '2019-01-01T00:00:00Z', '2019-01-04T00:00:00Z')],
        drugs=[('2019-01-02T00:00:00Z', 0, 2, 1.0)],
    )
    mini.add_patient(
        1002,
        timeseries=[
            ['2019-02-01T00:00:00Z', 0.5, 'U', '1-24', '', 'a note', 1],
            ['2019-02-02T00:00:00Z', 2.0, 'L', '25-50', '', '', ''],
            ['2019-02-03T00:00:00Z', 4.0, 'U', '0', '', 'third note', 1],
        ],
        stays=[('AMB', '2019-02-01T00:00:00Z', '2019-02-03T00:00:00Z')],
        drugs=[('2019-02-01T00:00:00Z', 0, 1, 2.0)],
    )
    mini.add_fold('fold0', train=[0, 1], val=[0, 1], test=[0, 1])
    assert extract_main([str(mini.finish())]) == 0

    with open(mini.extracted / 'text_strings.pkl', 'rb') as f:
        n_strings = len(pickle.load(f))
    np.save(mini.lookup_tables / 'text_embeddings.npy',
            np.repeat(np.arange(1, n_strings + 1, dtype=np.float32)[:, None],
                      TEXT_EMBED_DIM, axis=1))
    np.save(mini.lookup_tables / 'drug_embeddings.npy',
            np.vstack([np.repeat(np.arange(1, CLINVEC_ROWS + 1,
                                           dtype=np.float32)[:, None],
                                 CLINVEC_DIM, axis=1),
                       np.zeros((1, CLINVEC_DIM), dtype=np.float32)]))
    return mini


def _run(mini, stages, extra=()):
    return main([
        '--data-dir', str(mini.data_dir), '--fold', 'fold0',
        '--batch-size', '1', '--batches', '2', '--workers', '0',
        '--workers-for-device', '0', '--repeats', '1',
        '--batch-sizes', '1,2', '--profile-items', '2',
        '--device', 'cpu', '--stages', stages, *extra,
    ])


@pytest.mark.parametrize('stage', ['raw', 'loader', 'batch', 'device',
                                  'padding', 'profile'])
def test_every_stage_runs_against_a_real_extraction(extracted, stage, capsys):
    assert _run(extracted, stage) == 0
    assert capsys.readouterr().out.strip(), f'{stage} printed nothing'


def test_all_stages_run_in_one_invocation(extracted, capsys):
    assert _run(extracted, 'all') == 0
    out = capsys.readouterr().out
    for heading in ('raw:', 'loader:', 'batch:', 'padding:', 'profile:'):
        assert heading in out, heading


def test_the_byte_count_walks_the_nested_batch():
    """Rates are bytes over seconds, so a miscount is a wrong answer, not a typo."""
    import torch

    batch = {
        'val_data': {
            'masks': torch.ones(2, 4),                       # 32 B
            'numeric': {'values': [torch.ones(2, 4, 3),      # 96 B
                                   torch.ones(2, 4, 1)]},    # 32 B
        },
        'idx': 7,                                            # not a tensor
    }
    assert nbytes(batch) == (2 * 4 + 2 * 4 * 3 + 2 * 4 * 1) * 4


def test_padding_is_reported_against_the_array_width(extracted, capsys):
    """The stage that bounds what a shorter extraction could win."""
    assert _run(extracted, 'padding') == 0
    out = capsys.readouterr().out
    assert 'padding' in out and '% of every byte read' in out
