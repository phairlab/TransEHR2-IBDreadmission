"""The TSR benchmark, end to end against a real extraction.

The point is not the times a two-patient fixture prints -- they mean nothing --
but that the script builds the *production* classifier, walks the real loader
and runs `tsr_scores` over it without falling over. That is the failure this
script exists to prevent: `benchmark_grad_attribution.py` assembled a model by
hand at hardcoded widths, so when the real one changed its numbers went quietly
stale rather than breaking. A test that constructs the real builder is what
makes that drift loud.
"""

import numpy as np
import pickle
import pytest
import re
import yaml

from pathlib import Path

from benchmark_tsr import main, split_loader

from .conftest import CLINVEC_DIM, CLINVEC_ROWS, MiniRoot  # noqa: F401

TEXT_EMBED_DIM = 5


@pytest.fixture
def cohort(tmp_path):
    """Two patients, extracted, with the lookup tables and an experiment config."""
    from extract_data import main as extract_main

    mini = MiniRoot(tmp_path)
    mini.add_patient(
        1001,
        timeseries=[
            ['2019-01-01T00:00:00Z', 1.5, 'L', '0', '', 'a note', ''],
            ['2019-01-02T00:00:00Z', 2.5, 'U', '25-50', '', '', 1],
            ['2019-01-03T00:00:00Z', 3.5, 'L', '1-24', '', 'other note', ''],
        ],
        stays=[('DAD', '2019-01-01T00:00:00Z', '2019-01-03T00:00:00Z')],
        drugs=[('2019-01-02T00:00:00Z', 0, 2, 1.0)])
    mini.add_patient(
        1002,
        timeseries=[
            ['2019-02-01T00:00:00Z', 0.5, 'U', '1-24', '', 'a note', 1],
            ['2019-02-02T00:00:00Z', 2.0, 'L', '0', '', 'third note', ''],
        ],
        stays=[('AMB', '2019-02-01T00:00:00Z', '2019-02-02T00:00:00Z')],
        drugs=[('2019-02-01T00:00:00Z', 0, 1, 2.0)])
    mini.add_fold('fold0', train=[0, 1], val=[0, 1], test=[0, 1])
    dataset_config = mini.finish()
    assert extract_main([str(dataset_config)]) == 0

    with open(mini.extracted / 'text_strings.pkl', 'rb') as handle:
        n_strings = len(pickle.load(handle))
    np.save(mini.lookup_tables / 'text_embeddings.npy',
            np.zeros((n_strings, TEXT_EMBED_DIM), dtype=np.float32))
    np.save(mini.lookup_tables / 'drug_embeddings.npy',
            np.zeros((CLINVEC_ROWS + 1, CLINVEC_DIM), dtype=np.float32))

    # Defaults carry every DISCRIMINATOR_* setting; the event branch is off
    # because this cohort's episodes are too small to make it interesting and
    # TSR never perturbs it.
    experiment_config = tmp_path / 'experiment.yaml'
    experiment_config.write_text(yaml.safe_dump({
        'USE_EVENT_ENCODER': False,
        'TIME_GRID_CUTS_DAYS': [30.0, 365.0],
        'MODELLED_CAUSES': ['readmission', 'death'],
    }))
    return str(dataset_config), str(experiment_config)


def _run(cohort, extra=()):
    dataset_config, experiment_config = cohort
    return main([dataset_config, experiment_config, '--device', 'cpu',
                 '--fold', 'fold0', '--split', 'test', '--batch-sizes', '1',
                 '--batches', '1', '--warmup', '1', '--workers', '0',
                 '--chunk', '8', *extra])


def test_it_runs_the_production_classifier_over_real_episodes(cohort, capsys):
    assert _run(cohort) == 0
    out = capsys.readouterr().out
    assert 'tsr/ep' in out, out
    assert 'grad x input' in out, out


def test_integrated_gradients_is_reported_as_such(cohort, capsys):
    assert _run(cohort, ('--ig-steps', '2')) == 0
    assert 'IG x2' in capsys.readouterr().out


def test_every_batch_size_in_the_sweep_gets_a_row(cohort, capsys):
    dataset_config, experiment_config = cohort
    assert main([dataset_config, experiment_config, '--device', 'cpu',
                 '--fold', 'fold0', '--split', 'test', '--batch-sizes', '1,2',
                 '--batches', '1', '--warmup', '0', '--workers', '0',
                 '--chunk', '8']) == 0
    body = [line for line in capsys.readouterr().out.splitlines()
            if line.strip().startswith(('1 ', '2 ')) or
            line.strip()[:1].isdigit()]
    assert len(body) >= 2, body


def test_the_split_is_named_not_positional():
    """A two-loader tree has no val, so index 2 would be the wrong partition."""
    assert split_loader(['tr', 'va', 'te'], 'test') == 'te'
    assert split_loader(['tr', 'te'], 'test') == 'te'
    assert split_loader(['tr', 'va', 'te'], 'val') == 'va'


def test_a_config_without_grid_settings_falls_back_to_the_defaults(cohort, capsys,
                                                                   tmp_path):
    """`.get` returning None would override TimeGrid's defaults with None."""
    dataset_config, _ = cohort
    bare = tmp_path / 'bare.yaml'
    bare.write_text(yaml.safe_dump({'USE_EVENT_ENCODER': False}))
    assert main([dataset_config, str(bare), '--device', 'cpu', '--fold', 'fold0',
                 '--split', 'test', '--batch-sizes', '1', '--batches', '1',
                 '--warmup', '0', '--workers', '0', '--chunk', '8']) == 0
    assert 'tsr/ep' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------

def _structured_strings(cohort):
    """Replace the fixture's free-text notes with superfeature strings, row for row."""
    from IBDdataprep.superfeature import dump_superfeature

    dataset_config, _ = cohort
    extracted = Path(yaml.safe_load(Path(dataset_config).read_text())['DATA_DIR']) / 'extracted'
    path = extracted / 'text_strings.pkl'
    with open(path, 'rb') as handle:
        n_strings = len(pickle.load(handle))
    strings = [dump_superfeature({'DIAGNOSES': [{'DX_DESC': f'diagnosis {i}'}],
                                  'PROCEDURES': [f'procedure {i}']})
               for i in range(n_strings)]
    with open(path, 'wb') as handle:
        pickle.dump(strings, handle)


def test_phase2_times_every_cost_and_projects_the_cohort(cohort, capsys, monkeypatch):
    import torch

    import benchmark_tsr

    _structured_strings(cohort)
    monkeypatch.setattr(
        benchmark_tsr, 'build_embedder',
        lambda device, batch_size: (lambda strings: torch.randn(len(strings),
                                                                TEXT_EMBED_DIM)))
    dataset_config, experiment_config = cohort
    assert main([dataset_config, experiment_config, '--device', 'cpu', '--fold', 'fold0',
                 '--split', 'test', '--warmup', '0', '--batches', '2', '--workers', '0',
                 '--chunk', '8', '--phase2', '--horizon-days', '30', '--ig-steps', '2',
                 '--cohort', '10']) == 0
    out = capsys.readouterr().out
    for label in ('horizon 30 d (interpolated)', 'tsr ', 'fields ', 'fwd ', 'curves ',
                  'cohort '):
        assert label in out, out
    assert 'diagnosis' not in out
    # The text path really ran: the fixture's episodes carry notes, so strings were embedded.
    embedded = re.search(r'over (\d+) strings', out)
    assert embedded and int(embedded.group(1)) > 0, out
