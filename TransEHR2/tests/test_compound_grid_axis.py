"""One grid axis that writes several config keys.

The encoder depth is the case this exists for. TransEHR2 spells it twice --
``GENERATOR_ENCODER_N_ENCODER_BLOCKS`` and ``DISCRIMINATOR_ENCODER_N_ENCODER_BLOCKS`` -- but it
is one architectural choice, and sweeping the two names as two axes would cross them into four
cells, two of which pair a generator depth with a different discriminator depth. Those are
configurations nobody meant to run, and no reading of the result could untangle them from the
two that were meant.

``sets`` names the keys an axis writes. The trial's ``cell`` stays keyed by the axis, so a
ranking reads one depth rather than two equal numbers; only ``overrides``, which reaches the
config file, is expanded.
"""

import pytest
import yaml

from TransEHR2.hp_tuning.spec import (axis_keys, expand_assignment, expand_trials,
                                      load_spec)


def _written(spec, tmp_path):
    """Validation happens in load_spec, so a rejection test goes through a file."""
    path = tmp_path / 'spec.yaml'
    path.write_text(yaml.safe_dump(spec))
    return str(path)


def _spec(**grid_extra):
    grid = {
        'CMPNT_MASK_RATIO': {'values': [0.25, 0.5], 'select_on': 'readmission'},
        'N_ENCODER_BLOCKS': {
            'values': [1, 2],
            'sets': ['GENERATOR_ENCODER_N_ENCODER_BLOCKS',
                     'DISCRIMINATOR_ENCODER_N_ENCODER_BLOCKS'],
            'select_on': 'readmission',
        },
    }
    for name, entry in grid_extra.items():
        grid[name] = entry
    return {
        'SPEC_NAME': 'phase3', 'DESIGN': 'factorial', 'FOLD': 'fold0',
        'BASE_CONFIG': 'b.yaml', 'DATASET_CONFIG': 'd.yaml',
        'OUTPUT_DIR': 'out', 'MANIFEST': 'm.yaml',
        'ARMS': {'rope': {'POSITION_ENCODING': 'rope'}},
        'GRID': grid,
        'ALIASES': {'CMPNT_MASK_RATIO': 'cmask', 'N_ENCODER_BLOCKS': 'blocks'},
    }


def test_an_axis_without_sets_writes_its_own_name():
    assert axis_keys({'values': [1]}, 'CMPNT_MASK_RATIO') == ['CMPNT_MASK_RATIO']


def test_an_axis_with_sets_writes_all_of_them():
    entry = {'values': [1], 'sets': ['A', 'B']}
    assert axis_keys(entry, 'N_ENCODER_BLOCKS') == ['A', 'B']


def test_one_value_reaches_every_key_the_axis_sets():
    grid = _spec()['GRID']
    expanded = expand_assignment(grid, {'CMPNT_MASK_RATIO': 0.5,
                                        'N_ENCODER_BLOCKS': 2})
    assert expanded == {'CMPNT_MASK_RATIO': 0.5,
                        'GENERATOR_ENCODER_N_ENCODER_BLOCKS': 2,
                        'DISCRIMINATOR_ENCODER_N_ENCODER_BLOCKS': 2}


def test_the_depth_is_never_crossed_with_itself():
    """The whole point: two values of one axis, not four combinations of two axes."""
    trials = expand_trials(_spec())
    assert len(trials) == 4            # 2 mask ratios x 2 depths

    for trial in trials:
        overrides = trial['overrides']
        assert (overrides['GENERATOR_ENCODER_N_ENCODER_BLOCKS']
                == overrides['DISCRIMINATOR_ENCODER_N_ENCODER_BLOCKS'])


def test_the_cell_stays_keyed_by_the_axis_not_by_the_keys():
    """A ranking should report one depth, not the same number twice."""
    trials = expand_trials(_spec())
    assert set(trials[0]['cell']) == {'CMPNT_MASK_RATIO', 'N_ENCODER_BLOCKS'}


def test_the_axis_name_does_not_leak_into_the_config():
    """`generate_tuning_configs` rejects an override the base config has no key for, so a
    leaked axis name would fail generation rather than be ignored."""
    for trial in expand_trials(_spec()):
        assert 'N_ENCODER_BLOCKS' not in trial['overrides']


def test_the_trial_name_still_carries_one_token_for_the_axis():
    names = sorted(trial['name'] for trial in expand_trials(_spec()))
    assert names[0] == 'phase3_rope_cmask_0p25_blocks_1'


def test_two_axes_writing_one_key_is_refused(tmp_path):
    """Grid order would otherwise decide which wins, silently."""
    spec = _spec(OTHER={'values': [1, 2],
                        'sets': ['GENERATOR_ENCODER_N_ENCODER_BLOCKS'],
                        'select_on': 'readmission'})
    with pytest.raises(ValueError, match='both set'):
        load_spec(_written(spec, tmp_path))


def test_a_malformed_sets_is_refused(tmp_path):
    spec = _spec()
    spec['GRID']['N_ENCODER_BLOCKS']['sets'] = []
    with pytest.raises(ValueError, match='non-empty list'):
        load_spec(_written(spec, tmp_path))


def test_a_repeated_key_in_sets_is_refused(tmp_path):
    spec = _spec()
    spec['GRID']['N_ENCODER_BLOCKS']['sets'] = ['A', 'A']
    with pytest.raises(ValueError, match='repeats a key'):
        load_spec(_written(spec, tmp_path))


# --- whether a phase pretrains at all ---------------------------------------

def test_a_sweep_pretrains_unless_it_says_otherwise():
    """Absent reads as true: every sweep written before PRETRAINS existed did pretrain, and
    their manifests are on disk."""
    for trial in expand_trials(_spec()):
        assert trial['needs_pretrain'] is True


def test_a_sweep_can_declare_that_it_does_not_pretrain():
    spec = _spec()
    spec['PRETRAINS'] = False
    for trial in expand_trials(spec):
        assert trial['needs_pretrain'] is False


def test_the_pretrain_stage_is_empty_for_such_a_sweep():
    """Which is what makes submit_tune_array.sh refuse to queue it. The alternative is four
    jobs that each take a GPU, load a symlinked encoder, and skip to finetuning."""
    from TransEHR2.hp_tuning.spec import finetune_trials, pretrain_trials
    spec = _spec()
    spec['PRETRAINS'] = False
    manifest = {'trials': expand_trials(spec)}

    assert pretrain_trials(manifest) == []
    assert len(finetune_trials(manifest)) == 4


def test_a_manifest_written_before_the_flag_still_pretrains():
    """Phase 3's manifest is already on the cluster without the key."""
    from TransEHR2.hp_tuning.spec import pretrain_trials
    trials = expand_trials(_spec())
    for trial in trials:
        del trial['needs_pretrain']
    assert len(pretrain_trials({'trials': trials})) == len(trials)


def test_the_written_manifest_still_says_it_does_not_pretrain(tmp_path):
    """The one above checks the trials in memory, which is not where the submitter reads
    them. write_manifest copies a fixed list of fields, so a field left off that list is
    computed correctly and then lost: PRETRAINS was inert on disk for exactly that reason,
    and phase 3 reported a twelve-cell pretrain stage that phase 2 never had."""
    from TransEHR2.hp_tuning.spec import (load_manifest, pretrain_trials,
                                          write_manifest)
    spec = _spec()
    spec['PRETRAINS'] = False
    spec['SPEC_PATH'] = str(tmp_path / 'spec.yaml')
    spec['MODEL_DIR'] = str(tmp_path / 'models')
    spec['MANIFEST'] = str(tmp_path / 'm.yaml')
    trials = expand_trials(spec)
    for trial in trials:
        trial['config'] = str(tmp_path / f"{trial['name']}.yaml")

    written = load_manifest(write_manifest(spec, trials))

    assert all(t['needs_pretrain'] is False for t in written['trials'])
    assert pretrain_trials(written) == []


def test_a_non_boolean_pretrains_is_refused(tmp_path):
    spec = _spec()
    spec['PRETRAINS'] = 'no'
    with pytest.raises(ValueError, match='expected true or false'):
        load_spec(_written(spec, tmp_path))
