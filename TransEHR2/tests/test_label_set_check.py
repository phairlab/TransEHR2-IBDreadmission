"""The entry points refuse a DATA_DIR that holds another set's labels."""

import pytest

from TransEHR2.cli import LABEL_SET_FILE, check_label_set


def label_set_dir(tmp_path, name=None):
    if name is not None:
        (tmp_path / LABEL_SET_FILE).write_text(f"{name}\n")
    return str(tmp_path)


def test_the_current_labels_need_no_label_set(tmp_path):
    check_label_set(label_set_dir(tmp_path), {'EXPERIMENT_NAME': 'x'})


def test_a_matching_label_set_passes(tmp_path):
    check_label_set(label_set_dir(tmp_path, 'ibd'),
                    {'EXPERIMENT_NAME': 'x', 'LABEL_SET': 'ibd'})


@pytest.mark.parametrize('held, wanted', [
    (None, 'ibd'),       # a label-set experiment on the current DATA_DIR
    ('ibd', None),       # a current-label experiment on a label set
    ('ibd', 'other'),
])
def test_a_mismatch_is_refused(tmp_path, held, wanted):
    config = {'EXPERIMENT_NAME': 'x'}
    if wanted is not None:
        config['LABEL_SET'] = wanted
    with pytest.raises(ValueError, match='LABEL_SET'):
        check_label_set(label_set_dir(tmp_path, held), config)


def test_label_set_does_not_block_reusing_pretraining():
    # Pretraining reads no labels, so a label set's experiment may start
    # from the current experiment's encoders (PRETRAINED_FROM).
    from fit_transehr2_deephit import POST_PRETRAINING_KEYS
    assert 'LABEL_SET' in POST_PRETRAINING_KEYS
