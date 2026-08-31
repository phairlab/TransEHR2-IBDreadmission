"""Regression probe: the static width the model builds for must match the width on disk.

`MixedClassifier` concatenates the value embedding with `static_data` before its
feedforward layer, so that layer's input width is `d_model + static_width`. The
entry points passed `len(STATIC_FEATS)` -- a feature *count*. The extraction
writes `sum(static_feat_dims)`, and since 75c39e8 made categorical encoding
actually one-hot, a categorical static occupies `size` columns rather than one.
Upstream (9b47377) that divergence killed the first forward pass with
`mat1 and mat2 shapes cannot be multiplied`.

Here it is **dormant**: `RMT23345.yaml` sets `STATIC_FEATS: []` (blueprint A.3 --
every REG feature is attached to every timestep instead), so both derivations
give 0 and nothing currently breaks. These probes defend the property that they
cannot diverge once a static feature of `size > 1` is added.
"""

import os

import pytest

from TransEHR2.data.preprocessing import compute_static_feat_dims


REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

ENTRY_POINTS = [
    'run_experiment_accelerate.py',
    'tune_hyperparameters_accelerate.py',
    'dump_finetuned_predictions.py',
    os.path.join('TransEHR2', 'test_tune_hyperparameters.py'),
]


def test_a_multicolumn_static_is_wider_than_the_feature_count():
    """The bug in one line: a one-hot static makes count and width disagree."""
    var_properties = {
        'AGE': {'type': 'numeric', 'size': 1},
        'SEX': {'type': 'categorical', 'size': 3},
    }
    static_feats = ['AGE', 'SEX']
    dims = compute_static_feat_dims(var_properties, static_feats)
    assert dims == [1, 3]
    assert sum(dims) == 4
    assert sum(dims) != len(static_feats), (
        'count and width must be distinguishable, or the probe defends nothing'
    )


def test_size_is_the_width_for_every_type():
    """Blueprint 4.3: `size` is the per-timestep dimension for every type."""
    var_properties = {
        'ORD': {'type': 'ordinal', 'size': 5},
        'TXT': {'type': 'text', 'size': 1},
        'DRG': {'type': 'drug', 'size': 30},
    }
    dims = compute_static_feat_dims(var_properties, ['ORD', 'TXT', 'DRG'])
    assert dims == [5, 1, 30]


def test_an_empty_static_list_is_zero_wide():
    """The shipped RMT23345 case: no statics, so both derivations give 0."""
    assert compute_static_feat_dims({}, []) == []
    assert sum(compute_static_feat_dims({}, [])) == 0


def test_extraction_routes_through_the_shared_helper():
    """`_get_tensor_dimensions` must not reimplement the derivation."""
    import inspect

    from TransEHR2.data import preprocessing

    source = inspect.getsource(preprocessing._get_tensor_dimensions)
    assert 'compute_static_feat_dims(' in source, (
        'the extraction has stopped using the shared helper, so the two '
        'derivations can drift apart again -- which is what this prevents'
    )


def test_entry_points_do_not_pass_the_feature_count():
    """Guard every entry point at once: none may size statics by count again."""
    offenders = []
    for name in ENTRY_POINTS:
        path = os.path.join(REPO_ROOT, name)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for lineno, line in enumerate(fh, 1):
                stripped = line.strip()
                # The name appears in explanatory comments; only real code counts.
                if stripped.startswith('#'):
                    continue
                if 'len(STATIC_FEATS)' in stripped:
                    offenders.append(f'{name}:{lineno}: {stripped}')
    assert not offenders, (
        'static dimensions must come from compute_static_feat_dims(), not a '
        'feature count:\n' + '\n'.join(offenders)
    )
