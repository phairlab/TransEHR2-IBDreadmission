"""The benchmark's stage-1 sizing.

The rest of `benchmark_grad_attribution.py` is measurement, which a test cannot check. This
part is not: it decides what shape gets measured, and a wrong shape produces a plausible number
for a model the study does not train. The real config lives on the cluster, so the two files are
fabricated here and the assertion is that the widths come back per feature and summed as
`run_experiment` sums them.
"""

import yaml

from benchmark_grad_attribution import dataset_widths


def test_the_widths_are_per_feature_and_follow_variable_properties(tmp_path):
    properties = tmp_path / 'variable_properties.yaml'
    properties.write_text(yaml.safe_dump({
        'HGB': {'type': 'numeric', 'size': 1},
        'ZONE_NAME': {'type': 'categorical', 'size': 6,
                      'category_map': {str(i): i for i in range(6)}},
        'QUINTMAT': {'type': 'ordinal', 'size': 5},
    }))
    config = tmp_path / 'dataset.yaml'
    config.write_text(yaml.safe_dump({
        'VARIABLE_PROPERTIES_PATH': str(properties),
        'VALUED_FEATS': ['HGB', 'ZONE_NAME', 'QUINTMAT'],
        'TEXT_FEATS': ['TEXT_SUPERFEATURE'],
        'MAX_EPISODE_LEN_STEPS': 500,
    }))

    steps, widths = dataset_widths(str(config))

    assert steps == 500
    # Three features, twelve columns: count and width diverge the moment one is one-hot, which
    # is exactly what repeating a single --feat-width cannot represent.
    assert widths == [1, 6, 5]
    assert len(widths) == 3 and sum(widths) == 12


# ---------------------------------------------------------------------------
# Episode occupancy
# ---------------------------------------------------------------------------

def _extracted(tmp_path, masks, families):
    import numpy as np

    np.save(tmp_path / 'val_masks.npy', np.asarray(masks, dtype=np.float32))
    for name, array in families.items():
        np.save(tmp_path / f'{name}_indicators.npy', np.asarray(array, dtype=np.float32))
    return str(tmp_path)


def test_occupancy_counts_observed_cells_within_real_timesteps(tmp_path):
    from episode_occupancy import occupancy

    # Two episodes of length 2 and 3 in a width-4 array, left-padded as extraction writes them.
    data = _extracted(
        tmp_path,
        masks=[[0, 0, 1, 1], [0, 1, 1, 1]],
        families={
            'val_numeric': [[[1, 0], [1, 0], [1, 1], [0, 1]],
                            [[1, 1], [1, 0], [0, 0], [1, 1]]],
            'val_lookup': [[[1], [0], [0], [1]],
                           [[0], [1], [1], [0]]],
        })

    lengths, observed, n_features = occupancy(data)

    assert n_features == 3
    assert list(lengths) == [2, 3]
    # Episode 0's real timesteps are the last two: (1,1)+(0) and (0,1)+(1) = 2 + 2.
    # Episode 1's are the last three: (1,0)+(1), (0,0)+(1), (1,1)+(0) = 2 + 1 + 2.
    assert list(observed) == [4, 5]


def test_occupancy_padding_is_never_counted_as_absence(tmp_path):
    """A padded timestep is not a minute where nothing was observed; it is not a minute."""
    from episode_occupancy import occupancy

    data = _extracted(tmp_path, masks=[[0, 0, 0, 1]],
                      families={'val_numeric': [[[1, 1], [1, 1], [1, 1], [1, 0]]]})
    lengths, observed, _ = occupancy(data)
    assert list(lengths) == [1] and list(observed) == [1]


def test_a_recent_window_is_a_suffix_because_extraction_left_pads(tmp_path):
    from episode_occupancy import occupancy

    data = _extracted(tmp_path, masks=[[0, 1, 1, 1]],
                      families={'val_numeric': [[[1, 1], [1, 1], [0, 0], [1, 1]]]})
    lengths, observed, _ = occupancy(data, max_steps=2)
    # The last two timesteps only: (0,0) and (1,1).
    assert list(lengths) == [2] and list(observed) == [2]


def test_the_event_branch_is_left_out_of_the_pass_count(tmp_path):
    """TSR never perturbs the event branch, so its occupancy does not enter the budget."""
    from episode_occupancy import indicator_paths

    data = _extracted(tmp_path, masks=[[1]],
                      families={'val_numeric': [[[1]]], 'event': [[[1]]]})
    names = [p.rsplit('/', 1)[-1] for p in indicator_paths(data)]
    assert names == ['val_numeric_indicators.npy']
