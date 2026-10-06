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
