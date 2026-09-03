"""Does a feature type the extraction left empty keep its timestep axis?

Upstream ef1dc03 fixed a crash where ``load_dataset`` substituted a
``(0, 0, 0)`` array for a zero-feature type, so ``__getitem__`` returned a
1-D ``torch.empty(0)``; that collates to ``(batch, features)`` and
``_gen_val_assoc_feat_mask`` unpacks exactly three dimensions, so every
run died on its first batch. This fork loads the real array instead, so
the probe asks whether the same hazard exists here at all.
"""

import numpy as np
import torch

from TransEHR2.data.preprocessing import collate_tensorized
from TransEHR2.utils import generate_record_masks

from .test_datasets import extracted


def _no_ordinals(mini):
    """The mini cohort with every ordinal feature removed from the config."""
    mini.config['VALUED_FEATS'] = ['NUM', 'CAT']
    # The feature contract: every variable_properties entry must appear in some
    # config feature list, so the entries go with the config change.
    for feat in ('ORD', 'UB'):
        del mini.var_properties[feat]
    mini.add_patient(
        1001,
        timeseries=[
            ['2019-01-01T00:00:00Z', '', '', '', '', '', ''],
            ['2019-01-02T00:00:00Z', 1.5, 'L', '', '', 'a note', ''],
            ['2019-01-03T00:00:00Z', 2.5, 'U', '', '', '', 1],
            ['2019-01-04T00:00:00Z', 3.5, 'L', '', '', 'a note', ''],
        ],
        stays=[('DAD', '2019-01-01T00:00:00Z', '2019-01-04T00:00:00Z')],
        drugs=[('2019-01-01T00:00:00Z', 0, 2, 1.0)],
    )
    return mini


def test_zero_feature_type_keeps_its_timestep_axis(mini):
    dataset = extracted(_no_ordinals(mini))
    item = dataset[0]
    ind = item['val_ordinal_indicators']
    assert ind.ndim == 2, (
        f'ordinal indicators came back {ind.ndim}-D {tuple(ind.shape)}; '
        'the timestep axis is gone and collate cannot restore it'
    )
    assert ind.shape[1] == 0


def test_zero_feature_type_survives_collate_and_masking(mini):
    dataset = extracted(_no_ordinals(mini))
    batch = collate_tensorized([dataset[0], dataset[0]])
    ind = batch['val_data']['ordinal']['indicators']
    assert ind.ndim == 3, (
        f'collated ordinal indicators are {ind.ndim}-D {tuple(ind.shape)}; '
        '_gen_val_assoc_feat_mask unpacks exactly three dimensions'
    )
    # The unpack that upstream's bug reached: must not raise.
    generate_record_masks(batch)
