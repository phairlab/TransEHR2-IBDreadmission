"""Multi-label features, ported from upstream caf91a2 as inert plumbing.

RMT23345 declares no multi-label feature -- section 4.3's type table lists
numeric, categorical, ordinal, text and drug -- so nothing here exercises the
path in production. The port exists so that upstream changes touching the
multilabel branches apply cleanly. These probes therefore have to supply their
own feature, because the shipped config cannot.

A multi-hot cannot be expressed as a single index, so unlike categorical and
ordinal the value array is stored dense and `__getitem__` performs no
expansion. The cell format is semicolon-separated labels.
"""

import numpy as np
import torch

from TransEHR2.data.preprocessing import collate_tensorized
from TransEHR2.utils import generate_record_masks

from .conftest import STAYS_COLUMNS, write_csv
from .test_datasets import extracted

TS_COLUMNS = ['TIMESTAMP', 'NUM', 'CAT', 'ORD', 'UB', 'TXT', 'EVT', 'MULTI']

# Row 1 names two of the three labels, row 2 names one, row 3 names a label
# that is not declared -- observed but out of domain.
ROWS = [
    ['2019-01-01T00:00:00Z', '', '', '', '', '', '', ''],
    ['2019-01-02T00:00:00Z', 1.5, 'L', '0', 'None', 'a note', '', 'a;c'],
    ['2019-01-03T00:00:00Z', 2.5, 'U', '25-50', 'Few', '', 1, 'b'],
    ['2019-01-04T00:00:00Z', 3.5, 'L', '1-24', '', 'a note', '', 'zzz'],
]


def _with_multilabel(mini):
    """The mini cohort plus one declared multi-label feature."""
    mini.var_properties['MULTI'] = {
        'type': 'multilabel', 'size': 3,
        'category_map': {0: 'a', 1: 'b', 2: 'c'},
    }
    mini.config['VALUED_FEATS'] = ['NUM', 'CAT', 'ORD', 'UB', 'MULTI']
    mini.add_patient(
        1001,
        timeseries=[r[:-1] for r in ROWS],
        stays=[('DAD', '2019-01-01T00:00:00Z', '2019-01-04T00:00:00Z')],
        drugs=[('2019-01-01T00:00:00Z', 0, 2, 1.0)],
    )
    # add_patient writes the fixture's fixed column set; rewrite with MULTI.
    write_csv(mini.root / '1001' / 'timeseries.csv', ROWS, TS_COLUMNS)
    return mini


def test_semicolons_become_a_multi_hot_row(mini):
    dataset = extracted(_with_multilabel(mini))
    values = dataset[0]['val_multilabel_values'][0].numpy()
    indicators = dataset[0]['val_multilabel_indicators'].numpy()

    # Timesteps are right-aligned, so the four rows land in order.
    assert values.shape == (4, 3), values.shape
    np.testing.assert_array_equal(values[1], [1.0, 0.0, 1.0])  # 'a;c'
    np.testing.assert_array_equal(values[2], [0.0, 1.0, 0.0])  # 'b'
    # Observed but undeclared: indicator set, row all zero (section 4.3).
    np.testing.assert_array_equal(values[3], [0.0, 0.0, 0.0])
    assert indicators[3, 0] == 1.0
    # Unobserved: indicator clear, row all zero.
    np.testing.assert_array_equal(values[0], [0.0, 0.0, 0.0])
    assert indicators[0, 0] == 0.0


def test_multilabel_survives_collate_and_masking(mini):
    dataset = extracted(_with_multilabel(mini))
    batch = collate_tensorized([dataset[0], dataset[0]])
    block = batch['val_data']['multilabel']
    assert block['indicators'].shape == (2, 4, 1)
    assert block['values'][0].shape == (2, 4, 3)
    masks = generate_record_masks(batch)[0]
    assert 'multilabel' in masks, 'masking skipped the multilabel family'


def test_the_generator_is_not_shown_its_multilabel_target(mini):
    """caf91a2 predates d4a189a, so its multilabel branch carried the old
    polarity: it multiplied by the mask directly and was handed exactly the
    components it must reconstruct. The port inverts it like the rest."""
    from TransEHR2.modules import MaskedTokenGenerator

    class _Capture(torch.nn.Module):
        d_model = 8

        def forward(self, indicators, values, timestamps, timestep_masks):
            self.seen = values
            return torch.zeros(values.shape[0], values.shape[1], self.d_model)

    capture = _Capture()
    generator = MaskedTokenGenerator(
        encoder=capture, d_model=8, numeric_dims=[], categorical_classes=[],
        multilabel_classes=[3],
    )
    generator.eval()

    b, t = 2, 4
    values = torch.ones(b, t, 3)
    mask = torch.zeros(b, t, 3)
    mask[:, :, 1] = 1.0  # component 1 is the reconstruction target
    batch = {
        'multilabel': {'indicators': torch.ones(b, t, 1), 'values': [values]},
        'times': torch.zeros(b, t),
        'masks': torch.ones(b, t),
    }
    record_masks = {
        'multilabel': {'indicators': torch.zeros(b, t, 1), 'values': [mask]}
    }
    with torch.no_grad():
        generator(batch, record_masks)

    seen = capture.seen
    assert torch.all(seen[:, :, 1] == 0.0), (
        'the generator was shown the multilabel components it must predict'
    )
    assert torch.all(seen[:, :, 0] == 1.0), 'context was zeroed instead'
