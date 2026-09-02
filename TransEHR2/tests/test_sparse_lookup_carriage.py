"""The lookup family crosses the worker boundary sparsely.

``__getitem__`` used to densify each lookup feature to its full
``(T, D)`` or ``(T, S, D)`` extent before collation, and every byte of
that crossed the worker boundary and one host-side copy before reaching
the device. Records in the family are rare against the timestep axis --
section 4.5 measures ~1.2 KB of text and ~36 KB of drugs per episode
against ~1.1 KB per timestep of everything else -- so the dense form is
almost entirely zeros. At section 4.5's ``T = 500`` and a batch of 200 it
is about 4.8 GB per batch carrying about 7 MB of content.

The item now carries the CSR slice itself and ``densify_lookup_slots``
rebuilds the dense tensors inside ``move_batch_to_device``, on the device
the batch has landed on. Downstream sees the same ``slot_values`` /
``doses`` / ``masks`` it saw before, so this is a transfer format and not
a change to the model path.

``fixtures/sparse_lookup_baseline.npz`` holds the dense tensors captured
from the tree as it stood before the change, which is what makes the
equivalence claim checkable rather than self-referential.
"""

import numpy as np
import pytest
import torch

from pathlib import Path

from TransEHR2.data.preprocessing import collate_tensorized
from TransEHR2.utils import (
    densify_lookup_slots,
    generate_record_masks,
    move_batch_to_device,
)

from .carriage_cohort import build_carriage_root
from .conftest import CLINVEC_DIM
from .test_lookup_family import TEXT_EMBED_DIM, extract_and_load

BASELINE = Path(__file__).parent / 'fixtures' / 'sparse_lookup_baseline.npz'


@pytest.fixture
def carriage(tmp_path):
    """The dataset and the collated batch over all three episodes."""
    dataset = extract_and_load(build_carriage_root(tmp_path))
    batch = collate_tensorized([dataset[i] for i in range(3)])
    return {'dataset': dataset, 'batch': batch}


def _entry_counts(dataset):
    """CSR entries per lookup feature, over the whole cohort."""
    return [
        int(csr['offsets'][len(dataset)]) - int(csr['offsets'][0])
        for csr in dataset.lookup_csr
    ]


# --- what crosses the boundary ---------------------------------------

def test_the_collated_batch_carries_no_dense_slot_tensor(carriage):
    """The dense tensors are absent until something densifies."""
    lookup = carriage['batch']['val_data']['lookup']
    assert 'sparse' in lookup
    assert 'slot_values' not in lookup
    assert 'doses' not in lookup
    assert 'masks' not in lookup


def test_the_carried_values_are_exactly_the_csr_entries(carriage):
    """Not a ratio: the block's row count *is* the entry count.

    A saving stated as a fraction would pass on a fixture this small
    whatever the block held. The claim is that nothing beyond the CSR
    slice crosses, so the count is exact.
    """
    lookup = carriage['batch']['val_data']['lookup']
    expected = _entry_counts(carriage['dataset'])
    for block, n_entries in zip(lookup['sparse'], expected):
        assert block['values'].shape[0] == n_entries
        assert block['timestep_index'].shape == (n_entries,)
        assert block['episode_index'].shape == (n_entries,)


def test_a_single_slot_feature_carries_no_dose_or_mask(carriage):
    """The family's degenerate case survives the new format."""
    text, drug = carriage['batch']['val_data']['lookup']['sparse']
    assert text['doses'] is None and text['masks'] is None
    assert drug['doses'] is not None and drug['masks'] is not None


# --- equivalence to the dense path ------------------------------------

def test_densifying_reproduces_the_dense_baseline(carriage):
    """Bitwise, against the tree as it stood before the change."""
    baseline = np.load(BASELINE)
    lookup = densify_lookup_slots(carriage['batch'])['val_data']['lookup']

    assert np.array_equal(
        lookup['indicators'].numpy(), baseline['indicators']
    )
    for f in range(len(lookup['slot_values'])):
        assert np.array_equal(
            lookup['slot_values'][f].numpy(), baseline[f'slot_values_{f}']
        ), f'feature {f} slot values'
        for key in ('doses', 'masks'):
            want = baseline.get(f'{key}_{f}')
            if want is None:
                assert lookup[key][f] is None, f'feature {f} {key}'
            else:
                assert np.array_equal(lookup[key][f].numpy(), want), \
                    f'feature {f} {key}'


def test_an_episode_with_no_entries_densifies_to_zeros(carriage):
    """Episode 0 has neither a note nor a dispensation."""
    lookup = densify_lookup_slots(carriage['batch'])['val_data']['lookup']
    for values in lookup['slot_values']:
        assert torch.count_nonzero(values[0]) == 0
    for masks in lookup['masks']:
        if masks is not None:
            assert torch.count_nonzero(masks[0]) == 0


def test_the_densified_shapes_are_the_full_extent(carriage):
    """Ragged widths and a slot axis only where there are slots."""
    lookup = densify_lookup_slots(carriage['batch'])['val_data']['lookup']
    batch_size, max_ts_len = lookup['indicators'].shape[:2]
    text, drug = lookup['slot_values']
    assert text.shape == (batch_size, max_ts_len, TEXT_EMBED_DIM)
    assert drug.shape == (batch_size, max_ts_len, 3, CLINVEC_DIM)
    assert lookup['doses'][1].shape == (batch_size, max_ts_len, 3)


# --- the callers ------------------------------------------------------

def test_densify_is_idempotent(carriage):
    """A batch that has already been densified is returned untouched."""
    once = densify_lookup_slots(carriage['batch'])['val_data']['lookup']
    first = [v.clone() for v in once['slot_values']]
    twice = densify_lookup_slots(carriage['batch'])['val_data']['lookup']
    assert twice['slot_values'] is once['slot_values']
    for before, after in zip(first, twice['slot_values']):
        assert torch.equal(before, after)


def test_move_batch_to_device_densifies(carriage):
    """The real call site: densified after the move, not before."""
    lookup = move_batch_to_device(
        carriage['batch'], torch.device('cpu')
    )['val_data']['lookup']
    assert 'sparse' not in lookup
    assert len(lookup['slot_values']) == 2


def test_generate_record_masks_densifies_defensively(carriage):
    """A caller reading a collated batch directly must not fail.

    Nothing in the pipeline reaches the masks without passing through
    ``move_batch_to_device`` first, but the tests do, and a missing key
    is a worse failure than an idempotent call is a cost.
    """
    torch.manual_seed(0)
    record_masks, _ = generate_record_masks(
        carriage['batch'], feature_sample_rate=0.6, obs_unobs_ratio=2.0,
        subsample_rate=0.5
    )
    widths = [m.shape[-1] for m in record_masks['lookup']['embedded_values']]
    assert widths == [TEXT_EMBED_DIM, CLINVEC_DIM]
