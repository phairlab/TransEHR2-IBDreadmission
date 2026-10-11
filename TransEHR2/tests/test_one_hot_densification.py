"""The categorical and ordinal expansion, moved off the worker.

``collate_tensorized`` ships the stored int16 codes and
``densify_one_hot`` builds the one-hot rows inside
``move_batch_to_device``. For this cohort that is 59% of every collated
batch -- 37 categorical features over 1543 levels and 16 ordinal over 73
-- on a pretraining step measured at 60% loader.

The expansion itself is arithmetic; what these pin is the ``-1``
sentinel, which is the part that can go wrong quietly.
"""

import numpy as np
import torch

from TransEHR2.utils import _expand_codes, densify_one_hot


def reference(codes: np.ndarray, width: int) -> np.ndarray:
    """One-hot by construction, the way the worker used to build it."""
    dense = np.zeros((*codes.shape, width), dtype=np.float32)
    rows, cols = np.nonzero(codes >= 0)
    dense[rows, cols, codes[rows, cols]] = 1.0
    return dense


def batch_of(values, widths, family='categorical'):
    """A collated batch carrying nothing but one family's codes."""
    return {'val_data': {family: {'values': values, 'widths': widths}}}


def test_expansion_matches_a_reference_one_hot():
    """Every in-domain code sets its own column and no other."""
    rng = np.random.default_rng(0)
    codes = rng.integers(-1, 6, size=(4, 9)).astype(np.int16)
    expanded = _expand_codes(torch.from_numpy(codes), 6).numpy()
    assert np.array_equal(expanded, reference(codes, 6))


def test_a_negative_code_leaves_an_all_zero_row():
    """The state ``losses.py``'s ``target.sum(dim=-1) > 0`` tests for."""
    codes = torch.full((2, 3), -1, dtype=torch.int16)
    expanded = _expand_codes(codes, 5)
    assert expanded.shape == (2, 3, 5)
    assert expanded.sum() == 0.0


def test_a_negative_code_claims_no_column_at_all():
    """Not column 0, and not the last one either.

    The clamp that keeps ``scatter_`` from rejecting a ``-1`` maps it to
    column 0, so the row is only empty because validity rides in the
    source rather than the index. Swap those and every unobserved cell
    silently becomes class 0 -- the common category, and a plausible
    enough target that the loss would keep falling.
    """
    codes = torch.tensor([[-1, 0, -1]], dtype=torch.int16)
    expanded = _expand_codes(codes, 4)
    assert expanded[0, 0].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert expanded[0, 2].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert expanded[0, 1].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_the_widest_in_domain_code_still_lands():
    """``width - 1`` is a real class, not an off-by-one."""
    codes = torch.tensor([[3]], dtype=torch.int16)
    assert _expand_codes(codes, 4)[0, 0].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_densification_is_idempotent():
    """A second pass must not expand an already-dense batch again.

    ``generate_record_masks`` calls this defensively for callers that
    never went through ``move_batch_to_device``, so both orders have to
    reach the same place.
    """
    batch = batch_of([torch.tensor([[0, -1, 1]], dtype=torch.int16)], [2])
    once = densify_one_hot(batch)['val_data']['categorical']['values'][0]
    twice = densify_one_hot(batch)['val_data']['categorical']['values'][0]
    assert once.shape == (1, 3, 2)
    assert torch.equal(once, twice)


def test_a_batch_built_dense_by_hand_passes_through():
    """Several tests construct batches without a dataset behind them."""
    dense = torch.zeros(1, 3, 2)
    dense[0, 0, 1] = 1.0
    batch = batch_of([dense], [2])
    assert torch.equal(
        densify_one_hot(batch)['val_data']['categorical']['values'][0], dense)


def test_both_families_are_expanded():
    """Ordinal carries codes for the same reason categorical does."""
    batch = {'val_data': {
        'categorical': {'values': [torch.tensor([[1]], dtype=torch.int16)],
                        'widths': [2]},
        'ordinal': {'values': [torch.tensor([[2]], dtype=torch.int16)],
                    'widths': [3]},
    }}
    val = densify_one_hot(batch)['val_data']
    assert val['categorical']['values'][0][0, 0].tolist() == [0.0, 1.0]
    assert val['ordinal']['values'][0][0, 0].tolist() == [0.0, 0.0, 1.0]


def test_an_absent_or_empty_family_is_left_alone():
    """A cohort can declare no categorical feature at all."""
    assert densify_one_hot({'val_data': {}}) == {'val_data': {}}
    empty = batch_of([], [])
    assert densify_one_hot(empty)['val_data']['categorical']['values'] == []
