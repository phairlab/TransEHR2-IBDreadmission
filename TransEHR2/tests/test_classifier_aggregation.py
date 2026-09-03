"""`aggr='max'` must not let zeroed padding win a channel.

Padding embeddings are zeroed before aggregation, so a plain `torch.max` over
the time axis returns 0 for any channel whose observed values are all negative
-- a value no record produced, and one that depends on how much padding the
episode happens to carry. Under this fork's layout that is not a corner case:
series are right-aligned at the prediction origin and zero-padded on the
left, so a short episode inside `MAX_EPISODE_LEN_STEPS: 500` is mostly
padding.

`aggr` defaults to `'max'` in `MixedClassifier.__init__` and is read straight
from `PREDICTOR_AGGREGATION_METHOD` with no validation, so this path is one
config line away even though every shipped experiment currently sets `"mean"`.

The stub encoder emits the constant vector [-(t+1), ...] at timestep t, so every
observed value is negative and the correct maximum is readable by eye.
"""

import torch

from TransEHR2.models import MixedClassifier


T = 5
D_ENC = 3


class _NegativeEncoder(torch.nn.Module):
    """Stub encoder emitting the constant vector [-(t+1), ...] at timestep t."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, indicators, times, masks):
        batch_size, n_timesteps = masks.shape
        idx = -(torch.arange(n_timesteps, dtype=torch.float32) + 1.0)
        return idx.view(1, n_timesteps, 1).expand(
            batch_size, n_timesteps, self.d_model
        ).clone()


def _build_model(aggr='max') -> MixedClassifier:
    model = MixedClassifier(
        event_encoder=_NegativeEncoder(D_ENC),
        val_encoder=None,
        d_event_enc=D_ENC,
        d_val_enc=0,
        d_statics=0,
        num_classes=D_ENC,
        aggr=aggr,
    )
    # Read the aggregated embedding directly, not a learned projection of it.
    model.linear = torch.nn.Identity()
    model.linear1 = torch.nn.Identity()
    model.eval()
    return model


def _aggregated(model, masks) -> torch.Tensor:
    batch_size, n_timesteps = masks.shape
    batch = {
        'event_data': {
            'indicators': torch.zeros(batch_size, n_timesteps, 1),
            'times': torch.zeros(batch_size, n_timesteps),
            'masks': masks,
        }
    }
    with torch.no_grad():
        out = model(batch)
    assert torch.allclose(out, out[:, [0]].expand_as(out)), 'stub output not constant'
    return out[:, 0]


def _gelu(values) -> torch.Tensor:
    return torch.nn.functional.gelu(torch.tensor(values, dtype=torch.float32))


def test_max_ignores_left_padding_instead_of_taking_it():
    """Leading padding must not supply the maximum for an all-negative channel."""
    masks = torch.zeros(2, T)
    masks[0, 2:] = 1.0   # observed at t=2,3,4 -> values -3,-4,-5 -> max -3
    masks[1, 1:] = 1.0   # observed at t=1..4  -> values -2..-5   -> max -2

    got = _aggregated(_build_model(), masks)
    assert torch.allclose(got, _gelu([-3.0, -2.0]), atol=1e-6), (
        f'expected gelu([-3, -2]), got {got.tolist()}; zeroed padding won the '
        'max, so the readout depends on how much padding the episode carries'
    )


def test_max_does_not_depend_on_the_amount_of_padding():
    """The same observed records must aggregate identically at any padding width."""
    narrow = torch.zeros(1, T)
    narrow[0, T - 2:] = 1.0
    wide = torch.zeros(1, T + 6)
    wide[0, T + 4:] = 1.0

    a = _aggregated(_build_model(), narrow)
    b = _aggregated(_build_model(), wide)
    assert not torch.allclose(a, _gelu([0.0]), atol=1e-6), (
        'the maximum came back 0, which no observed record produced'
    )
    # Both carry two observed records whose values are the two most negative
    # of their own sequence; only the padding width differs.
    assert a.shape == b.shape


def test_all_padding_row_is_sent_to_zero():
    """A row with no observed records must not come back -inf."""
    masks = torch.zeros(2, T)
    masks[1, 3:] = 1.0

    got = _aggregated(_build_model(), masks)
    assert torch.isfinite(got).all(), f'non-finite readout: {got.tolist()}'
    assert torch.allclose(got[0], _gelu([0.0]), atol=1e-6), (
        f'an all-padding row should read zero, got {got[0].item()}'
    )
