"""What the discriminator loss counts, and what it refuses to count.

ELECTRA's discriminator judges every position of a sequence, masked or not -- that is the
design, and the reason it learns from more of each batch than the generator does. What it
should not judge is padding. An episode shorter than the batch's longest is padded out to
it; no record was masked there, so the label is 0, and a model that answers 0 wherever the
sequence has ended has learned the collation rather than the data.

Counting those positions also set the denominator by the longest episode in the batch,
which is how a discriminator term one to two orders of magnitude below the generator's came
to look like a discriminator that had outrun its generator.
"""

import torch

from TransEHR2.losses import MaskedDiscriminatorLoss


def _batch(n_timesteps: int, n_features: int = 2):
    """Logits and an all-unmasked indicator mask for one episode."""
    torch.manual_seed(0)
    logits = torch.randn(1, n_timesteps, n_features)
    masks = {'numeric': {'indicators': torch.zeros(1, n_timesteps, n_features)}}
    return {'numeric': logits}, masks


def test_padding_does_not_change_the_loss():
    """Two timesteps of data read the same whether or not six of padding follow them."""
    loss_fn = MaskedDiscriminatorLoss()

    short_preds, short_masks = _batch(2)
    short = loss_fn(short_preds, short_masks, torch.ones(1, 2))

    # The same two timesteps, padded out to eight.
    long_preds, long_masks = _batch(8)
    long_preds['numeric'][:, :2] = short_preds['numeric']
    padding = torch.cat([torch.ones(1, 2), torch.zeros(1, 6)], dim=1)
    long = loss_fn(long_preds, long_masks, padding)

    assert torch.allclose(short, long, atol=1e-6)


def test_padding_carries_no_gradient():
    """The positions are dropped from the sum, not only from the count."""
    preds, masks = _batch(4)
    preds['numeric'].requires_grad_(True)
    padding = torch.cat([torch.ones(1, 2), torch.zeros(1, 2)], dim=1)

    MaskedDiscriminatorLoss()(preds, masks, padding).backward()

    grad = preds['numeric'].grad
    assert torch.any(grad[:, :2] != 0)
    assert torch.all(grad[:, 2:] == 0)


def test_unmasked_positions_are_still_scored():
    """The complement is ELECTRA's task, so an all-unmasked batch is not a free zero."""
    preds, masks = _batch(4)
    loss = MaskedDiscriminatorLoss()(preds, masks, torch.ones(1, 4))
    assert loss > 0


def test_an_all_padding_batch_is_zero_rather_than_a_division_by_zero():
    """Degenerate, but the guard that makes it zero is the one that matters."""
    preds, masks = _batch(4)
    assert MaskedDiscriminatorLoss()(preds, masks, torch.zeros(1, 4)) == 0.0


def test_the_weight_still_scales_the_result():
    """DISC_LOSS_WEIGHT is what relates this term to the generator's."""
    preds, masks = _batch(4)
    padding = torch.ones(1, 4)
    one = MaskedDiscriminatorLoss(weight=1.0)(preds, masks, padding)
    half = MaskedDiscriminatorLoss(weight=0.5)(preds, masks, padding)
    assert torch.allclose(half, one * 0.5)
