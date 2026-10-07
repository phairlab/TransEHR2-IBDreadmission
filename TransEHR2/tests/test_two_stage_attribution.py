"""The claim `attribution.py` rests on: splitting the backward pass changes no number.

Token attributions are computed in two stages -- TransEHR2's backward gives dL/d(e_pool), and a
second, local backward through the encoder turns that into per-token scores. The whole reason
to do it that way is to keep the encoder out of the training graph, which is only legitimate if
the split is exact.

It is, by the chain rule. These tests check it anyway, because the ways it can stop being exact
are all plumbing: a pooling that drifts between the two paths, a mask dropped on one side, a
whitening transform applied in the wrong direction or not at all. Each would produce plausible
scores and no error.

`test_the_pooling_jacobian_shortcut_is_degenerate_under_cls` is the reason this module exists
at all: it pins the failure of the cheaper approach that avoids the encoder backward entirely.
"""

import pytest
import torch

from TransEHR2.attribution import (SCORERS, text_cell_attributions,
                                   token_attributions)
from TransEHR2.modules import GradientTraceableLLM

# The fixture builds a two-layer XLM-RoBERTa and a tokenizer; importing it is cheaper than a
# second copy, and the copies would drift.
from TransEHR2.tests.test_llm_compat import HIDDEN, _encode, tiny_model  # noqa: F401

STRINGS = ["crohn disease of small intestine", "ulcerative colitis"]
LENGTH = 16


@pytest.fixture
def downstream():
    """Stands in for TransEHR2: anything differentiable that consumes a pooled embedding."""
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(HIDDEN, 4), torch.nn.GELU(),
                               torch.nn.Linear(4, 1))


def _llm(path, pooling):
    llm = GradientTraceableLLM(path, max_length=LENGTH, pooling=pooling,
                               use_gradient_checkpointing=False)
    llm.model.eval()
    return llm


def _end_to_end(llm, encoded, downstream, whitening=None):
    """Scores with the encoder inside the graph, as a reintegrated model would compute them."""
    inputs_embeds = llm.model.get_input_embeddings()(
        encoded['input_ids']).detach().requires_grad_(True)
    pooled = llm.forward_from_embeds(inputs_embeds, encoded['attention_mask'])
    consumed = pooled if whitening is None else pooled @ whitening
    downstream(consumed).sum().backward()
    return (inputs_embeds.grad * inputs_embeds).sum(dim=-1)


def _stage_one_gradient(llm, encoded, downstream, whitening=None):
    """dL/d(the embedding the downstream model consumed) -- what the lookup path can supply.

    Written against a detached vector on purpose: that is the situation at training time, where
    the row comes off disk and the encoder is nowhere in the graph. The whitening is applied
    *before* the leaf, because the table holds whitened rows -- so the gradient this returns is
    dL/dz, and mapping it back through W is `token_attributions`' job.
    """
    with torch.no_grad():
        pooled = llm.forward_from_embeds(
            llm.model.get_input_embeddings()(encoded['input_ids']),
            encoded['attention_mask'])
        stored = pooled if whitening is None else pooled @ whitening
    stored = stored.clone().requires_grad_(True)
    downstream(stored).sum().backward()
    return stored.grad


@pytest.mark.parametrize('pooling', ['mean', 'cls'])
def test_the_split_reproduces_end_to_end_backpropagation(tiny_model, downstream, pooling):
    llm = _llm(tiny_model, pooling)
    encoded = _encode(llm, STRINGS, LENGTH)

    reference = _end_to_end(llm, encoded, downstream)
    upstream = _stage_one_gradient(llm, encoded, downstream)
    scores = token_attributions(llm, encoded['input_ids'],
                                encoded['attention_mask'], upstream)

    assert scores.shape == encoded['input_ids'].shape
    assert torch.allclose(scores, reference, atol=1e-5)


def test_whitening_composes_into_the_upstream_gradient(tiny_model, downstream):
    """A whitened encoder stays attributable: W contributes one matrix product, nothing else."""
    torch.manual_seed(1)
    a = torch.randn(HIDDEN, HIDDEN)
    whitening = a @ a.t() / HIDDEN            # symmetric, as ZCA's transform is
    llm = _llm(tiny_model, 'cls')
    encoded = _encode(llm, STRINGS, LENGTH)

    reference = _end_to_end(llm, encoded, downstream, whitening=whitening)
    upstream = _stage_one_gradient(llm, encoded, downstream, whitening=whitening)
    scores = token_attributions(llm, encoded['input_ids'], encoded['attention_mask'],
                                upstream, whitening_transform=whitening)

    assert torch.allclose(scores, reference, atol=1e-5)


def test_whitening_in_the_wrong_direction_is_not_silently_equivalent(tiny_model, downstream):
    """Guards the one line that could be dropped without failing anything above."""
    torch.manual_seed(1)
    a = torch.randn(HIDDEN, HIDDEN)
    whitening = a @ a.t() / HIDDEN
    llm = _llm(tiny_model, 'cls')
    encoded = _encode(llm, STRINGS, LENGTH)

    upstream = _stage_one_gradient(llm, encoded, downstream, whitening=whitening)
    with_w = token_attributions(llm, encoded['input_ids'], encoded['attention_mask'],
                                upstream, whitening_transform=whitening)
    without_w = token_attributions(llm, encoded['input_ids'], encoded['attention_mask'],
                                   upstream)
    assert not torch.allclose(with_w, without_w, atol=1e-4)


def test_the_pooling_jacobian_shortcut_is_degenerate_under_cls(tiny_model, downstream):
    """Why the encoder backward cannot be skipped.

    For pooling that is a fixed weighted sum, e_pool = sum_i w_i e_i, the Jacobian is w_i I and
    the score of token i is w_i <e_i, dL/d(e_pool)> -- no encoder backward needed. Mean pooling
    gives w_i = 1/n and that is a real, if contextualised, attribution. CLS pooling gives
    w_0 = 1 and w_i = 0, so every token but the first scores exactly zero while the true
    gradients are not zero at all.
    """
    llm = _llm(tiny_model, 'cls')
    encoded = _encode(llm, STRINGS, LENGTH)
    upstream = _stage_one_gradient(llm, encoded, downstream)

    with torch.no_grad():
        final = llm.model(encoded['input_ids'],
                          attention_mask=encoded['attention_mask']).last_hidden_state
    shortcut = torch.zeros(encoded['input_ids'].shape)
    shortcut[:, 0] = (final[:, 0] * upstream).sum(dim=-1)   # w_0 = 1, every other w_i = 0

    assert (shortcut[:, 1:] == 0).all()
    true_scores = token_attributions(llm, encoded['input_ids'],
                                     encoded['attention_mask'], upstream)
    assert true_scores[:, 1:].abs().max() > 1e-6


# ---------------------------------------------------------------------------------------------
# Pluggable scoring, and the join that renders a cell's tokens
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('name', sorted(SCORERS))
def test_every_scorer_returns_one_number_per_token(tiny_model, downstream, name):
    llm = _llm(tiny_model, 'cls')
    encoded = _encode(llm, STRINGS, LENGTH)
    upstream = _stage_one_gradient(llm, encoded, downstream)

    scores = token_attributions(llm, encoded['input_ids'], encoded['attention_mask'],
                                upstream, score=SCORERS[name])
    assert scores.shape == encoded['input_ids'].shape
    assert torch.isfinite(scores).all()


def test_the_default_scorer_is_grad_x_input(tiny_model, downstream):
    """Switching scorers must not be able to change the default silently."""
    llm = _llm(tiny_model, 'cls')
    encoded = _encode(llm, STRINGS, LENGTH)
    upstream = _stage_one_gradient(llm, encoded, downstream)

    assert torch.equal(
        token_attributions(llm, encoded['input_ids'], encoded['attention_mask'], upstream),
        token_attributions(llm, encoded['input_ids'], encoded['attention_mask'], upstream,
                           score=SCORERS['grad_x_input']))


def test_the_unsigned_scorers_are_the_magnitude_of_the_signed_one(tiny_model, downstream):
    llm = _llm(tiny_model, 'cls')
    encoded = _encode(llm, STRINGS, LENGTH)
    upstream = _stage_one_gradient(llm, encoded, downstream)

    signed = token_attributions(llm, encoded['input_ids'], encoded['attention_mask'], upstream)
    unsigned = token_attributions(llm, encoded['input_ids'], encoded['attention_mask'],
                                  upstream, score=SCORERS['abs_grad_x_input'])
    assert torch.allclose(unsigned, signed.abs())
    assert (signed < 0).any(), 'a signed scorer that never goes negative proves nothing'


def test_the_cell_join_scores_the_same_tokens_as_a_direct_call(tiny_model, downstream):
    """`text_cell_attributions` is a gather plus a mask around `token_attributions`."""
    llm = _llm(tiny_model, 'cls')
    encoded = _encode(llm, STRINGS, LENGTH)
    upstream = _stage_one_gradient(llm, encoded, downstream)
    pad_id = llm.tokenizer.pad_token_id

    # The global table, as embed.py writes it: one row per unique string. The cells being
    # explained name rows in it, in an order of their own.
    table = encoded['input_ids'].cpu().numpy()
    rows = torch.tensor([1, 0])

    joined = text_cell_attributions(llm, table, rows, upstream.flip(0), pad_id)
    direct = token_attributions(llm, encoded['input_ids'].flip(0),
                                encoded['attention_mask'].flip(0), upstream.flip(0))
    mask = encoded['attention_mask'].flip(0).to(direct.dtype)
    assert torch.allclose(joined, direct * mask)


def test_padding_scores_zero_because_the_mask_cuts_the_gradient_path(tiny_model, downstream):
    """Not because `text_cell_attributions` zeroes it -- it is already zero by then.

    A masked position is removed from every attention computation, so no path runs from its
    input embedding to the pooled output and its gradient is exactly zero. The multiply in
    `text_cell_attributions` is insurance for a caller that supplies no mask, which the test
    below exercises; this pins the reason the common case needs none.
    """
    llm = _llm(tiny_model, 'cls')
    # Longer than LENGTH so the strings leave padding behind; at LENGTH they fill every
    # position and this would pass without testing anything.
    encoded = _encode(llm, STRINGS, LENGTH * 2)
    upstream = _stage_one_gradient(llm, encoded, downstream)
    pad_id = llm.tokenizer.pad_token_id

    padding = encoded['input_ids'] == pad_id
    assert padding.any(), 'the fixture strings must be shorter than the length for this to test'

    scores = token_attributions(llm, encoded['input_ids'], encoded['attention_mask'], upstream)
    assert (scores[padding] == 0).all()


def test_the_join_returns_zero_at_padding(tiny_model, downstream):
    """The contract a report depends on, asserted on the join's own output."""
    llm = _llm(tiny_model, 'cls')
    encoded = _encode(llm, STRINGS, LENGTH * 2)
    upstream = _stage_one_gradient(llm, encoded, downstream)
    pad_id = llm.tokenizer.pad_token_id
    padding = encoded['input_ids'] == pad_id
    assert padding.any()

    scores = text_cell_attributions(llm, encoded['input_ids'].cpu().numpy(),
                                    torch.tensor([0, 1]), upstream, pad_id)
    assert (scores[padding] == 0).all()
    assert (scores[~padding] != 0).any(), 'all-zero scores would satisfy this vacuously'


def test_the_cell_join_batches_without_changing_the_answer(tiny_model, downstream):
    llm = _llm(tiny_model, 'cls')
    encoded = _encode(llm, STRINGS, LENGTH)
    upstream = _stage_one_gradient(llm, encoded, downstream)
    pad_id = llm.tokenizer.pad_token_id
    table = encoded['input_ids'].cpu().numpy()
    rows = torch.tensor([0, 1, 1, 0])
    grads = upstream[[0, 1, 1, 0]]

    whole = text_cell_attributions(llm, table, rows, grads, pad_id, batch_size=8)
    split = text_cell_attributions(llm, table, rows, grads, pad_id, batch_size=1)
    assert torch.allclose(whole, split)
