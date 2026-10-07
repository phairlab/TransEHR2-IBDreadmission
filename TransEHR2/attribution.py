"""Token-level attribution for the text encoder, split from the model's backward pass.

The text features reach TransEHR2 as rows of a precomputed lookup table, so the encoder is not
in the training graph and a backward pass stops at the stored vector. Reinstating the encoder to
get token-level scores would mean carrying its weights and activations through every step of
every epoch -- the arrangement the Llama-era model was restructured to escape.

It is not necessary. The chain rule splits at the lookup boundary:

    stage 1   TransEHR2 forward/backward over the stored embeddings, which yields
              dL/d(e_pool) for each text slot. Unchanged, and already hooked
              (`GradientTraceableLLM._save_gradients`).
    stage 2   for the records being explained only, re-run the encoder on their token ids and
              seed a local backward with the stage 1 gradient.

The result is identical to backpropagating end-to-end through a reintegrated model -- it is the
same product of Jacobians, evaluated lazily for a handful of records rather than eagerly for
every record in every epoch. `tests/test_two_stage_attribution.py` checks that equality rather
than assuming it.

Why the input embedding layer
-----------------------------

Scores are taken against `inputs_embeds`, the output of the embedding lookup, where a vector is
still one token's own embedding plus its position and attention has mixed nothing in. That is
what licenses rendering a score onto the token a clinician reads.

Attributing to the *final* hidden states instead -- which a shortcut through the pooling
Jacobian alone would force -- measures contextualised positions, not tokens: each vector there
already carries the whole sequence. That shortcut has a second problem under CLS pooling, where
e_pool = e_0 exactly, so every non-CLS token scores zero.
"""

import torch

from torch import Tensor
from typing import Callable, Dict, Optional


# ------------------------------------------------------------------ scoring
#
# A scorer reduces (N, T, D) gradients and (N, T, D) input embeddings to one number per token.
# It is a parameter rather than a constant because the choice is a reporting decision and not a
# correctness one: the gradient is the same object either way, and these differ only in how a
# D-vector is collapsed. `grad_x_input` is the default because it is what TSR's own R(.) uses,
# so a token map and the cell score above it are answering the same question.

def grad_x_input(grad: Tensor, inputs_embeds: Tensor) -> Tensor:
    """Signed: how much this token's own embedding contributed, with direction."""
    return (grad * inputs_embeds).sum(dim=-1)


def grad_l2(grad: Tensor, inputs_embeds: Tensor) -> Tensor:
    """Unsigned sensitivity: how much any change to this token would move the target."""
    return grad.norm(dim=-1)


def grad_l1(grad: Tensor, inputs_embeds: Tensor) -> Tensor:
    """Unsigned, and less dominated by one large component than `grad_l2`."""
    return grad.abs().sum(dim=-1)


def abs_grad_x_input(grad: Tensor, inputs_embeds: Tensor) -> Tensor:
    """`grad_x_input`'s magnitude, for ranking without regard to direction."""
    return (grad * inputs_embeds).sum(dim=-1).abs()


SCORERS: Dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    'grad_x_input': grad_x_input,
    'abs_grad_x_input': abs_grad_x_input,
    'grad_l2': grad_l2,
    'grad_l1': grad_l1,
}


def token_attributions(
    llm,
    token_ids: Tensor,
    attention_mask: Optional[Tensor],
    upstream_grad: Tensor,
    whitening_transform: Optional[Tensor] = None,
    score: Callable[[Tensor, Tensor], Tensor] = grad_x_input,
) -> Tensor:
    """Gradient x input scores, one per token, at the encoder's input embedding layer.

    Args:
        llm: A `GradientTraceableLLM`, frozen and in eval mode.
        token_ids (Tensor): (N, max_token_length) integer ids, as `text_tokens.npy` stores them.
        attention_mask (Tensor, optional): (N, max_token_length) padding mask.
        upstream_grad (Tensor): (N, embed_dim) dL/d(embedding the downstream model consumed),
            as saved by the stage 1 backward pass. One row per sequence in `token_ids`.
        whitening_transform (Tensor, optional): (embed_dim, embed_dim) `W`, when the consumed
            embedding was whitened as `z = (e_pool - mu) W`. The gradient is mapped back through
            it, `dL/d(e_pool) = (dL/dz) W^T`. ZCA's `W` is symmetric, so this is also `(dL/dz) W`
            -- equivalently the token vectors could be whitened instead, for identical scores.

        score (callable, optional): Reduces the (N, T, D) gradient and the (N, T, D) input
            embeddings to one number per token. One of `SCORERS`; defaults to `grad_x_input`.
            Only the reduction is a choice -- every scorer sees the same gradient, computed the
            same way, so switching one for another does not re-run the backward differently.

    Returns:
        Tensor: (N, max_token_length) scores. Padding positions score exactly zero whenever
        `attention_mask` marks them: a masked position is removed from every attention
        computation, so no path runs from its input embedding to the pooled output. Pass no
        mask and they score whatever attention made of them, under either pooling --
        `text_cell_attributions` always derives one.
    """
    inputs_embeds = llm.model.get_input_embeddings()(token_ids).detach().requires_grad_(True)
    pooled = llm.forward_from_embeds(inputs_embeds, attention_mask)

    seed = upstream_grad.to(dtype=pooled.dtype, device=pooled.device)
    if whitening_transform is not None:
        seed = seed @ whitening_transform.t().to(dtype=seed.dtype, device=seed.device)

    grad, = torch.autograd.grad(pooled, inputs_embeds, grad_outputs=seed)
    return score(grad, inputs_embeds)


def text_cell_attributions(
    llm,
    tokens_table,
    rows: Tensor,
    upstream_grad: Tensor,
    pad_token_id: int,
    whitening_transform: Optional[Tensor] = None,
    score: Callable[[Tensor, Tensor], Tensor] = grad_x_input,
    batch_size: int = 8,
) -> Tensor:
    """Per-token scores for a set of text cells, padding masked out.

    This is the join between the two stages. A text cell is one lookup slot holding a row index
    into the global token table, so its tokens are `tokens_table[row]` and its upstream gradient
    is `tsr_scores(...)['lookup_grads'][feature][episode, timestep]`. Gathering the rows here
    rather than in the caller keeps the padding rule in one place.

    The attention mask is derived as `ids != pad_token_id`. That is the same derivation
    `embed.py` makes when it builds the table, and it checks it there against the tokenizer's
    own mask for every string -- so it is verified once over the corpus rather than trusted
    here.

    Args:
        llm: A `GradientTraceableLLM`, frozen and in eval mode.
        tokens_table: `text_tokens.npy`, indexable by row. A memory map is fine; only the
            requested rows are read.
        rows (Tensor): (N,) integer row indices, one per cell being explained.
        upstream_grad (Tensor): (N, embed_dim) stage 1 gradients for those cells, same order.
        pad_token_id (int): The tokenizer's pad id, as `embed.py` recorded it.
        whitening_transform (Tensor, optional): See `token_attributions`.
        score (callable, optional): See `token_attributions`.
        batch_size (int, optional): Cells per encoder backward. The encoder runs for these cells
            only, so this trades memory against call count and nothing else.

    Returns:
        Tensor: (N, max_token_length) scores, zero at padding positions. The derived mask
        already makes them zero -- a masked position has no gradient path to the pooled output
        -- so the multiply states the guarantee locally rather than inheriting it from the
        encoder's attention implementation. Zero rather than NaN so rows sum and sort without
        special handling, which does mean a real token scoring exactly zero is
        indistinguishable from padding: a caller writing tokens out should write the ids too
        rather than inferring the length from the scores.
    """
    device = next(llm.model.parameters()).device
    out = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        ids = torch.as_tensor(
            tokens_table[chunk.cpu().numpy()], dtype=torch.long, device=device)
        mask = (ids != pad_token_id).long()
        scores = token_attributions(
            llm, ids, mask, upstream_grad[start:start + batch_size],
            whitening_transform=whitening_transform, score=score)
        out.append(scores * mask.to(scores.dtype))
    return torch.cat(out, dim=0) if out else upstream_grad.new_zeros((0, 0))
