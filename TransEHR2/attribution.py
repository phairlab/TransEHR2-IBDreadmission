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
from typing import Optional


def token_attributions(
    llm,
    token_ids: Tensor,
    attention_mask: Optional[Tensor],
    upstream_grad: Tensor,
    whitening_transform: Optional[Tensor] = None,
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

    Returns:
        Tensor: (N, max_token_length) scores. Padding positions score ~0 under mean pooling,
        which weights them at zero; under CLS pooling they are whatever attention made of them,
        and the caller should mask them out for display.
    """
    inputs_embeds = llm.model.get_input_embeddings()(token_ids).detach().requires_grad_(True)
    pooled = llm.forward_from_embeds(inputs_embeds, attention_mask)

    seed = upstream_grad.to(dtype=pooled.dtype, device=pooled.device)
    if whitening_transform is not None:
        seed = seed @ whitening_transform.t().to(dtype=seed.dtype, device=seed.device)

    grad, = torch.autograd.grad(pooled, inputs_embeds, grad_outputs=seed)
    return (grad * inputs_embeds).sum(dim=-1)
