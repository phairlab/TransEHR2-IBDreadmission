"""Temporal Saliency Rescaling over TransEHR2's value-associated time series.

TSR (Ismail et al., 2020) corrects a failure every gradient-based saliency method has on time
series: the raw map confounds *when* something mattered with *what* mattered. It measures the
two by perturbation -- delete a timestep, delete a cell, and see how far the whole saliency map
moves -- and rescales the map by their product.

Followed from the reference implementation, `getTwoStepRescaling` in
`ayaabdelsalam91/TS-Interpretability-Benchmark` (MNIST Experiments/Scripts/interpret.py), not
from the paper's pseudocode, which disagrees with it in two places:

* The pseudocode's ``Mask feature i at time t: X_i,: = 0`` masks feature i across all time,
  which would make the feature term independent of t and the method a 1 + T + N one. The
  reference masks the single cell, ``newInput[:,c,t]=assignment``, inside the time loop, and so
  recomputes the feature profile at every passing timestep. The nested loop only earns its place
  under the second reading.
* The pseudocode's summation reuses its loop variables as bound variables. Both deltas are
  scalars obtained by summing ``|R(X) - R(X~)|`` over the *entire* saliency map; the subscript
  says which input was deleted, not what was summed. The reference is unambiguous:
  ``np.sum(np.absolute(ActualGrad - perturbed))``.

The reference also min-max scales both contributions before multiplying them and damps gated
timesteps to a flat 0.1. Neither is in the pseudocode, and neither is followed here -- see
Deviations below.

What this costs
---------------

R(.) returns one scalar per (feature, timestep) cell, so every evaluation is a TransEHR2 forward
and backward over the **stored lookup rows**. The text encoder is not in that graph and is never
run by TSR. The cost is

    1 + T + (passing timesteps) x N   TransEHR2 passes   and   0   encoder passes

which at the default open gate is ``1 + T + T*N``, and at the reference's 0.55 gate about
``1 + T + 0.45*T*N``. The encoder backward
(`attribution.token_attributions`) enters once per text cell whose tokens are actually rendered,
after TSR has chosen them; `tsr_scores` returns the lookup gradients that pass needs.

T and N above are one *episode*'s timesteps and features -- the whole patient timeline. A
patient-minute is one timestep within it, which is what ContrastiveBMMB calls a record; the
two senses of the word collide, so this module says episode and timestep.

Deviations from the reference, and why
--------------------------------------

* **The mask value.** The reference assigns ``input[0,0,0]`` -- MNIST's top-left pixel, i.e. its
  background. TransEHR2 has no such corner: absence here is the value zeroed *and* the indicator
  cleared, which is what the model was trained to read as missing. A zeroed value with the
  indicator still set is an observed zero, a different statement.
* **The feature term is not scaled.** The reference min-max scales the feature profile within
  each timestep, which pins that minute's least relevant feature at exactly 0 and leaves
  magnitudes incomparable between minutes. ``delta_feature`` is already non-negative with a
  meaningful origin -- zero means deleting that cell moved nothing -- so the scaling destroys a
  real zero to manufacture one. It is dropped.
* **The time term is divided by its maximum, not min-max scaled.** The bound is the only thing
  the scaling was buying. Min-max decomposes into subtracting the minimum and dividing by the
  range; the division is one constant per episode, so within an episode the subtraction is the
  whole of its effect, and the subtraction is the same manufactured zero. Dividing by the
  maximum keeps the [0, 1] bound, keeps the ordering the raw deltas had, and keeps the origin.
  The gate is untouched either way: a quantile is equivariant under an increasing affine map, so
  the comparison that defines it gives the same answer scaled or not.
* **The gate is open by default and gated timesteps are zeroed, not floored.** The gate exists
  to avoid work, and the reference's 0.55 is a constant tuned on MNIST-shaped data. ``quantile``
  defaults to 0, which measures every cell and so costs ``1 + T + T*N``; the flat floor then has
  nothing to cover for. With the gate closed down, the floor is 0 rather than the reference's
  0.1: any positive constant orders the gated cells identically, by their time term alone, so
  the constant decides only where cells nobody measured sit among cells somebody did. A
  faithfulness curve that deletes the top k% of cells should know that those are tied.
* **Batching.** The reference explains one sample. A masked pass here is shared across the
  batch, which is exact -- no item's output depends on another's inputs -- but means the gate is
  evaluated per item while the passes are taken whenever *any* item needs them. At batch 1, the
  intended use for explanation, the two coincide.
"""

import torch

from torch import Tensor
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from TransEHR2.utils import densify_lookup_slots, resolve_lookup_embeddings

FAMILIES = ('numeric', 'categorical', 'ordinal', 'multilabel', 'lookup')


def feature_index(batch) -> List[Tuple[str, int]]:
    """``[(family, position within that family), ...]``, one entry per saliency column.

    The order columns are built in, so a caller can find the text feature's column without
    guessing. Fixed as `FAMILIES` is ordered, which is also the order the model concatenates in.
    """
    val_data = batch['val_data']
    return [(family, position)
            for family in FAMILIES if family in val_data
            for position in range(val_data[family]['indicators'].shape[-1])]


def _repeat(tensor: Tensor, copies: int) -> Tensor:
    """Tile along the batch axis, replica-major: replica ``r``'s item ``i`` is row ``r*B + i``."""
    if copies == 1:
        return tensor
    return tensor.repeat(copies, *([1] * (tensor.dim() - 1)))


def _leaf_values(batch, copies: int = 1) -> Tuple[Dict, Dict[str, List[Tensor]]]:
    """A shallow copy of ``batch`` whose consumed value tensors are fresh leaves.

    The gradients TSR needs are with respect to what the model *consumes*: the per-feature value
    tensors, and -- for the lookup family -- the pooled embedding, not the slots behind it. The
    pooling is memoized under ``embedded_values``, so resolving it here and replacing the entry
    puts the leaf exactly where the encoder's output would sit if the encoder were in the graph.
    That is also what makes the returned gradient usable as
    `attribution.token_attributions`' `upstream_grad`.

    Args:
        batch: A collated `MixedTensorDataset`.
        copies (int): Tile the batch this many times along its first axis, so one model call
            carries one perturbation per replica. The tiling happens before the leaves are made
            rather than after, because a repeat of a leaf is not itself a leaf and would hand
            back no gradient of its own.
    """
    densify_lookup_slots(batch)
    val_data = dict(batch['val_data'])
    leaves = {}

    for key in ('times', 'masks'):
        if key in val_data:
            val_data[key] = _repeat(val_data[key], copies)

    for family in FAMILIES:
        if family not in val_data:
            continue
        entry = dict(val_data[family])
        if family == 'lookup':
            pooled = resolve_lookup_embeddings(val_data[family])
            tensors = [_repeat(p.detach().clone(), copies).requires_grad_(True)
                       for p in pooled]
            entry['embedded_values'] = tensors
        else:
            tensors = [_repeat(v.detach().clone(), copies).requires_grad_(True)
                       for v in entry['values']]
            entry['values'] = tensors
        entry['indicators'] = _repeat(entry['indicators'].detach().clone(), copies)
        leaves[family] = tensors
        val_data[family] = entry

    masked = dict(batch)
    masked['val_data'] = val_data

    # The event branch is never perturbed, but it is still read, so it has to be as tall as the
    # value branch or the encodings will not concatenate.
    if copies > 1 and 'event_data' in batch:
        event = dict(batch['event_data'])
        for key in ('indicators', 'times', 'masks'):
            if torch.is_tensor(event.get(key)):
                event[key] = _repeat(event[key], copies)
        masked['event_data'] = event
    return masked, leaves


def _zero(batch, leaves, index, timestep: int, feature: Optional[int] = None,
          rows: slice = slice(None)) -> None:
    """Delete a whole timestep, or a single (timestep, feature) cell.

    In place on the leaves, which are clones, and before the graph exists. The indicator goes
    with the value: a feature whose value is zero but whose indicator still says "observed" is
    not a deleted feature, it is an observed zero, and the model reads the two differently.

    ``rows`` restricts the deletion to one replica's block of a tiled batch, so each replica
    carries a different perturbation of the same episode.
    """
    val_data = batch['val_data']
    if feature is None:
        for family, tensors in leaves.items():
            for tensor in tensors:
                tensor.data[rows, timestep] = 0
            val_data[family]['indicators'][rows, timestep] = 0
        return
    family, position = index[feature]
    leaves[family][position].data[rows, timestep] = 0
    val_data[family]['indicators'][rows, timestep, position] = 0


def saliency_many(model, batch, perturbations: Sequence[Optional[Tuple[int, Optional[int]]]],
                  target: Optional[Callable] = None,
                  index: Optional[Sequence] = None) -> Tuple[Tensor, List[Tensor]]:
    """R(X~) for several perturbations in one forward and one backward.

    The batch is tiled once per perturbation and each replica has its own cell deleted, so a
    loop over perturbations becomes a loop over chunks of them. This is exact rather than an
    approximation: no item's output depends on another's inputs, so summing the target over the
    tiled batch and backpropagating once gives every replica the gradient it would have had
    alone. It is worth doing because the pass is latency-bound at batch 1 -- the work was always
    parallel, and only the implementation was serial.

    Args:
        model: Any module taking the batch and returning a tensor.
        batch: A collated `MixedTensorDataset`.
        perturbations: One entry per replica. ``None`` leaves that replica unperturbed,
            ``(timestep, None)`` deletes the whole timestep, ``(timestep, feature)`` one cell.
        target (callable, optional): Maps the model's output to the scalar to differentiate.
            Defaults to summing it.
        index (sequence, optional): `feature_index(batch)`, required when any perturbation names
            a feature.

    Returns:
        (Tensor, list): scores of shape (len(perturbations), batch, max_ts_len, n_features), and
        the per-feature lookup gradients over the tiled batch, each
        (len(perturbations) * batch, max_ts_len, D_f).
    """
    copies = len(perturbations)
    val_data = batch['val_data']
    # Off a family's indicators rather than `val_data['masks']`, which a batch carrying no
    # padding need not have; `feature_index` already takes the same tensor as given.
    n_items = next(val_data[family]['indicators'].shape[0]
                   for family in FAMILIES if family in val_data)
    masked, leaves = _leaf_values(batch, copies=copies)

    for replica, perturbation in enumerate(perturbations):
        if perturbation is None:
            continue
        timestep, feature = perturbation
        _zero(masked, leaves, index, timestep, feature,
              rows=slice(replica * n_items, (replica + 1) * n_items))

    output = model(masked)
    scalar = output.sum() if target is None else target(output)
    scalar.backward()

    columns = []
    for family in FAMILIES:
        if family not in leaves:
            continue
        for tensor in leaves[family]:
            # grad x input, summed over the feature's own width to one scalar per cell.
            columns.append((tensor.grad * tensor).sum(dim=-1))
    scores = torch.stack(columns, dim=-1)
    lookup_grads = [t.grad for t in leaves.get('lookup', [])]
    return scores.detach().reshape(copies, n_items, *scores.shape[1:]), lookup_grads


def saliency(model, batch, target: Optional[Callable] = None,
             timestep: Optional[int] = None, feature: Optional[int] = None,
             index: Optional[Sequence] = None) -> Tuple[Tensor, List[Tensor]]:
    """R(X): gradient x input per (batch, timestep, feature), and the lookup gradients.

    One perturbation, which is `saliency_many` at a single replica. This is the shape the
    baseline pass and the benchmark's per-pass timing want.

    Args:
        model: Any module taking the batch and returning a tensor.
        batch: A collated `MixedTensorDataset`.
        target (callable, optional): See `saliency_many`.
        timestep (int, optional): Delete this timestep before the forward pass.
        feature (int, optional): With `timestep`, delete only that one cell.
        index (sequence, optional): `feature_index(batch)`, required when `feature` is given.

    Returns:
        (Tensor, list): scores of shape (batch, max_ts_len, n_features), and the per-feature
        lookup gradients, each (batch, max_ts_len, D_f).
    """
    perturbation = None if timestep is None else (timestep, feature)
    scores, lookup_grads = saliency_many(model, batch, [perturbation], target, index)
    return scores[0], lookup_grads


def _deltas(model, batch, perturbations, baseline, target, index, chunk) -> Tensor:
    """``sum |R(X) - R(X~)|`` over the whole saliency map, per perturbation and batch item.

    Reduced inside the chunk loop rather than after it: the maps themselves are
    (perturbations, batch, T, N) and only their summed absolute difference is wanted, so
    carrying them all would cost hundreds of megabytes to throw away.
    """
    out = []
    for start in range(0, len(perturbations), chunk):
        scores, _ = saliency_many(model, batch, perturbations[start:start + chunk],
                                  target, index)
        out.append((baseline.unsqueeze(0) - scores).abs().sum(dim=(2, 3)))
    return torch.cat(out, dim=0)


def _maxscale(values: Tensor) -> Tensor:
    """Row-wise division by the maximum, bounding to [0, 1] without moving the origin.

    An all-zero row stays zeros. Unlike min-max this preserves the ratios between entries, so a
    timestep that moved the map half as far as the strongest one scores 0.5 rather than whatever
    its rank implies.
    """
    high = values.max(dim=-1, keepdim=True).values
    return torch.where(high > 0, values / high.clamp(min=1e-12),
                       torch.zeros_like(values))


def tsr_scores(model, batch, target: Optional[Callable] = None,
               quantile: float = 0.0, gated_value: float = 0.0,
               chunk: int = 64) -> Dict:
    """Temporal Saliency Rescaling over a collated batch.

    Args:
        model: Any module taking the batch and returning a tensor.
        batch: A collated `MixedTensorDataset`.
        target (callable, optional): See `saliency`.
        quantile (float, optional): The time-relevance gate, as a quantile of the time
            contributions. It decides what fraction of timesteps get a feature profile at all,
            and so sets both the cost and how much of the map is measured. Defaults to 0, which
            admits every timestep and costs ``1 + T + T*N``; the reference's 0.55 is a constant
            tuned on MNIST-shaped data, and is the setting to pass when the full map is too
            expensive.
        gated_value (float, optional): What the feature contribution is held at for timesteps
            below the gate. Defaults to 0, which erases them. Any positive constant orders those
            cells identically -- by their time term alone, with every feature of a given minute
            tied -- so the constant decides only where unmeasured cells sit among measured ones.
            The reference's 0.1 was calibrated against a feature term scaled to [0, 1], which
            this one is not, so it does not carry over.
        chunk (int, optional): How many perturbations share one forward and backward. The pass
            is latency-bound well past 64 at the shapes this model runs at, so this is close to
            a pure speedup; what it costs is memory, since the batch is tiled `chunk` times.

    Returns:
        dict: ``scores`` (batch, max_ts_len, n_features) rescaled saliency, ``baseline`` the
        unrescaled R(X), ``delta_time`` the raw time relevance, ``time_contribution`` it
        divided by its row maximum, ``feature_contribution`` (batch, max_ts_len, n_features)
        holding the raw, unscaled feature relevance,
        ``gate`` (batch, max_ts_len), ``lookup_grads`` from the unmasked pass -- the upstream
        gradients `attribution.token_attributions` consumes -- ``index`` from `feature_index`,
        and ``passes``, the number of perturbations evaluated. ``passes`` counts work, not model
        calls: `chunk` of them share one forward and backward.
    """
    index = feature_index(batch)
    baseline, lookup_grads = saliency(model, batch, target)
    n_batch, n_steps, n_features = baseline.shape

    delta_time = _deltas(model, batch, [(t, None) for t in range(n_steps)],
                         baseline, target, index, chunk).t().contiguous()
    passes = 1 + n_steps

    time_contribution = _maxscale(delta_time)
    if quantile <= 0:
        # Not the same as a quantile of 0, which a strict `>` would read as excluding the
        # single least relevant timestep.
        gate = torch.ones_like(time_contribution, dtype=torch.bool)
    else:
        threshold = torch.quantile(time_contribution, quantile, dim=-1, keepdim=True)
        gate = time_contribution > threshold

    feature_contribution = torch.full((n_batch, n_steps, n_features), gated_value,
                                      device=baseline.device)
    # Every cell of every timestep any item admits, in one list: the two loops the reference
    # nests are independent of each other given the gate, so they flatten into one chunked run.
    cells = [(t, c) for t in range(n_steps) if bool(gate[:, t].any())
             for c in range(n_features)]
    if cells:
        delta_feature = _deltas(model, batch, cells, baseline, target, index, chunk).t()
        passes += len(cells)
        steps = torch.tensor([t for t, _ in cells], device=baseline.device)
        columns = torch.tensor([c for _, c in cells], device=baseline.device)
        # Unscaled, so the magnitudes are comparable across timesteps and a cell reading zero
        # is one whose deletion moved nothing. An item that did not admit this timestep keeps
        # the floor even though the pass was taken for some other item that did.
        feature_contribution[:, steps, columns] = torch.where(
            gate[:, steps], delta_feature,
            torch.full_like(delta_feature, gated_value))

    scores = time_contribution.unsqueeze(-1) * feature_contribution
    return {'scores': scores, 'baseline': baseline, 'delta_time': delta_time,
            'time_contribution': time_contribution,
            'feature_contribution': feature_contribution, 'gate': gate,
            'lookup_grads': lookup_grads, 'index': index, 'passes': passes}
