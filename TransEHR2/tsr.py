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

    1 + T + (occupied cells of the passing timesteps)   TransEHR2 passes   and   0   encoder

At the default open gate that is ``1 + T + (occupied cells)``, which at EHR density is far
below the ``1 + T + T*N`` the shape suggests: a patient-minute carries a handful of the
feature set, and deleting a cell that was never observed provably moves nothing (see
`_occupied`). The reference's 0.55 gate would cut it by a further 45%. The encoder backward
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
* **The feature term is each cell's share of its own timestep.** The reference min-max scales
  the profile within a timestep, which pins the least relevant feature at exactly 0 whether or
  not deleting it moved anything, and leaves a timestep's total mass growing with the number of
  features it happens to carry -- so the time term stops deciding how much a timestep gets.
  Dividing by the timestep's own total makes the product a decomposition: the time term sets the
  mass, the feature term only apportions it, and a timestep's cells sum to its time
  contribution. A cell whose deletion moved nothing still reads zero; a timestep whose cells all
  do reads zeros rather than dividing by one.
* **The time term is divided by its maximum, not min-max scaled.** The bound is the only thing
  the scaling was buying. Min-max decomposes into subtracting the minimum and dividing by the
  range; the division is one constant per episode, so within an episode the subtraction is the
  whole of its effect, and the subtraction is the same manufactured zero. Dividing by the
  maximum keeps the [0, 1] bound, keeps the ordering the raw deltas had, and keeps the origin.
  The gate is untouched either way: a quantile is equivariant under an increasing affine map, so
  the comparison that defines it gives the same answer scaled or not.
* **Empty cells are not perturbed.** The reference explains MNIST, where every pixel has a
  value; an EHR timestep has a value for a few of its features and nothing for the rest. Those
  cells are skipped and scored zero, which is exact rather than an approximation, and it is the
  difference between a tractable method and an intractable one at N in the hundreds.
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

import math
import torch

from torch import Tensor
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from TransEHR2.utils import (densify_lookup_slots, densify_one_hot,
                             resolve_lookup_embeddings)

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


def _with_input_gradients(batch, copies: int = 1) -> Tuple[Dict, Dict[str, List[Tensor]]]:
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
    densify_one_hot(batch)
    val_data = dict(batch['val_data'])
    leaves = {}

    if 'times' in val_data:
        val_data['times'] = _repeat(val_data['times'], copies)
    if 'masks' in val_data:
        # Cloned as well as tiled, because deleting a timestep writes here: `_repeat` hands back
        # the caller's own tensor at one copy, and a write through it would mask the timestep in
        # the batch every later pass reads.
        val_data['masks'] = _repeat(val_data['masks'], copies).clone()

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


def _units(perturbation) -> List[Tuple]:
    """One replica's perturbation as a list of units: None is none, a bare unit is one."""
    if perturbation is None:
        return []
    if isinstance(perturbation, tuple) and isinstance(perturbation[0], int):
        return [perturbation]
    return list(perturbation)


def _zero_many(batch, leaves, index, perturbations, n_items: int) -> None:
    """Delete each replica's units, with one scatter per tensor where it can.

    In place on the leaves, which are clones, and before the graph exists. The indicator goes
    with the value: a feature whose value is zero but whose indicator still says "observed" is
    not a deleted feature, it is an observed zero, and the model reads the two differently.

    **A deleted timestep is masked**, not just emptied: a timestep with every cell zeroed but
    its mask still set is a record of nothing, which no extraction ever writes. The same goes
    for a timestep that the cell and drug deletions in one replica leave empty, which is checked
    after all of them are written rather than assumed from the unit list.

    **A drug is deleted by its ATC code.** The model consumes the dose-weighted pool of a
    timestep's slots, divided by the *count* of filled slots, so clearing the mask of every slot
    holding the code and re-pooling is the record with that drug never dispensed -- the other
    drugs then share the pool as they would in a real record. Zeroing the code's vector instead
    would leave it counted in the divisor.

    Written against the whole chunk rather than one replica at a time because the writes, not
    the arithmetic, were the cost. A replica deleting a whole timestep touches every feature's
    tensor, so a chunk of 300 of them over 149 features issued ~45,000 kernel launches of a few
    elements each -- measured at ~8 us apiece on an H200, which came to more than half the
    runtime. Gathering the replicas that touch the same tensor into one indexed write makes it
    one launch per tensor instead of one per (replica, tensor).

    Args:
        perturbations: One entry per replica, as `saliency_many` takes them.
        n_items: Episodes per replica, so replica ``r`` owns rows ``r*n_items`` onward.
    """
    val_data = batch['val_data']
    device = next(iter(leaves.values()))[0].device
    offsets = torch.arange(n_items, device=device)
    masks = val_data.get('masks')

    def rows_and_steps(entries):
        """The (row, timestep) pairs a group of replicas writes to, flattened."""
        replicas = torch.tensor([r for r, _ in entries], device=device)
        steps = torch.tensor([t for _, t in entries], device=device)
        rows = (replicas.unsqueeze(1) * n_items + offsets).reshape(-1)
        return rows, steps.repeat_interleave(n_items)

    whole, by_feature, by_drug, embeds = [], {}, {}, []
    for replica, perturbation in enumerate(perturbations):
        for unit in _units(perturbation):
            if unit[1] is None:
                whole.append((replica, unit[0]))
            elif len(unit) == 2:
                by_feature.setdefault(unit[1], []).append((replica, unit[0]))
            elif torch.is_tensor(unit[2]):
                embeds.append((replica,) + tuple(unit))
            else:
                by_drug.setdefault(unit[1], []).append((replica, unit[0], int(unit[2])))

    if whole:
        rows, steps = rows_and_steps(whole)
        for family, tensors in leaves.items():
            for tensor in tensors:
                tensor.data[rows, steps] = 0
            val_data[family]['indicators'][rows, steps] = 0
        if masks is not None:
            masks[rows, steps] = 0

    touched_rows, touched_steps = [], []
    for feature, entries in by_feature.items():
        family, position = index[feature]
        rows, steps = rows_and_steps(entries)
        leaves[family][position].data[rows, steps] = 0
        val_data[family]['indicators'][rows, steps, position] = 0
        touched_rows.append(rows)
        touched_steps.append(steps)

    for feature, entries in by_drug.items():
        family, position = index[feature]
        lookup = val_data['lookup']
        rows, steps = rows_and_steps([(r, t) for r, t, _ in entries])
        codes = torch.tensor([c for _, _, c in entries],
                             device=device).repeat_interleave(n_items)
        items = rows % n_items
        held = lookup['slot_indices'][position][items, steps]
        filled = lookup['masks'][position][items, steps]
        kept = filled * (held != codes.unsqueeze(-1))
        # An item that never held the code at this timestep is left alone rather than
        # re-pooled, so it stays bit-identical to the unperturbed pass.
        present = (kept != filled).any(dim=-1)
        rows, steps, items, kept = rows[present], steps[present], items[present], kept[present]
        if not rows.numel():
            continue
        weights = (lookup['doses'][position][items, steps] * kept).unsqueeze(-1)
        pooled = ((lookup['slot_values'][position][items, steps] * weights).sum(dim=-2)
                  / kept.sum(dim=-1).clamp(min=1).unsqueeze(-1))
        leaves[family][position].data[rows, steps] = pooled
        emptied = kept.sum(dim=-1) == 0
        val_data[family]['indicators'][rows[emptied], steps[emptied], position] = 0
        touched_rows.append(rows)
        touched_steps.append(steps)

    for replica, step, feature, vector in embeds:
        family, position = index[feature]
        rows = replica * n_items + offsets
        leaves[family][position].data[rows, step] = vector.to(leaves[family][position].dtype)

    if masks is not None and touched_rows:
        rows, steps = torch.cat(touched_rows), torch.cat(touched_steps)
        alive = torch.zeros(rows.shape, dtype=torch.bool, device=device)
        for family, tensors in leaves.items():
            alive |= (val_data[family]['indicators'][rows, steps] != 0).any(dim=-1)
            for tensor in tensors:
                alive |= (tensor.data[rows, steps] != 0).flatten(1).any(dim=-1)
        masks[rows[~alive], steps[~alive]] = 0


def saliency_many(model, batch, perturbations: Sequence[Optional[Tuple[int, Optional[int]]]],
                  target: Optional[Callable] = None, index: Optional[Sequence] = None,
                  alpha: float = 1.0) -> Tuple[Tensor, List[Tensor]]:
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
        perturbations: One entry per replica: ``None`` leaves that replica unperturbed, and
            otherwise a unit or a list of units, all deleted together. A unit is
            ``(timestep, None)``, the whole timestep; ``(timestep, feature)``, one cell;
            ``(timestep, feature, code)``, the drug with table row ``code`` in a multi-slot
            lookup cell; or ``(timestep, feature, vector)``, a lookup cell's value replaced by
            ``vector`` -- a re-embedded text, which is how a text field is deleted.
        target (callable, optional): Maps the model's output to the scalar to differentiate.
            Defaults to summing it.
        index (sequence, optional): `feature_index(batch)`, required when any perturbation names
            a feature.
        alpha (float, optional): Scale the value tensors to this fraction of themselves before
            the forward pass, which is one point on integrated gradients' path from the
            all-zero baseline. The gradient then comes back evaluated at `alpha * x` while the
            leaf still holds `alpha * x`, so a caller wanting `grad(alpha*x) . x` divides the
            returned scores by alpha. Indicators are left alone: an indicator is a statement
            that a value was observed, not a quantity, and half of one describes no record.

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
    masked, leaves = _with_input_gradients(batch, copies=copies)
    _zero_many(masked, leaves, index, perturbations, n_items)

    # After the deletions, so a re-pooled drug or a substituted text sits on the path too.
    if alpha != 1.0:
        for tensors in leaves.values():
            for tensor in tensors:
                tensor.data.mul_(alpha)

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


def midpoint_alphas(steps: int) -> List[float]:
    """``steps`` points of the Riemann midpoint rule on (0, 1].

    Midpoints rather than endpoints for two reasons: the rule is exact for an integrand linear
    in alpha, where a left or right rule is not, and it never evaluates at alpha = 0, where the
    division that recovers ``grad . x`` would not be defined. ``steps=1`` gives [0.5], which for
    a model linear in its values is already exact.
    """
    return [(k + 0.5) / steps for k in range(steps)]


def _relevance(model, batch, perturbations, target, index,
               alphas) -> Tuple[Tensor, List[Tensor]]:
    """R(.) per perturbation: grad x input, or integrated gradients when `alphas` is longer.

    Integrated gradients from the all-zero baseline is ``x . mean_alpha grad(alpha * x)``, and
    `saliency_many` hands back ``grad(alpha*x) . (alpha*x)``, so each term is divided by its own
    alpha before the terms are averaged. With one alpha of 1.0 that is exactly grad x input,
    which is why both modes run through here rather than through two paths that could disagree.

    The lookup gradients are averaged *without* that division, because stage 2 consumes a
    gradient and multiplies by its own input: `attribution.token_attributions` wants
    ``mean_alpha grad(alpha * x)``, not ``grad . x``. Integrating them alongside R(.) is what
    keeps the token pass consistent with the cell pass when IG is on -- the two would otherwise
    answer different questions about the same cell.

    The alpha loop is innermost so a chunk's maps are averaged while they are in hand: the
    average has to happen before the perturbation is reduced against the baseline, and holding
    every perturbation's map to do it afterwards would cost hundreds of megabytes.
    """
    total, grads = None, None
    for alpha in alphas:
        scores, lookup = saliency_many(model, batch, perturbations, target, index,
                                       alpha=alpha)
        contribution = scores if alpha == 1.0 else scores / alpha
        total = contribution if total is None else total + contribution
        grads = lookup if grads is None else [a + b for a, b in zip(grads, lookup)]
    n = len(alphas)
    return total / n, [g / n for g in grads]


def _deltas(model, batch, perturbations, baseline, target, index, chunk,
            alphas) -> Tensor:
    """``sum |R(X) - R(X~)|`` over the whole saliency map, per perturbation and batch item.

    Reduced inside the chunk loop rather than after it: the maps themselves are
    (perturbations, batch, T, N) and only their summed absolute difference is wanted, so
    carrying them all would cost hundreds of megabytes to throw away.
    """
    out = []
    for start in range(0, len(perturbations), chunk):
        relevance, _ = _relevance(model, batch, perturbations[start:start + chunk],
                                  target, index, alphas)
        out.append((baseline.unsqueeze(0) - relevance).abs().sum(dim=(2, 3)))
    return torch.cat(out, dim=0)


def _occupied(batch) -> Tensor:
    """(batch, max_ts_len, n_features) bool: cells whose deletion could change anything.

    `_zero_many` sets a cell's value row and its indicator to zero. A cell where both are already
    zero is therefore bit-identical before and after, so the model returns the same output, the
    same gradients, and ``|R(X) - R(X~)| == 0`` exactly. Those are not passes worth approximating
    away -- they are arithmetic that does not need the model run to do it, and at EHR density
    they are the overwhelming majority of the T*N term.

    Both conditions are read off the batch rather than assumed from one. Extraction does fill
    unobserved numerics with zeros (`preprocessing.ProcessedEpisode._process_numeric` allocates
    zeros and writes `nan_to_num`), but a cell carrying a stray value under a cleared indicator
    would make the skip wrong, and checking costs one reduction per feature.

    Padding is excluded by the mask as well. Gating keeps a padded cell at zero, but a
    timestep the mask says is not there is not one whose cells can be deleted.

    The column order is `feature_index`'s, which is the same iteration.
    """
    densify_lookup_slots(batch)
    densify_one_hot(batch)
    val_data = batch['val_data']
    columns = []
    for family in FAMILIES:
        if family not in val_data:
            continue
        entry = val_data[family]
        tensors = (resolve_lookup_embeddings(entry) if family == 'lookup'
                   else entry['values'])
        for position, tensor in enumerate(tensors):
            columns.append((entry['indicators'][..., position] != 0)
                           | (tensor != 0).any(dim=-1))
    occupied = torch.stack(columns, dim=-1)
    if 'masks' in val_data:
        occupied &= (val_data['masks'] != 0).unsqueeze(-1)
    return occupied


@torch.no_grad()
def outputs_many(model, batch, perturbations: Sequence, index: Optional[Sequence] = None,
                 chunk: int = 64) -> Tensor:
    """The model's output under each perturbation, forward only.

    The deletion and insertion experiments need f(X') and no gradient. This runs exactly the
    perturbation `saliency_many` differentiates through -- the same leaves, the same deletion
    routine, the same masking of emptied timesteps -- so what TSR measured and what the
    experiments delete cannot drift apart. An insertion is the deletion of everything not
    restored.

    Args:
        perturbations: One entry per replica, as `saliency_many` takes them.
        index (sequence, optional): `feature_index(batch)`; built when omitted.
        chunk (int): Replicas per model call.

    Returns:
        Tensor: (len(perturbations), batch, *output shape).
    """
    index = feature_index(batch) if index is None else index
    val_data = batch['val_data']
    n_items = next(val_data[family]['indicators'].shape[0]
                   for family in FAMILIES if family in val_data)
    out = []
    for start in range(0, len(perturbations), chunk):
        part = list(perturbations[start:start + chunk])
        masked, leaves = _with_input_gradients(batch, copies=len(part))
        _zero_many(masked, leaves, index, part, n_items)
        output = model(masked)
        out.append(output.reshape(len(part), n_items, *output.shape[1:]))
    return torch.cat(out, dim=0)


def crop_to_length(batch) -> Tuple[Dict, int]:
    """Drop the leading columns no episode in the batch occupies.

    Extraction left-pads every episode to one width, so an episode explained alone carries
    columns of padding in front of its first record. They cost a pass each in the time loop
    and contribute nothing: a masked timestep is not read. Cropping before TSR removes them.

    Every (batch, max_ts_len, ...) tensor under ``val_data`` and ``event_data`` is sliced,
    densified first so the lookup family has its dense tensors to slice. The memoized pool is
    dropped and rebuilt from the sliced slots on the next read.

    Returns:
        (dict, int): the cropped batch, and the number of columns dropped -- add it to a
        cropped timestep to get the column it had on the extracted axis.
    """
    densify_lookup_slots(batch)
    densify_one_hot(batch)
    masks = batch['val_data']['masks']
    n_items, width = masks.shape
    used = (masks != 0).any(dim=0).nonzero()
    # One column is kept even for a batch with nothing in it, so no tensor has a zero-length
    # time axis, which the model's guards do not cover.
    start = int(used[0]) if used.numel() else width - 1

    def crop(obj):
        if torch.is_tensor(obj):
            if obj.dim() >= 2 and obj.shape[0] == n_items and obj.shape[1] == width:
                return obj[:, start:]
            return obj
        if isinstance(obj, dict):
            return {k: crop(v) for k, v in obj.items() if k != 'embedded_values'}
        if isinstance(obj, list):
            return [crop(v) for v in obj]
        return obj

    cropped = dict(batch)
    for key in ('val_data', 'event_data'):
        if key in batch:
            cropped[key] = crop(batch[key])
    return cropped, start


def _maxscale(values: Tensor) -> Tensor:
    """Row-wise division by the maximum, bounding to [0, 1] without moving the origin.

    An all-zero row stays zeros. Unlike min-max this preserves the ratios between entries, so a
    timestep that moved the map half as far as the strongest one scores 0.5 rather than whatever
    its rank implies.
    """
    high = values.max(dim=-1, keepdim=True).values
    return torch.where(high > 0, values / high.clamp(min=1e-12),
                       torch.zeros_like(values))


# Below this a timestep's deltas are treated as having moved nothing. One constant for both
# the test and the divisor floor in `_share`; see the note there for why they must agree.
SHARE_EPS = 1e-12


def _share(values: Tensor) -> Tensor:
    """Each timestep's feature relevances as shares of their own total.

    The deltas are sums of absolute differences and so are non-negative, which is what makes a
    share well defined. A timestep none of whose cells moved the map has no total to divide by
    and stays zeros -- it carries no information to apportion, and the time term has already
    said so.

    A timestep with one occupied cell takes 1 there without a special case, which is the
    common shape in this cohort: a dispensation record carries the drug embedding and nothing
    else, so there is no apportioning to do.
    """
    total = values.sum(dim=-1, keepdim=True)
    # The threshold and the divisor floor are the same number deliberately. Testing `total > 0`
    # and then dividing by `clamp(min=EPS)` would silently break the invariant this exists for:
    # a total of 1e-15 passes the test, is divided by 1e-12, and sums to 1e-3 instead of 1.
    # Below EPS a timestep is treated as having moved nothing, which is what it did.
    return torch.where(total > SHARE_EPS, values / total.clamp(min=SHARE_EPS),
                       torch.zeros_like(values))


def tsr_scores(model, batch, target: Optional[Callable] = None,
               quantile: float = 0.0, gated_value: float = 0.0,
               chunk: int = 64, ig_steps: int = 0) -> Dict:
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
        gated_value (float, optional): How much of its time contribution a timestep below the
            gate keeps, its features tied. Defaults to 0, which erases it. The reference's 0.1
            floored a profile it had min-max scaled; under a share the same intent -- an
            unmeasured timestep counts for less -- is a weight on the timestep's mass, so 0.1
            here means a tenth of what it would have had, and 1 means all of it.
        chunk (int, optional): How many perturbations share one forward and backward. The pass
            is latency-bound well past 64 at the shapes this model runs at, so this is close to
            a pure speedup; what it costs is memory, since the batch is tiled `chunk` times.
        ig_steps (int, optional): Replace grad x input as TSR's R(.) with integrated gradients
            over this many path points. 0, the default, keeps grad x input. Every pass is
            multiplied by this, and nothing else about the method changes -- TSR rescales
            whatever R(.) returns. ``lookup_grads`` are then the path-averaged gradients
            ``mean_alpha grad(alpha * x)`` (see `_relevance`), which is the upstream gradient
            `attribution.token_attributions` wants for stage 2 to integrate alongside stage 1.

    Returns:
        dict: ``scores`` (batch, max_ts_len, n_features) rescaled saliency, ``baseline`` the
        unrescaled R(X), ``delta_time`` the raw time relevance, ``time_contribution`` it
        divided by its row maximum, ``feature_contribution`` (batch, max_ts_len, n_features)
        holding each cell's share of its timestep -- summing to 1 over a timestep any of whose
        cells moved the map, and to 0 over one where none did,
        ``gate`` (batch, max_ts_len), ``lookup_grads`` from the unmasked pass -- the upstream
        gradients `attribution.token_attributions` consumes -- ``index`` from `feature_index`,
        ``occupied`` (batch, max_ts_len, n_features) bool, the cells a pass was worth taking
        for, and ``passes``, the number of perturbations evaluated. ``passes`` counts work, not
        model calls -- ``calls`` is those -- and it is below ``1 + T + T*N`` by however many
        cells were empty or alone in their timestep.
        ``drugs`` holds one entry per (item, timestep, drug column, ATC row) held, as equal-
        length tensors: ``item``, ``step``, ``column``, ``code``, ``share`` of the cell, ``score``
        (the cell's TSR score times that share) and ``baseline``, the code's part of the cell's
        R(X). See `_drug_units`.

    A timestep holding a single occupied cell takes no cell pass. Its share is 1 whatever the
    delta, and deleting its only cell is deleting the timestep -- both mask it -- so the raw
    delta is ``delta_time`` exactly, and is recorded as that.
    """
    index = feature_index(batch)
    alphas = midpoint_alphas(ig_steps) if ig_steps else [1.0]
    baseline, lookup_grads = _relevance(model, batch, [None], target, index, alphas)
    baseline = baseline[0]
    n_batch, n_steps, n_features = baseline.shape

    delta_time = _deltas(model, batch, [(t, None) for t in range(n_steps)],
                         baseline, target, index, chunk, alphas).t().contiguous()
    passes = len(alphas) * (1 + n_steps)
    # Model invocations, which is not `passes / chunk`: a call carries up to `chunk`
    # perturbations at ONE alpha, so the alphas multiply the calls as well as the passes.
    calls = len(alphas) * (1 + math.ceil(n_steps / chunk))

    time_contribution = _maxscale(delta_time)
    if quantile <= 0:
        # Not the same as a quantile of 0, which a strict `>` would read as excluding the
        # single least relevant timestep.
        gate = torch.ones_like(time_contribution, dtype=torch.bool)
    else:
        threshold = torch.quantile(time_contribution, quantile, dim=-1, keepdim=True)
        gate = time_contribution > threshold

    occupied = _occupied(batch)
    single = occupied.sum(dim=-1) == 1
    feature_contribution = torch.where(
        occupied, torch.full_like(baseline, gated_value),
        torch.zeros_like(baseline))
    # Every occupied cell of every timestep any item admits, in one list: the two loops the
    # reference nests are independent of each other given the gate, so they flatten into one
    # chunked run. An empty cell is left at exactly zero rather than the floor -- that is a
    # measurement, not a stand-in for one, because deleting what is not there moves nothing.
    # A cell alone in its timestep is left out too; it is filled from `delta_time` below.
    cells = [(int(t), int(c)) for t, c in
             (occupied & (gate & ~single).unsqueeze(-1)).any(dim=0).nonzero().tolist()]
    if cells:
        delta_feature = _deltas(model, batch, cells, baseline, target, index, chunk,
                                alphas).t()
        passes += len(alphas) * len(cells)
        calls += len(alphas) * math.ceil(len(cells) / chunk)
        steps = torch.tensor([t for t, _ in cells], device=baseline.device)
        columns = torch.tensor([c for _, c in cells], device=baseline.device)
        # Raw deltas here; `_share` normalizes them per timestep below. An item that did not
        # admit this timestep keeps the floor even though the pass was taken for some other
        # item that did.
        feature_contribution[:, steps, columns] = torch.where(
            gate[:, steps], delta_feature,
            torch.full_like(delta_feature, gated_value))

    alone = occupied & (gate & single).unsqueeze(-1)
    feature_contribution = torch.where(
        alone, delta_time.unsqueeze(-1).expand_as(feature_contribution),
        feature_contribution)

    feature_contribution = _share(feature_contribution)
    if gated_value:
        # After the share, not before. A constant held across a gated timestep's cells shares
        # evenly, so without this the timestep would take its WHOLE time contribution and the
        # gate would change only the split, never the weight -- the opposite of what the
        # reference's floor was for. Applied here, `gated_value` scales the timestep's mass:
        # 0 erases it, 0.1 leaves a tenth, 1 keeps it with its features tied.
        feature_contribution = torch.where(
            gate.unsqueeze(-1), feature_contribution,
            feature_contribution * gated_value)
    scores = time_contribution.unsqueeze(-1) * feature_contribution
    drugs, drug_passes, drug_calls = _drug_units(
        model, batch, scores, baseline, lookup_grads, occupied & gate.unsqueeze(-1),
        target, index, chunk, alphas)
    passes += drug_passes
    calls += drug_calls
    return {'scores': scores, 'baseline': baseline, 'delta_time': delta_time,
            'time_contribution': time_contribution,
            'feature_contribution': feature_contribution, 'gate': gate,
            'occupied': occupied, 'calls': calls, 'drugs': drugs,
            'lookup_grads': lookup_grads, 'index': index, 'passes': passes}


def _drug_units(model, batch, scores, baseline, lookup_grads, admitted, target, index,
                chunk, alphas) -> Tuple[Dict[str, Tensor], int, int]:
    """Split each drug cell's TSR score among the ATC codes it holds.

    The cell is the pool of a timestep's dispensations, so its score says the drugs mattered
    and not which. This is TSR's own step one level down: delete each code (see `_zero_many`),
    measure ``sum |R(X) - R(X~)|``, and give each code its share of the cell. A cell holding one
    code gives it the whole score without a pass, for the reason a lone cell does in
    `tsr_scores`. Shares sum to 1 over a cell, so code scores sum to the cell's score and sit on
    the same scale as every other unit.

    ``baseline`` is the code's part of the cell's R(X), exact rather than apportioned: the pool
    is linear in each slot, so ``R = g . pool = sum_code g . (sum of its weighted slots) / n``
    with ``g`` the path-averaged gradient at the pool. Its sign is the code's direction, against
    the zero baseline the IG path starts from.

    Codes come from ``slot_indices``, which `MixedDataset` carries. A batch without them -- one
    built by hand -- has no drug units, and this returns empty tensors.
    """
    device = scores.device
    names = ('item', 'step', 'column', 'code')
    rows = {name: [] for name in names}
    rows.update(share=[], score=[], baseline=[])
    lookup = batch['val_data'].get('lookup', {})
    columns = [(c, position) for c, (family, position) in enumerate(index)
               if family == 'lookup' and 'slot_indices' in lookup
               and lookup['doses'][position] is not None]

    units, needs_pass = [], set()
    for column, position in columns:
        codes = lookup['slot_indices'][position]
        filled = lookup['masks'][position] != 0
        for item, step in admitted[:, :, column].nonzero().tolist():
            held = torch.unique(codes[item, step][filled[item, step]]).tolist()
            units.append((item, step, column, position, held))
            if len(held) > 1:
                needs_pass.update((step, column, code) for code in held)

    perturbations = sorted(needs_pass)
    deltas = (_deltas(model, batch, perturbations, baseline, target, index, chunk, alphas)
              if perturbations else None)
    where = {p: i for i, p in enumerate(perturbations)}

    for item, step, column, position, held in units:
        if len(held) > 1:
            raw = torch.stack([deltas[where[(step, column, code)], item] for code in held])
            share = _share(raw.unsqueeze(0))[0]
        else:
            share = torch.ones(len(held), device=device)
        slots = lookup['slot_values'][position][item, step]
        weights = lookup['doses'][position][item, step] * lookup['masks'][position][item, step]
        n_filled = lookup['masks'][position][item, step].sum().clamp(min=1)
        gradient = lookup_grads[position][item, step]
        for k, code in enumerate(held):
            mine = (lookup['slot_indices'][position][item, step] == code).float() * weights
            part = (slots * mine.unsqueeze(-1)).sum(dim=0) / n_filled
            for name, value in zip(names, (item, step, column, code)):
                rows[name].append(value)
            rows['share'].append(share[k])
            rows['score'].append(scores[item, step, column] * share[k])
            rows['baseline'].append((gradient * part).sum())

    out = {name: torch.tensor(rows[name], dtype=torch.long, device=device) for name in names}
    for name in ('share', 'score', 'baseline'):
        out[name] = (torch.stack(rows[name]) if rows[name]
                     else torch.zeros(0, device=device))
    n = len(alphas)
    return out, n * len(perturbations), n * math.ceil(len(perturbations) / chunk)
