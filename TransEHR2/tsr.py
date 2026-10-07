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


def _zero_many(batch, leaves, index, perturbations, n_items: int) -> None:
    """Delete each replica's timestep or cell, with one scatter per tensor.

    In place on the leaves, which are clones, and before the graph exists. The indicator goes
    with the value: a feature whose value is zero but whose indicator still says "observed" is
    not a deleted feature, it is an observed zero, and the model reads the two differently.

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

    def rows_and_steps(entries):
        """The (row, timestep) pairs a group of replicas writes to, flattened."""
        replicas = torch.tensor([r for r, _ in entries], device=device)
        steps = torch.tensor([t for _, t in entries], device=device)
        rows = (replicas.unsqueeze(1) * n_items + offsets).reshape(-1)
        return rows, steps.repeat_interleave(n_items)

    whole = [(replica, perturbation[0])
             for replica, perturbation in enumerate(perturbations)
             if perturbation is not None and perturbation[1] is None]
    if whole:
        rows, steps = rows_and_steps(whole)
        for family, tensors in leaves.items():
            for tensor in tensors:
                tensor.data[rows, steps] = 0
            val_data[family]['indicators'][rows, steps] = 0

    by_feature: Dict[int, List[Tuple[int, int]]] = {}
    for replica, perturbation in enumerate(perturbations):
        if perturbation is not None and perturbation[1] is not None:
            by_feature.setdefault(perturbation[1], []).append(
                (replica, perturbation[0]))
    for feature, entries in by_feature.items():
        family, position = index[feature]
        rows, steps = rows_and_steps(entries)
        leaves[family][position].data[rows, steps] = 0
        val_data[family]['indicators'][rows, steps, position] = 0


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
        perturbations: One entry per replica. ``None`` leaves that replica unperturbed,
            ``(timestep, None)`` deletes the whole timestep, ``(timestep, feature)`` one cell.
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
    masked, leaves = _leaf_values(batch, copies=copies)

    if alpha != 1.0:
        for tensors in leaves.values():
            for tensor in tensors:
                tensor.data.mul_(alpha)

    _zero_many(masked, leaves, index, perturbations, n_items)

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

    `_zero` sets a cell's value row and its indicator to zero. A cell where both are already
    zero is therefore bit-identical before and after, so the model returns the same output, the
    same gradients, and ``|R(X) - R(X~)| == 0`` exactly. Those are not passes worth approximating
    away -- they are arithmetic that does not need the model run to do it, and at EHR density
    they are the overwhelming majority of the T*N term.

    Both conditions are read off the batch rather than assumed from one. Extraction does fill
    unobserved numerics with zeros (`preprocessing.ProcessedEpisode._process_numeric` allocates
    zeros and writes `nan_to_num`), but a cell carrying a stray value under a cleared indicator
    would make the skip wrong, and checking costs one reduction per feature.

    The column order is `feature_index`'s, which is the same iteration.
    """
    densify_lookup_slots(batch)
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
    return torch.stack(columns, dim=-1)


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
        gated_value (float, optional): What the feature contribution is held at for timesteps
            below the gate. Defaults to 0, which erases them. Any positive constant orders those
            cells identically -- by their time term alone, with every feature of a given minute
            tied -- so the constant decides only where unmeasured cells sit among measured ones.
            The reference's 0.1 was calibrated against a feature term scaled to [0, 1], which
            this one is not, so it does not carry over.
        chunk (int, optional): How many perturbations share one forward and backward. The pass
            is latency-bound well past 64 at the shapes this model runs at, so this is close to
            a pure speedup; what it costs is memory, since the batch is tiled `chunk` times.
        ig_steps (int, optional): Replace grad x input as TSR's R(.) with integrated gradients
            over this many path points. 0, the default, keeps grad x input. Every pass is
            multiplied by this, and nothing else about the method changes -- TSR rescales
            whatever R(.) returns. Note that the quantity reaching the token pass is then no
            longer a gradient, so `attribution.token_attributions` would have to integrate too
            for stage 2 to stay consistent with stage 1.

    Returns:
        dict: ``scores`` (batch, max_ts_len, n_features) rescaled saliency, ``baseline`` the
        unrescaled R(X), ``delta_time`` the raw time relevance, ``time_contribution`` it
        divided by its row maximum, ``feature_contribution`` (batch, max_ts_len, n_features)
        holding the raw, unscaled feature relevance,
        ``gate`` (batch, max_ts_len), ``lookup_grads`` from the unmasked pass -- the upstream
        gradients `attribution.token_attributions` consumes -- ``index`` from `feature_index`,
        ``occupied`` (batch, max_ts_len, n_features) bool, the cells a pass was worth taking
        for, and ``passes``, the number of perturbations evaluated. ``passes`` counts work, not
        model calls -- ``calls`` is those -- and it is below ``1 + T + T*N`` by however many
        cells were empty.
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
    feature_contribution = torch.where(
        occupied, torch.full_like(baseline, gated_value),
        torch.zeros_like(baseline))
    # Every occupied cell of every timestep any item admits, in one list: the two loops the
    # reference nests are independent of each other given the gate, so they flatten into one
    # chunked run. An empty cell is left at exactly zero rather than the floor -- that is a
    # measurement, not a stand-in for one, because deleting what is not there moves nothing.
    cells = [(int(t), int(c)) for t, c in
             (occupied & gate.unsqueeze(-1)).any(dim=0).nonzero().tolist()]
    if cells:
        delta_feature = _deltas(model, batch, cells, baseline, target, index, chunk,
                                alphas).t()
        passes += len(alphas) * len(cells)
        calls += len(alphas) * math.ceil(len(cells) / chunk)
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
            'occupied': occupied, 'calls': calls,
            'lookup_grads': lookup_grads, 'index': index, 'passes': passes}
