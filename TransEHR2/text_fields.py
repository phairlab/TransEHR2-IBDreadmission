"""Text-field units: TSR one level inside a text cell, by masking a field and re-embedding.

A text cell reaches TransEHR2 as one pooled vector of a structured string -- a
``TEXT_SUPERFEATURE`` holding ``DIAGNOSES``, ``PROCEDURES``, ``CLAIMS`` and so on -- so its TSR
score says the record mattered and not which part of it. The finest unit a reader can act on
without word-level attribution is the field: shown the field that carried the score, a clinician
can find the phrases in it themselves.

A field is deleted by dropping it from the string, writing the rest back in exactly the dialect
extraction wrote (`IBDdataprep.superfeature`), and re-embedding. The re-embedded vector replaces
the cell's value inside the TransEHR2 pass (a ``(timestep, column, vector)`` unit, see
`tsr._zero_many`); the indicator stays, because the record is still there. A cell left with no
field at all is deleted outright.

Re-embedding is the expensive step, so only the cells TSR ranks highest are opened. Scoring
then follows TSR's own rule one level down: each field's ``sum |R(X) - R(X~)|``, as a share of
its cell's fields, apportions the cell's score. The shares sum to 1, so field scores sum to the
cell's score and sit on the same scale as every other unit.

Each field also gets a signed value, ``g . (e - e_masked)``: the path-averaged gradient at the
cell's embedding against the embedding's change when the field goes. It is the first-order
estimate of what the field contributes to the cell's R(X), and its sign the field's direction.
Unlike a drug code's part of a pool, it is an estimate, because the encoder is not linear in its
fields; the parts need not sum to the cell's R(X).

Everything here explains one episode at a time. A ``(timestep, column, vector)`` unit writes the
vector into every item of a replica, which is only right when the replica holds one episode.
"""

from __future__ import annotations

import torch

from torch import Tensor
from typing import Callable, Dict, List, Optional, Sequence

from IBDdataprep.superfeature import dump_superfeature, parse_superfeature

from TransEHR2.constants import MAX_TOKEN_LENGTH
from TransEHR2.tsr import _deltas, _share, midpoint_alphas


def drop_fields(text: str, fields: Sequence[str]) -> str:
    """``text`` with the named top-level fields removed, in extraction's dialect.

    A field absent from the string is ignored. Everything else -- the remaining fields, their
    order, their items' order -- is kept as parsed, so dropping nothing gives back the
    dialect-normalized string, which is every stored string (they are all written by
    `dump_superfeature`). Returns ``""`` when nothing is left.
    """
    node = parse_superfeature(text)
    return dump_superfeature({key: value for key, value in node.items()
                              if key not in set(fields)})


def keep_fields(text: str, fields: Sequence[str]) -> str:
    """``text`` with only the named top-level fields: the insertion counterpart of
    `drop_fields`."""
    node = parse_superfeature(text)
    return dump_superfeature({key: value for key, value in node.items() if key in set(fields)})


def present_fields(text: str, fields: Sequence[str]) -> List[str]:
    """Which of ``fields`` the string holds, in the order given."""
    node = parse_superfeature(text)
    return [field for field in fields if field in node]


def make_embedder(llm, device, max_length: int = MAX_TOKEN_LENGTH,
                  batch_size: int = 16) -> Callable[[Sequence[str]], Tensor]:
    """Strings to pooled embeddings, as the lookup table holds them.

    Tokenized as `scripts/embed.py` does -- truncated at the same length, with the same
    tokenizer -- but padded to the longest string in each call rather than to ``max_length``.
    The pooled output reads only attended positions, so the padding length changes nothing but
    the cost; the table was built at 2,048 for storage reasons that do not apply here.

    Args:
        llm: A `modules.GradientTraceableLLM` with the table's encoder and pooling.
        device: Where the encoder runs.
        max_length: Truncation length; must match the table's.
        batch_size: Strings per encoder call.
    """
    def embed(strings: Sequence[str]) -> Tensor:
        out = []
        for start in range(0, len(strings), batch_size):
            encoded = llm.tokenizer(list(strings[start:start + batch_size]),
                                    max_length=max_length, padding='longest',
                                    truncation=True, return_attention_mask=True,
                                    return_tensors='pt')
            with torch.no_grad():
                out.append(llm(encoded['input_ids'].to(device), trace_grads=False,
                               attention_mask=encoded['attention_mask'].to(device)).float())
        return torch.cat(out, dim=0)

    return embed


def text_columns(batch, index) -> List[tuple]:
    """``(column, lookup position)`` for every single-slot lookup feature, i.e. text."""
    lookup = batch['val_data'].get('lookup', {})
    return [(c, position) for c, (family, position) in enumerate(index)
            if family == 'lookup' and lookup['doses'][position] is None]


def text_field_units(model, batch, result: Dict, strings: Sequence[str],
                     embed: Callable[[Sequence[str]], Tensor], fields: Sequence[str],
                     top_cells: int = 10, target: Optional[Callable] = None,
                     chunk: int = 64, ig_steps: int = 0) -> Dict:
    """Score the fields of an episode's top-ranked text cells.

    Args:
        model: The classifier `result` was computed for.
        batch: The one-episode batch `result` was computed on.
        result: `tsr.tsr_scores` for that batch, with the same ``target`` and ``ig_steps``.
        strings: The text table's key order (``text_strings.pkl``); a cell's string is
            ``strings[slot_indices]``.
        embed: Strings to embeddings, as `make_embedder` returns.
        fields: The top-level fields that are units, e.g. ``MIS_CODE``, ``PROVIDER_SVC``,
            ``PROCEDURES``, ``DIAGNOSES``, ``CLAIMS``.
        top_cells: How many text cells to open, highest TSR score first, over all text columns.
        target, chunk, ig_steps: As for `tsr.tsr_scores`, and the same values.

    Returns:
        dict of equal-length entries, one per (cell, field present): ``step``, ``column``,
        ``field`` (list of str), ``share``, ``score`` and ``baseline`` (the signed first-order
        part, see the module docstring); plus ``cells``, the opened ``(step, column)`` pairs in
        rank order, and ``embeddings``, each entry's re-embedded vector, which the deletion and
        insertion experiments reuse rather than re-embed.
    """
    if result['scores'].shape[0] != 1:
        raise ValueError('text fields are scored one episode at a time; the batch holds '
                         f"{result['scores'].shape[0]}")
    index = result['index']
    lookup = batch['val_data']['lookup']
    scores = result['scores'][0]
    alphas = midpoint_alphas(ig_steps) if ig_steps else [1.0]

    candidates = []
    for column, position in text_columns(batch, index):
        for step in result['occupied'][0, :, column].nonzero().flatten().tolist():
            candidates.append((float(scores[step, column]), step, column, position))
    candidates.sort(key=lambda c: -c[0])
    opened = candidates[:top_cells]

    entries, variants = [], []
    for _, step, column, position in opened:
        text = strings[int(lookup['slot_indices'][position][0, step])]
        for field in present_fields(text, fields):
            entries.append((step, column, position, field))
            variants.append(drop_fields(text, [field]))

    out = {'step': [], 'column': [], 'field': [], 'share': [], 'score': [], 'baseline': [],
           'cells': [(step, column) for _, step, column, _ in opened], 'embeddings': []}
    if not entries:
        return out

    nonempty = [i for i, text in enumerate(variants) if text]
    vectors = [None] * len(variants)
    if nonempty:
        embedded = embed([variants[i] for i in nonempty])
        for row, i in enumerate(nonempty):
            vectors[i] = embedded[row]

    device = result['baseline'].device
    perturbations = []
    for (step, column, position, _), vector in zip(entries, vectors):
        perturbations.append((step, column) if vector is None
                             else (step, column, vector.to(device)))
    deltas = _deltas(model, batch, perturbations, result['baseline'], target, index, chunk,
                     alphas)[:, 0]

    for cell_step, cell_column, position in {(s, c, p) for s, c, p, _ in entries}:
        mine = [i for i, (s, c, _, _) in enumerate(entries) if (s, c) == (cell_step, cell_column)]
        share = _share(deltas[mine].unsqueeze(0))[0]
        # A text cell has one slot, so its pooled value is its slot (`pool_lookup_slots`).
        original = lookup['slot_values'][position][0, cell_step].to(device)
        gradient = result['lookup_grads'][position][0, cell_step]
        for k, i in enumerate(mine):
            step, column, _, field = entries[i]
            masked = (torch.zeros_like(original) if vectors[i] is None
                      else vectors[i].to(device))
            out['step'].append(step)
            out['column'].append(column)
            out['field'].append(field)
            out['share'].append(share[k])
            out['score'].append(scores[step, column] * share[k])
            out['baseline'].append((gradient * (original - masked)).sum())
            out['embeddings'].append(vectors[i])

    order = sorted(range(len(out['step'])), key=lambda i: (out['step'][i], out['column'][i],
                                                          list(fields).index(out['field'][i])))
    for key in ('step', 'column', 'field', 'share', 'score', 'baseline', 'embeddings'):
        out[key] = [out[key][i] for i in order]
    for key in ('share', 'score', 'baseline'):
        out[key] = torch.stack(out[key])
    out['step'] = torch.tensor(out['step'], dtype=torch.long)
    out['column'] = torch.tensor(out['column'], dtype=torch.long)
    return out
