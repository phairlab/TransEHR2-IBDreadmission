#!/usr/bin/env python
"""Build the global text and drug lookup tables into ``lookup_tables/``.

Three tables, built **once for the study** rather than once per fold or
once per episode:

    {DATA_DIR}/lookup_tables/text_embeddings.npy   (U, D)     float32
    {DATA_DIR}/lookup_tables/text_tokens.npy       (U, 1024)  int32
    {DATA_DIR}/lookup_tables/drug_embeddings.npy   (V+1, 128) float32

``U`` is the number of unique text strings the cohort contains, and the
row order is ``extracted/text_strings.pkl``'s -- the order
``extract_data.py`` interned them in, which is what makes the extracted
``text_values`` valid indices into the table. ``V`` is the row count of
``ClinVec_atc.csv``, the vocabulary the drug preparation step indexed
against, so ``drug_embeddings.npy`` is that file plus one all-zero pad
row at index ``V``.

Embedding once per unique string rather than once per timestep per fold
is what makes text feasible at all: it is a few million forward passes
for the whole study against a couple of hundred million otherwise.

Usage:
    python embed.py TransEHR2/configs/datasets/RMT23345.yaml
    python embed.py <config> --tables drug     # no LLM is loaded
    python embed.py <config> --tables text --batch-size 32

Design decisions this script commits to
---------------------------------------

* **The tables live in ``{DATA_DIR}/lookup_tables/``, beside
  ``extracted/`` rather than inside it.** ``extract_data.py`` clears every
  ``.npy``, ``.pkl`` and ``.npz`` in its own directory before writing, so
  a text embedding table kept there -- tens of gigabytes and hours of GPU
  time -- would not survive the next extraction. A sibling directory
  settles that by construction rather than by a rule the clearing has to
  remember, and it gives ``drug_embeddings.npy`` -- a pure function of
  ``ClinVec_atc.csv``, cohort-independent, and not invalidated by a
  re-extraction -- a home that no extraction owns.
* **Tokenizing and embedding are one pass, so the attention mask never
  reaches disk.** The mask is needed only to embed each string;
  ``mask == (ids != pad_id)`` recovers it exactly from
  ``text_tokens.npy``, and storing it would cost gigabytes for a read
  that never happens. That derivation is checked against every row of the
  corpus while both are in hand, which is also the only check that the
  pad token never occurs as literal content.
* **A blank string in ``text_strings.pkl`` is refused, not skipped.** The
  extractor already keeps blanks out: a textless timestep carries
  indicator 0 and no sparse entry, so nothing blank is ever interned.
  Dropping one *here* would shift every later row of the table out from
  under the indices the extractor already assigned, so the case is an
  upstream bug to report rather than a filter to apply.
* **The LLM's own tokenizer is used, not a second one.**
  ``GradientTraceableLLM`` resizes its embedding matrix to the vocabulary
  that adding the pad token produced, so ``pad_token_id`` has to be the
  one that resize was done against. It is the resolved id recorded in
  ``metadata.pkl``, and this is where it is established -- extraction
  loads no tokenizer.
* **Checksums are recorded here and verified when the tables are loaded**
  (``TransEHR2.data.manifest``). Both tables are order-sensitive: a row is
  meaningful only as the embedding of the string or drug at that
  position, so a table that copied wrong keeps its shape and its dtype
  and gathers the wrong rows. Nothing but a checksum notices. Rows in
  ``manifest.csv`` are updated, never added: new entries go in by hand, so
  that data which cannot be shared is not registered for distribution by
  a script.
"""

import argparse
import numpy as np
import os
import pandas as pd
import pickle
import sys
import torch
import yaml

from typing import Callable, List, Sequence

from TransEHR2.constants import MAX_TOKEN_LENGTH
from TransEHR2.data.manifest import record_checksum


TABLES_DIRNAME = 'lookup_tables'


def build_drug_table(clinvec_path: str, out_dir: str) -> str:
    """Write ``drug_embeddings.npy`` from the ClinVec ATC vocabulary.

    ``CLINVEC_INDEX`` is the 0-based row of ``ClinVec_atc.csv``
    (``prepare_RMT23345_PIN.R:75-77``), so the table is that file's
    vectors in that order with one all-zero row appended at index ``V``
    -- the pad index unused drug slots are filled with. There is no UNK
    row: every dispensation Stage A kept mapped to a vocabulary
    entry, and an unmapped one is dropped upstream rather than pooled
    into a shared vector here.

    Args:
        clinvec_path: ``CLINVEC_PATH`` from the dataset config.
        out_dir: ``{DATA_DIR}/lookup_tables``.

    Returns:
        The path written.
    """
    table = pd.read_csv(clinvec_path)
    # Column 0 is ``node_id``, the ATC code; the rest are the vector.
    vectors = table.iloc[:, 1:].to_numpy(dtype=np.float32)
    padded = np.zeros(
        (vectors.shape[0] + 1, vectors.shape[1]), dtype=np.float32
    )
    padded[:-1] = vectors

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'drug_embeddings.npy')
    np.save(path, padded)
    print(f"Wrote {path}  {padded.shape} float32 "
          f"(V = {vectors.shape[0]}, pad row {vectors.shape[0]})")
    return path


def load_text_strings(extracted_dir: str) -> List[str]:
    """Read ``text_strings.pkl``, refusing a blank row.

    The list is the table's key order, assigned by ``extract_data.py``;
    row *i* of ``text_embeddings.npy`` must be the embedding of element
    *i*, so nothing here may reorder or drop it.
    """
    path = os.path.join(extracted_dir, 'text_strings.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. It is written by extract_data.py and is "
            f"the row order of text_embeddings.npy; the table cannot be "
            f"built without it."
        )
    with open(path, 'rb') as f:
        strings = pickle.load(f)

    blank = [
        i for i, s in enumerate(strings)
        if s is None or not str(s).strip()
    ]
    if blank:
        raise ValueError(
            f"text_strings.pkl has {len(blank)} blank string(s), at "
            f"row(s) {blank[:5]}{' ...' if len(blank) > 5 else ''}. "
            f"Blanks are kept out of the table, and extraction already "
            f"does that -- a textless timestep carries indicator 0 and no "
            f"sparse entry -- so this is an upstream bug. Skipping the "
            f"row here would shift every later row out from under the "
            f"indices extraction assigned."
        )
    return [str(s) for s in strings]


def build_text_tables(
    strings: Sequence[str],
    tokenizer,
    embed: Callable[[np.ndarray, np.ndarray], np.ndarray],
    out_dir: str,
    embed_dim: int,
    max_length: int = MAX_TOKEN_LENGTH,
    batch_size: int = 64,
) -> List[str]:
    """Write ``text_tokens.npy`` and ``text_embeddings.npy`` in one pass.

    Both arrays are opened as ``.npy`` memory maps and filled a batch at
    a time: at a few million unique strings they run to tens of
    gigabytes, so neither is held whole. The attention mask stays a local
    variable and is checked against ``ids != pad_id`` on every row before
    it is discarded.

    Args:
        strings: The table's key order, from ``text_strings.pkl``.
        tokenizer: A HuggingFace tokenizer with ``pad_token_id`` set.
        embed: ``(input_ids, attention_mask) -> (batch, embed_dim)``
            float32. Injected so the pass can be tested without the LLM.
        out_dir: ``{DATA_DIR}/lookup_tables``.
        embed_dim: ``D``, the LLM's hidden size.
        max_length: Token sequence length; 1024 for this study.
        batch_size: Strings per tokenizer and LLM call.

    Returns:
        The two paths written.
    """
    os.makedirs(out_dir, exist_ok=True)
    tokens_path = os.path.join(out_dir, 'text_tokens.npy')
    embeddings_path = os.path.join(out_dir, 'text_embeddings.npy')
    pad_id = tokenizer.pad_token_id

    n = len(strings)
    tokens = np.lib.format.open_memmap(
        tokens_path, mode='w+', dtype=np.int32, shape=(n, max_length)
    )
    embeddings = np.lib.format.open_memmap(
        embeddings_path, mode='w+', dtype=np.float32, shape=(n, embed_dim)
    )

    try:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            encoded = tokenizer(
                list(strings[start:end]),
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='np',
            )
            ids = np.asarray(encoded['input_ids'])
            mask = np.asarray(encoded['attention_mask'])

            # No mask is stored, so this is the sole guard on deriving
            # it, and it is checked here because this is the one moment
            # both are in hand. It is also the only check that '[PAD]'
            # never appears as literal content: HuggingFace splits added
            # tokens out of input text, so a string containing those five
            # characters would emit pad_id mid-sequence and make the
            # derivation wrong.
            derived = (ids != pad_id).astype(mask.dtype)
            if not np.array_equal(mask, derived):
                row = int(np.flatnonzero((mask != derived).any(axis=1))[0])
                raise AssertionError(
                    f"String {start + row}: the tokenizer's attention "
                    f"mask is not (ids != {pad_id}). No mask is stored -- "
                    f"it is derived from the tokens -- so this string "
                    f"cannot be embedded as it stands. Store a uint8 mask "
                    f"beside the tokens instead, and record why here.\n"
                    f"The string itself is not printed: this runs over "
                    f"real records, and the row index locates it in "
                    f"text_strings.pkl."
                )

            tokens[start:end] = ids.astype(np.int32)
            embeddings[start:end] = embed(ids, mask)

            if start % (batch_size * 200) == 0 or end == n:
                print(f"  {end} / {n} strings", flush=True)
    finally:
        tokens.flush()
        embeddings.flush()
        del tokens, embeddings

    print(f"Wrote {tokens_path}  ({n}, {max_length}) int32")
    print(f"Wrote {embeddings_path}  ({n}, {embed_dim}) float32")
    return [tokens_path, embeddings_path]


def record_pad_token_id(extracted_dir: str, pad_token_id: int) -> None:
    """Add the resolved ``pad_token_id`` to ``metadata.pkl``.

    ``LLM_NAME``, ``TOKENIZER_PAD_TOKEN``, the resolved id and
    ``MAX_TOKEN_LENGTH`` are recorded together so that a tokenizer change
    -- which would invalidate both ``text_tokens.npy`` and the mask
    derivation -- is detectable rather than silent. Extraction writes the
    other three and loads no tokenizer; the id is established here.
    """
    path = os.path.join(extracted_dir, 'metadata.pkl')
    with open(path, 'rb') as f:
        metadata = pickle.load(f)
    metadata['pad_token_id'] = int(pad_token_id)
    with open(path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Recorded pad_token_id = {pad_token_id} in {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the global text and drug lookup tables"
    )
    parser.add_argument(
        'dataset_config',
        type=str,
        help="YAML file specifying dataset parameters (RMT23345.yaml)"
    )
    parser.add_argument(
        '--tables', '-t',
        choices=['all', 'text', 'drug'],
        default='all',
        help="Which tables to build. 'drug' loads no LLM (default: all)"
    )
    parser.add_argument(
        '--data_dir', '-d',
        type=str,
        default=None,
        help="Override the config's DATA_DIR, e.g. to run over a fixture"
    )
    parser.add_argument(
        '--llm-name',
        type=str,
        default=None,
        help="HuggingFace model name; defaults to LLM_NAME in "
             "TransEHR2.constants"
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=64,
        help="Strings per LLM forward pass (default: 64)"
    )
    args = parser.parse_args(argv)

    with open(args.dataset_config, 'r') as f:
        config = yaml.safe_load(f)

    data_dir = args.data_dir or config['DATA_DIR']
    extracted_dir = os.path.join(data_dir, 'extracted')
    out_dir = os.path.join(data_dir, TABLES_DIRNAME)
    written = []

    if args.tables in ('all', 'drug'):
        clinvec_path = config.get('CLINVEC_PATH')
        if not clinvec_path:
            print("CLINVEC_PATH is not in the dataset config; it is the "
                  "vocabulary Stage A indexed against and fixes both the "
                  "pad index and the width of drug_embeddings.npy.",
                  file=sys.stderr)
            return 1
        written.append(build_drug_table(clinvec_path, out_dir))

    if args.tables in ('all', 'text'):
        strings = load_text_strings(extracted_dir)
        print(f"{len(strings)} unique string(s) in {extracted_dir}/"
              f"text_strings.pkl")

        # Imported here so that --tables drug needs neither torch's LLM
        # stack nor a HuggingFace login.
        from TransEHR2.modules import GradientTraceableLLM

        print("Loading the LLM with device_map='auto' ...")
        llm_kwargs = {'model_name': args.llm_name} if args.llm_name else {}
        llm = GradientTraceableLLM(
            use_gradient_checkpointing=False,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            **llm_kwargs,
        )
        llm.eval()
        embed_dim = llm.model.config.hidden_size
        # Inputs go on the device of the model's first parameter, which
        # is where device_map put the embedding layer.
        input_device = next(llm.model.parameters()).device
        print(f"  {llm.model.config._name_or_path}, embed_dim="
              f"{embed_dim}, input_device={input_device}")

        @torch.no_grad()
        def embed(ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
            return llm(
                torch.from_numpy(ids).long().to(input_device),
                trace_grads=False,
                attention_mask=torch.from_numpy(mask).long().to(
                    input_device
                ),
            ).cpu().float().numpy()

        written.extend(build_text_tables(
            strings, llm.tokenizer, embed, out_dir, embed_dim,
            batch_size=args.batch_size
        ))
        record_pad_token_id(extracted_dir, llm.tokenizer.pad_token_id)

    print("Recording checksums in manifest.csv:")
    for path in written:
        checksum = record_checksum(path)
        if checksum is not None:
            print(f"  {os.path.basename(path)}  {checksum}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
