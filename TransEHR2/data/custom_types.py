"""Shapes the extractor and the model pass between them.

Design decisions this module commits to
---------------------------------------

* **Text and drugs are one "lookup" family, not two.** They share one
  sparse writer, one sparse reader and one weighted-mean-pool op, text
  being the degenerate case with one slot and weight 1 -- so
  ``TensorDimensions`` carries ``n_lookup_feats`` / ``lookup_slot_dims``
  / ``lookup_table_dims`` rather than a text-only set, and ``EpisodeData``
  carries one ``val_lookup_sparse`` list over the family. On-disk names
  stay ``text_*`` / ``drug_*``, because the webapp finds drugs by
  filename; only the internals are one path.
* **``lookup_pad_indices`` is per-feature data like the other two lists.**
  Unused drug slots are padded with ``V``, the row index of
  ``drug_embeddings``'s all-zero pad row, so that number has to reach the
  sparse builder somehow.
* **A single-slot feature writes no doses and no masks.** Its weight is 1
  by definition, so both arrays would be constant. The entries in
  ``val_lookup_sparse`` hold ``None`` there and the writer emits no file.
* **A text lookup entry carries the string, not an index.** The row index
  into ``text_embeddings.npy`` is assigned later, in
  ``preprocessing._finalize_sparse_lookup``, the only place that sees
  every episode in canonical row order. Workers cannot share the intern
  table, so they hand back the string itself.
* **``static_data`` and its two dimension fields are kept**, even though
  ``STATIC_FEATS`` is empty and ``static_total_dim == 0``. The zero-width
  *file* is not written; the code path stays so that restoring a static
  feature does not reach into ``models.py``.
"""

import numpy as np

from dataclasses import dataclass
from numpy import ndarray
from torch import Tensor
from typing import Dict, List, NamedTuple, Union


EventAssociatedDataEntry = Dict[str, List[List[ndarray]]]
StaticDataEntry = List[ndarray]
ValueAssociatedDataEntry = Dict[str, Dict[str, List[List[ndarray]]]]
TargetDataEntry = Dict[str, ndarray]

# Data types created by preprocessing functions that act on MixedDataset. Used as input to models.
EventAssociatedTensorData = Dict[str, Tensor]
StaticTensorData = Tensor
ValueAssociatedTensorData = Dict[str, Union[Dict[str, Union[Tensor, List[Tensor]]], Tensor]]
TargetTensorData = Dict[str, Tensor]

MixedTensorDataset = Dict[
    str, Union[ValueAssociatedTensorData, EventAssociatedTensorData, StaticTensorData, TargetTensorData]
]


@dataclass
class TensorDimensions:
    """
    Stores pre-computed tensor dimensions for tensorized dataset allocation.

    These dimensions are derived from the dataset configuration and variable
    properties, allowing pre-allocation of output tensors before processing
    begins. Nothing here may be hard-coded: the per-family feature counts
    are counts of ``VALUED_FEATS`` entries by ``type``, read from the two
    config files at extraction time.

    Attributes:
        n_episodes: Number of episodes, i.e. rows of ``labels.csv``
        max_ts_len: Maximum timesteps per episode. One value, not one per
            stream: value- and event-associated data share the same
            right-aligned window, so the old ``max_ts_len_event``
            duplicate said nothing ``max_ts_len_val`` did not.
        n_numeric_feats: Number of numeric features
        n_categorical_feats: Number of categorical features
        n_ordinal_feats: Number of ordinal features
        n_multilabel_feats: Number of multilabel features
        n_lookup_feats: Number of lookup (text + drug) features
        n_event_feats: Number of event features
        numeric_feat_dims: Per-feature vector dimension
        categorical_feat_dims: Per-feature number of categories
        ordinal_feat_dims: Per-feature number of levels
        multilabel_feat_dims: Per-feature number of classes
        lookup_slot_dims: Per-feature slots per timestep -- 1 for text,
            30 for drugs (``size`` in variable_properties.yaml)
        lookup_table_dims: Per-feature width of the embedding table row,
            or None where the table is not built yet. Drugs are 128 from
            ClinVec; text is the LLM's hidden size and is filled in when
            ``embed.py`` builds the table.
        lookup_pad_indices: Per-feature row used to pad unused slots, or
            None for a single-slot feature, which never pads
        static_feat_dims: List of dimensions for each static feature
        static_total_dim: Total dimension of concatenated static features
    """
    n_episodes: int
    max_ts_len: int
    n_numeric_feats: int
    n_categorical_feats: int
    n_ordinal_feats: int
    n_multilabel_feats: int
    n_lookup_feats: int
    n_event_feats: int
    numeric_feat_dims: list
    categorical_feat_dims: list
    ordinal_feat_dims: list
    multilabel_feat_dims: list
    lookup_slot_dims: list
    lookup_table_dims: list
    lookup_pad_indices: list
    static_feat_dims: list
    static_total_dim: int


class EpisodeData(NamedTuple):
    """
    Container for a single processed episode's data before tensor insertion.

    This is an intermediate format returned by worker processes during
    parallel extraction. Data is stored as numpy arrays with minimal
    padding, then inserted into pre-allocated tensors by the main process.

    There is no history/episode distinction, and no left region reserved
    for one: an episode is every record at or before its
    ``INDEX_TIME``, the most recent ``max_episode_len_steps`` of them, and
    the series is right-aligned so that ``t = 0`` -- the index time, and the
    last record -- lands in the final column of every row.

    Attributes:
        row: Row in ``labels.csv``, which is the canonical episode order
            and therefore this episode's output row
        val_len: Number of value-associated timesteps (before padding)
        event_len: Number of event timesteps (before padding)
        val_times: Minutes before ``INDEX_TIME``, shape (val_len,), int32
            and <= 0
        val_numeric_indicators: Array of shape (val_len, n_numeric_feats)
        val_numeric_values: List of arrays, each shape (val_len, feat_dim)
        val_categorical_indicators: (val_len, n_categorical_feats)
        val_categorical_values: List of (val_len,) int16 category indices,
            -1 where the value is not in ``category_map``
        val_ordinal_indicators: (val_len, n_ordinal_feats)
        val_ordinal_values: List of (val_len,) int16 level indices
        val_multilabel_indicators: (val_len, n_multilabel_feats)
        val_multilabel_values: List of (val_len, n_classes) multi-hot rows
        val_lookup_indicators: (val_len, n_lookup_feats)
        val_categorical_misses: (n_categorical_feats,) int64 count of
            out-of-domain values, for the per-feature report
        val_ordinal_misses: (n_ordinal_feats,) int64, likewise
        val_lookup_sparse: Per-feature list of ``(timestep, values, doses,
            masks)`` for the non-empty timesteps only. ``values`` is an
            int32 (n_slots,) array for a drug feature and the raw string
            for a text feature, interned when the CSR arrays are built;
            ``doses`` and ``masks`` are float32 (n_slots,) for a drug
            feature and None for a single-slot one.
        event_times: Minutes before ``INDEX_TIME``, shape (event_len,)
        event_indicators: Array of shape (event_len, n_event_feats)
        static_data: Array of shape (static_total_dim,)
        time_to_event: Minutes from ``INDEX_TIME`` to the event or to
            censoring, from ``labels.csv``
        event_type: 0 censored, 1 unplanned readmission, 2 death,
            3 out-migration
        index_time: The prediction origin as a numpy datetime64, saved for
            timestamp recovery; not model input
    """
    row: int
    val_len: int
    event_len: int
    val_times: 'np.ndarray'
    val_numeric_indicators: 'np.ndarray'
    val_numeric_values: list
    val_categorical_indicators: 'np.ndarray'
    val_categorical_values: list
    val_ordinal_indicators: 'np.ndarray'
    val_ordinal_values: list
    val_multilabel_indicators: 'np.ndarray'
    val_multilabel_values: list
    val_lookup_indicators: 'np.ndarray'
    val_categorical_misses: 'np.ndarray'
    val_ordinal_misses: 'np.ndarray'
    val_lookup_sparse: list
    event_times: 'np.ndarray'
    event_indicators: 'np.ndarray'
    static_data: 'np.ndarray'
    time_to_event: float
    event_type: int
    index_time: 'np.datetime64'
