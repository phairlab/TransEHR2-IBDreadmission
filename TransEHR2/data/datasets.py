"""One episode as the model sees it (sections 4.2, 4.3, 5, 5.1).

The extractor writes a cohort-wide, compact, integer-coded image of the
data (section 4.4); this module turns one row of it into the tensors the
model already consumes. Everything that is cheap per episode and
expensive per cohort happens here rather than on disk: the one-hot
expansion, the embedding gather, the fold's standardization and the
minutes-to-days conversion.

Readings this module commits to
-------------------------------

* **The series stays right-aligned.** Section 4.2 puts ``t = 0`` -- the
  index time, and the last record -- in the final column, with padding on
  the left, and C1 verified that both encoders are indifferent to the
  alignment: ``TemporalPositionEncoding`` is a function of the timestamp
  value and never of the array index, and the aggregation in ``models.py``
  is a masked max or mean over the time axis. Nothing here normalizes it
  back to left-aligned.
* **Minutes become days for the record timestamps only.** Section 4.2
  converts ``val_times`` and ``event_times`` here, where "magnitudes are
  small and the THP wants a sanely-scaled input". ``time_to_event`` is a
  *target* and stays in the minutes ``labels.csv`` records (section 3):
  its scaling belongs with the prediction head, which is section 8 item 3
  and still open, and rescaling it here would silently change the loss.
* **``(indicator, index)`` is a two-bit code, and both ``-1`` cases
  expand to an all-zero one-hot row** (section 4.3). ``(0, -1)`` is not
  observed, ``(1, -1)`` is observed but out of domain, ``(1, k)`` is
  category *k*. The zero row is a state a one-hot can express and an index
  cannot, and it is what keeps the four ``target.sum(dim=-1) > 0`` guards
  in ``losses.py`` (``:183``, ``:222``, ``:398``, ``:443``) firing with
  that file untouched -- worth real money against C3's regression gate.
  The row is a *representation*, not a discard: the timestep still carries
  its indicator bit, so the indicator loss still trains on it.
* **Standardization reproduces the pre-C1 rule exactly**, including its
  two surprising parts: a feature whose p5 equals its p95 is zeroed
  outright, and the scaling is applied to the whole row rather than to the
  observed timesteps alone, so an unobserved or padded position becomes
  ``-mean / (p95 - p5)`` rather than staying at zero. Both are what
  ``standardize_feats`` did in place at extraction time before section 5
  moved it here, and the subtract-then-divide order is preserved so the
  result is bitwise what it was.
* **Nothing is pooled here** (sections 4.3, 5.1). A drug timestep leaves
  as ``(T, 30, D)`` beside its doses and masks; the encoder input is where
  the dose-weighted mean happens, which is what leaves a gradient on an
  individual slot and therefore on an individual DIN. Pooling earlier
  destroys that attribution irrecoverably.
* **Cast at the gather** (section 5.1). Every lookup family gather is
  ``float32``, whatever the table's own dtype, so the ``torch.cat`` in the
  encoder sees one dtype. A mismatch introduced later surfaces deep in the
  encoder instead of at the boundary where it originated.
* **A lookup table is per *type*, and its own shape is authoritative.**
  Section 4.4 names two files, ``text_embeddings.npy`` and
  ``drug_embeddings.npy``, not one per feature, so features of a type
  share a table and therefore a width. That width is read from the table
  rather than from ``metadata.pkl``: the text entry of
  ``lookup_table_dims`` is ``None`` until C4 builds the table, and section
  5.1 makes the dimension per-feature in any case.
* **``static_data = None`` is a real case, not an accident** (section
  A.3). ``STATIC_FEATS`` is empty, section 4.4 omits the zero-width file,
  and the key is omitted from the item rather than carried as a
  ``(batch, 0)`` tensor -- ``models.py:339`` and ``:521`` both already
  treat its absence correctly.

Why the arrays are opened per process
-------------------------------------

``MixedDataset`` holds ``LazyArray`` path descriptors rather than open
memmaps, and each process opens its own read-only mapping on first touch.
This is the one piece of machinery here that is not asked for by the
blueprint, so the reasoning is recorded in full -- if throughput or memory
behaviour ever looks wrong, start here.

Measured 2026 August 25, C2: **a ``np.memmap`` pickles by value.** A
32 MB array produces a 32 MB pickle, and inside a DataLoader worker
started with ``multiprocessing_context='spawn'`` the array is a ``memmap``
in name only -- ``filename`` is ``None`` and its base is ``bytes``, i.e.
the whole thing was copied into the worker's private memory. Under
``fork`` it stays a genuine file mapping. ``prepare_dataloaders`` asks for
``spawn``, so passing open memmaps to the dataset would give every worker
a private copy of every array it holds; section 4.5 puts the cohort at
~75 GB, and section 2's decision to extract once for the whole cohort
means one dataset now spans all five folds' worth of rows rather than one
partition's. A path descriptor pickles to a few hundred bytes, and the
mapping each worker then opens is backed by the shared page cache, which
is what "workers share one read-only mapping" has to mean to be true.

If this ever needs rethinking, the two alternatives considered and their
costs: switching the context to ``fork`` fixes the mapping and reopens the
fork-after-CUDA-init hang, which is the wrong trade; and building one
dataset per partition instead of one per cohort bounds the damage to a
partition's rows without removing it, at the cost of three memmap sets
over the same files. Nothing in the distributed path depends on this
choice -- ``TextBalancedDistributedSampler`` reads only ``len(dataset)``,
so per-rank sample counts are unchanged, and section 5.1's
distributed-training subsection is untouched.
"""

import numpy as np
import torch

from torch.utils.data import Dataset
from typing import Dict, List, Optional


# Section 4.2 stores integer minutes; the model is handed days.
MINUTES_PER_DAY = 1440.0


class LazyArray:
    """One ``.npy`` file, mapped read-only by whichever process reads it.

    Behaves like the array for the operations ``MixedDataset`` performs on
    one -- ``arr[key]``, ``len(arr)``, ``arr.shape``, ``arr.dtype`` -- and
    carries only its path across a pickle. See this module's docstring for
    why that matters.
    """

    def __init__(self, path: str):
        self.path = path
        self._array = None

    @property
    def array(self) -> np.ndarray:
        """The mapping, opened on first use in this process."""
        if self._array is None:
            self._array = np.load(self.path, mmap_mode='r')
        return self._array

    def __getitem__(self, key):
        return self.array[key]

    def __len__(self) -> int:
        return len(self.array)

    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    def __getstate__(self) -> dict:
        # Deliberately drops the mapping: see the module docstring.
        return {'path': self.path, '_array': None}

    def __setstate__(self, state: dict) -> None:
        self.path = state['path']
        self._array = None


class MixedDataset(Dataset):
    """A cohort of episodes, in section 4.4's on-disk representation.

    Row order is ``labels.csv``'s and spans the whole cohort; a fold is a
    set of row indices into it (section 3), applied by a ``Subset`` around
    this dataset rather than by the dataset itself. Every array may be a
    ``LazyArray``, an ``np.memmap`` or an ordinary ``ndarray``.

    Args:
        val_times: (n, T) int32, minutes before ``INDEX_TIME``, <= 0.
        val_masks: (n, T) float32, 1 on a real timestep.
        val_numeric_indicators: (n, T, n_numeric).
        val_numeric_values: Per-feature (n, T, dim) float32.
        val_categorical_indicators: (n, T, n_categorical).
        val_categorical_values: Per-feature (n, T) int16 category index,
            ``-1`` where the value was not in ``category_map``.
        val_ordinal_indicators: (n, T, n_ordinal).
        val_ordinal_values: Per-feature (n, T) int16 level index.
        categorical_feat_dims: Per-feature number of categories, i.e. the
            one-hot width ``__getitem__`` expands to.
        ordinal_feat_dims: Per-feature number of levels, likewise.
        lookup_indicators: ``{feature type: (n, T, n_feats_of_type)}``.
            Split by type because section 4.4 keeps ``text_*`` and
            ``drug_*`` distinct on disk for the webapp's benefit.
        lookup_csr: Per lookup feature, the CSR arrays ``offsets``,
            ``timesteps`` and ``values``, plus ``doses`` and ``masks``
            for a multi-slot feature.
        lookup_tables: Per lookup feature, the ``(rows, D_f)`` embedding
            table its values index. Features of one type share a table.
        lookup_slot_dims: Per-feature slots per timestep -- 1 for text,
            30 for drugs.
        lookup_feat_types: Per-feature ``'text'`` or ``'drug'``.
        event_indicators: (n, T, n_event).
        event_times: (n, T) int32 minutes.
        event_masks: (n, T) float32.
        time_to_event: (n,) float32 minutes to the event or censoring.
        event_type: (n,) int8 -- 0 censored, 1 readmission, 2 death,
            3 out-migration.
        max_ts_len: T of section 4.4.
        numeric_stats: ``{'means', 'p5', 'p95'}`` for one fold's training
            rows, or None to return the values unscaled. None is for
            inspection and for tests; a model trained on standardized
            input must not be evaluated on raw input.
        static_data: (n, static_total_dim) or None. None is the live case
            (section A.3) and omits the key from the item.
        episode_ids: Row -> ``(PATID, STAY_INDEX)``, for mapping model
            output back to a source episode.
    """

    def __init__(
        self,
        val_times,
        val_masks,
        val_numeric_indicators,
        val_numeric_values: List,
        val_categorical_indicators,
        val_categorical_values: List,
        val_ordinal_indicators,
        val_ordinal_values: List,
        categorical_feat_dims: List[int],
        ordinal_feat_dims: List[int],
        lookup_indicators: Dict[str, object],
        lookup_csr: List[Dict[str, object]],
        lookup_tables: List,
        lookup_slot_dims: List[int],
        lookup_feat_types: List[str],
        event_indicators,
        event_times,
        event_masks,
        time_to_event,
        event_type,
        max_ts_len: int,
        numeric_stats: Optional[Dict[str, np.ndarray]] = None,
        static_data=None,
        episode_ids: Optional[List] = None,
    ):
        self.n_episodes = val_times.shape[0]
        self.max_ts_len = max_ts_len

        self.val_times = val_times
        self.val_masks = val_masks
        self.val_numeric_indicators = val_numeric_indicators
        self.val_numeric_values = val_numeric_values
        self.val_categorical_indicators = val_categorical_indicators
        self.val_categorical_values = val_categorical_values
        self.val_ordinal_indicators = val_ordinal_indicators
        self.val_ordinal_values = val_ordinal_values
        self.categorical_feat_dims = categorical_feat_dims
        self.ordinal_feat_dims = ordinal_feat_dims

        self.lookup_indicators = lookup_indicators
        self.lookup_csr = lookup_csr
        self.lookup_tables = lookup_tables
        self.lookup_slot_dims = lookup_slot_dims
        self.lookup_feat_types = lookup_feat_types
        # The table's own width, not metadata's: text has no declared
        # width until C4 builds the table (section 4.4).
        self.lookup_table_dims = [t.shape[1] for t in lookup_tables]

        self.event_indicators = event_indicators
        self.event_times = event_times
        self.event_masks = event_masks
        self.time_to_event = time_to_event
        self.event_type = event_type
        self.static_data = static_data
        self.episode_ids = episode_ids

        # One fold's statistics, resolved to what the rule needs: the
        # mean, the span, and which features the span degenerates on.
        if numeric_stats is None:
            self._means = self._span = self._degenerate = None
        else:
            self._means = np.asarray(numeric_stats['means'], np.float32)
            p5 = np.asarray(numeric_stats['p5'], np.float32)
            p95 = np.asarray(numeric_stats['p95'], np.float32)
            self._span = p95 - p5
            self._degenerate = self._span == 0

    def __len__(self) -> int:
        return self.n_episodes

    def _days(self, minutes: np.ndarray) -> torch.Tensor:
        """Section 4.2's conversion, at the one place it belongs."""
        return torch.from_numpy(
            np.asarray(minutes, dtype=np.float32) / MINUTES_PER_DAY
        )

    def _numeric_values(self, f: int, idx: int) -> torch.Tensor:
        """One numeric feature's row, scaled by the fold's statistics.

        The pre-C1 rule verbatim: zero the feature outright where p5 and
        p95 coincide, otherwise subtract the mean and divide by the span,
        in that order and over the whole row.
        """
        values = np.array(self.val_numeric_values[f][idx], dtype=np.float32)
        if self._span is None:
            return torch.from_numpy(values)
        if self._degenerate[f]:
            values[:] = 0.0
        else:
            values -= self._means[f]
            values /= self._span[f]
        return torch.from_numpy(values)

    @staticmethod
    def _one_hot(indices: np.ndarray, width: int) -> torch.Tensor:
        """Expand section 4.3's index codes to one-hot rows.

        A negative index -- the ``-1`` sentinel, whether the timestep was
        unobserved or its value was out of domain -- leaves an all-zero
        row, which is exactly the state ``losses.py``'s
        ``target.sum(dim=-1) > 0`` guards test for.
        """
        one_hot = np.zeros((indices.shape[0], width), dtype=np.float32)
        in_domain = indices >= 0
        one_hot[in_domain, indices[in_domain]] = 1.0
        return torch.from_numpy(one_hot)

    def _gather_lookup(self, f: int, idx: int):
        """Gather one lookup feature's embeddings for one episode.

        One code path for both members of the family (section 4.3),
        branching only on slot count: a single-slot feature's values are
        rank 1 on disk and it carries no dose or mask array, so it yields
        ``(T, D)``; a multi-slot feature yields ``(T, S, D)`` with
        ``(T, S)`` doses and masks beside it. Unused slots index the
        table's all-zero pad row and are zeroed by the mask regardless.

        Returns:
            ``(embeddings, doses, masks)``, the last two None for a
            single-slot feature.
        """
        csr = self.lookup_csr[f]
        n_slots = self.lookup_slot_dims[f]
        width = self.lookup_table_dims[f]

        if n_slots == 1:
            embeddings = np.zeros((self.max_ts_len, width), dtype=np.float32)
            doses = masks = None
        else:
            embeddings = np.zeros(
                (self.max_ts_len, n_slots, width), dtype=np.float32
            )
            doses = np.zeros((self.max_ts_len, n_slots), dtype=np.float32)
            masks = np.zeros((self.max_ts_len, n_slots), dtype=np.float32)

        start = int(csr['offsets'][idx])
        end = int(csr['offsets'][idx + 1])
        if end > start:
            timesteps = np.asarray(csr['timesteps'][start:end])
            values = np.asarray(csr['values'][start:end])
            # Section 5.1's cast at the gather: whatever the table's
            # dtype, the model's torch.cat sees float32.
            embeddings[timesteps] = self.lookup_tables[f][values].astype(
                np.float32, copy=False
            )
            if n_slots > 1:
                doses[timesteps] = csr['doses'][start:end]
                masks[timesteps] = csr['masks'][start:end]

        return embeddings, doses, masks

    def __getitem__(self, idx: int) -> Dict:
        """One episode as torch tensors, in ``labels.csv`` row order."""
        item = {'idx': idx}

        item['val_times'] = self._days(self.val_times[idx])
        item['val_masks'] = torch.from_numpy(
            np.array(self.val_masks[idx], dtype=np.float32)
        )

        item['val_numeric_indicators'] = torch.from_numpy(
            np.array(self.val_numeric_indicators[idx], dtype=np.float32)
        )
        item['val_numeric_values'] = [
            self._numeric_values(f, idx)
            for f in range(len(self.val_numeric_values))
        ]

        item['val_categorical_indicators'] = torch.from_numpy(
            np.array(self.val_categorical_indicators[idx], dtype=np.float32)
        )
        item['val_categorical_values'] = [
            self._one_hot(np.asarray(values[idx]), width)
            for values, width in zip(self.val_categorical_values,
                                     self.categorical_feat_dims)
        ]

        item['val_ordinal_indicators'] = torch.from_numpy(
            np.array(self.val_ordinal_indicators[idx], dtype=np.float32)
        )
        item['val_ordinal_values'] = [
            self._one_hot(np.asarray(values[idx]), width)
            for values, width in zip(self.val_ordinal_values,
                                     self.ordinal_feat_dims)
        ]

        for feat_type, indicators in self.lookup_indicators.items():
            item[f'val_{feat_type}_indicators'] = torch.from_numpy(
                np.array(indicators[idx], dtype=np.float32)
            )
        for f, feat_type in enumerate(self.lookup_feat_types):
            embeddings, doses, masks = self._gather_lookup(f, idx)
            if doses is None:
                item.setdefault(f'val_{feat_type}_embeddings', []).append(
                    torch.from_numpy(embeddings)
                )
            else:
                item.setdefault(f'val_{feat_type}_slots', []).append(
                    torch.from_numpy(embeddings)
                )
                item.setdefault(f'val_{feat_type}_doses', []).append(
                    torch.from_numpy(doses)
                )
                item.setdefault(f'val_{feat_type}_masks', []).append(
                    torch.from_numpy(masks)
                )

        item['event_indicators'] = torch.from_numpy(
            np.array(self.event_indicators[idx], dtype=np.float32)
        )
        item['event_times'] = self._days(self.event_times[idx])
        item['event_masks'] = torch.from_numpy(
            np.array(self.event_masks[idx], dtype=np.float32)
        )

        # Minutes, not days: this is a target, and its scaling belongs
        # with the prediction head (section 8 item 3).
        item['time_to_event'] = torch.tensor(
            float(self.time_to_event[idx]), dtype=torch.float32
        )
        # A class label, so torch's label dtype rather than the int8 the
        # array stores.
        item['event_type'] = torch.tensor(
            int(self.event_type[idx]), dtype=torch.long
        )

        if self.static_data is not None:
            item['static_data'] = torch.from_numpy(
                np.array(self.static_data[idx], dtype=np.float32)
            )
        if self.episode_ids is not None:
            item['episode_id'] = self.episode_ids[idx]

        return item
