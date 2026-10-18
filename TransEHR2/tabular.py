"""The tabular DeepHit baseline: its feature encoding, its data and its model.

Shared by ``scripts/fit_baseline_deephit.py``, which fits the encoding and
the weights on a fold's train partition, and
``scripts/dump_baseline_deephit_predictions.py``, which applies both to
train, val and test. The encoding is fitted once, saved beside the weights,
and only ever applied after that, so a re-dump reads every split exactly
as training did.

The encoding is the CSC's (plan, "Features"), plus the standardization a
network needs and a Cox model does not:

* **Linear terms** are standardized with the train partition's mean and
  population SD, taken after any median fill.
* **Median-filled terms** (age and the quintiles) have their missing cells
  replaced by the train median of the observed values, and gain a 0/1
  ``<feature>_missing`` indicator. The indicator is not standardized. A
  missing cell in any other linear term is an error: L, A, E and the CCI are
  computed, never empty.
* **Categorical terms** are one-hot over the levels seen on the train
  partition, sorted, plus a missing level. Unlike the CSC's treatment
  dummies no level is dropped: the head's first layer has a bias, and a
  network does not need the reference level to be identified.

As in ``fit_baseline_csc.R``, an indicator or a missing level exists only if
train has a missing value, and a level only if train has it. A val or test
row that needs one train lacks stops the run: the weights behind it were
never trained.

Rows are keyed by ``labels.csv`` row (``idx``), the key of
``baseline_features.csv``. A fold's row arrays are positions in
``extracted_rows.npy``, so a partition's rows are
``extracted_rows[positions]``.
"""

import json
import numpy as np
import os
import pandas as pd
import torch

from dataclasses import asdict, dataclass
from torch import Tensor
from torch.utils.data import (
    BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler
)
from typing import Dict, List, Optional, Sequence

from TransEHR2.cli import TASK
from TransEHR2.modules import DeepHitHead
from TransEHR2.survival import TimeGrid


ENCODING_FILE = 'baseline_encoding.json'
WEIGHTS_FILE = f'finetuned_{TASK}.pt'
MISSING_LEVEL = '<missing>'
SELECTION_FILE = 'extracted_rows.npy'
LABELS_FILE = 'labels.csv'


def read_baseline_features(path: str) -> pd.DataFrame:
    """``baseline_features.csv`` as strings, indexed by ``idx``.

    Read as strings with pandas' NA strings switched off, so that the empty
    cell is the only missing marker, as ``lace.py`` writes it.
    """
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.index = frame['idx'].astype(np.int64)
    return frame


def _numeric(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """The named columns as floats, an empty cell as NaN."""
    return pd.DataFrame({f: pd.to_numeric(frame[f].where(frame[f] != ''))
                         for f in features}, index=frame.index)


def _refuse_missing(numeric: pd.DataFrame, features: Sequence[str],
                    split: str) -> None:
    counts = {f: int(numeric[f].isna().sum()) for f in features}
    counts = {f: n for f, n in counts.items() if n}
    if counts:
        raise ValueError(
            f"{split}: missing values in {counts}. Only a "
            f"MEDIAN_FILL_FEATURES term that train has missing values in may "
            f"be missing.")


@dataclass
class BaselineEncoding:
    """One fold's fitted encoding: all the dump needs to apply it."""

    linear: List[str]
    median_fill: List[str]
    categorical: List[str]
    indicated: List[str]
    medians: Dict[str, float]
    means: Dict[str, float]
    sds: Dict[str, float]
    levels: Dict[str, List[str]]

    @classmethod
    def fit(cls, train: pd.DataFrame, linear: Sequence[str],
            median_fill: Sequence[str],
            categorical: Sequence[str]) -> 'BaselineEncoding':
        """Fit on the train partition's rows of ``baseline_features.csv``."""
        stray = [f for f in median_fill if f not in linear]
        if stray:
            raise ValueError(f"MEDIAN_FILL_FEATURES {stray} are not in "
                             f"LINEAR_FEATURES")
        numeric = _numeric(train, linear)
        _refuse_missing(numeric, [f for f in linear if f not in median_fill],
                        'train')
        medians = {f: float(numeric[f].median()) for f in median_fill}
        filled = numeric.fillna(medians)
        return cls(
            linear=list(linear), median_fill=list(median_fill),
            categorical=list(categorical),
            indicated=[f for f in median_fill if numeric[f].isna().any()],
            medians=medians,
            means={f: float(filled[f].mean()) for f in linear},
            sds={f: float(filled[f].std(ddof=0)) for f in linear},
            levels={f: sorted(set(train[f]) - {''})
                    + ([MISSING_LEVEL] if (train[f] == '').any() else [])
                    for f in categorical})

    def columns(self) -> List[str]:
        """The encoded columns, in the order :meth:`transform` writes them."""
        return (list(self.linear)
                + [f'{f}_missing' for f in self.indicated]
                + [f'{f}={level}' for f in self.categorical
                   for level in self.levels[f]])

    def transform(self, frame: pd.DataFrame, split: str) -> np.ndarray:
        """``frame``'s rows encoded, (n, len(columns())) float32."""
        numeric = _numeric(frame, self.linear)
        _refuse_missing(
            numeric, [f for f in self.linear if f not in self.indicated],
            split)
        missing = numeric[self.indicated].isna().to_numpy(np.float64)
        filled = numeric.fillna(self.medians)
        blocks = [np.stack([(filled[f].to_numpy() - self.means[f])
                            / self.sds[f] for f in self.linear], axis=1),
                  missing]
        for f in self.categorical:
            values = frame[f].where(frame[f] != '', MISSING_LEVEL)
            unseen = sorted(set(values) - set(self.levels[f]))
            if unseen:
                raise ValueError(
                    f"{split}: {f} takes level(s) {unseen} that the train "
                    f"partition never did")
            blocks.append((values.to_numpy()[:, None]
                           == np.array(self.levels[f])[None, :])
                          .astype(np.float64))
        return np.concatenate(blocks, axis=1).astype(np.float32)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f_out:
            json.dump(asdict(self), f_out, indent=2)

    @classmethod
    def load(cls, path: str) -> 'BaselineEncoding':
        with open(path, 'r') as f_in:
            return cls(**json.load(f_in))


def partition_rows(data_dir: str, fold_name: str, split: str) -> np.ndarray:
    """The ``labels.csv`` rows of one fold's partition."""
    selection = np.load(os.path.join(data_dir, SELECTION_FILE))
    positions = np.load(os.path.join(data_dir, fold_name,
                                     f'{fold_name}_{split}_rows.npy'))
    return selection[positions]


class TabularDataset(Dataset):
    """One partition's encoded rows and labels, indexed a batch at a time.

    Items are batches in the shape the finetuning and inference loops read:
    ``idx``, ``x`` and ``targets`` with ``time_to_event`` (minutes, float32
    as TransEHR2 extracts it) and ``event_type``. :func:`make_loader` hands
    it a list of positions per batch.
    """

    def __init__(self, x: np.ndarray, idx: np.ndarray,
                 time_to_event: np.ndarray, event_type: np.ndarray):
        self.x = torch.from_numpy(x)
        self.idx = torch.from_numpy(idx.astype(np.int64))
        self.time_to_event = torch.from_numpy(
            time_to_event.astype(np.float32))
        self.event_type = torch.from_numpy(event_type.astype(np.int64))

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, positions: List[int]) -> Dict:
        at = torch.as_tensor(positions)
        return {'idx': self.idx[at], 'x': self.x[at],
                'targets': {'time_to_event': self.time_to_event[at],
                            'event_type': self.event_type[at]}}


def load_split(data_dir: str, features: pd.DataFrame, labels: pd.DataFrame,
               encoding: BaselineEncoding, fold_name: str,
               split: str) -> TabularDataset:
    """One partition, encoded.

    ``labels`` is ``labels.csv`` in its row order, ``features`` is
    :func:`read_baseline_features`.
    """
    rows = partition_rows(data_dir, fold_name, split)
    absent = np.setdiff1d(rows, features.index.to_numpy())
    if absent.size:
        raise ValueError(
            f"{fold_name} {split}: {absent.size} row(s) have no "
            f"baseline_features.csv entry; rebuild it with lace.py against "
            f"the current extracted_rows.npy")
    return TabularDataset(
        encoding.transform(features.loc[rows], f'{fold_name} {split}'), rows,
        labels['TIME_TO_EVENT'].to_numpy()[rows],
        labels['EVENT_TYPE'].to_numpy()[rows])


def make_loader(dataset: TabularDataset, batch_size: int,
                shuffle_seed: Optional[int] = None) -> DataLoader:
    """Batches of ``dataset``, shuffled by a seeded generator given a seed."""
    sampler = (RandomSampler(dataset, generator=torch.Generator()
                             .manual_seed(shuffle_seed))
               if shuffle_seed is not None else SequentialSampler(dataset))
    return DataLoader(dataset, batch_size=None,
                      sampler=BatchSampler(sampler, batch_size,
                                           drop_last=False))


class TabularDeepHit(torch.nn.Module):
    """TransEHR2's DeepHit head applied to the encoded features directly."""

    def __init__(self, n_features: int, grid: TimeGrid,
                 d_shared: Sequence[int], d_cause: Sequence[int],
                 dropout: float):
        super().__init__()
        self.head = DeepHitHead(
            d_in=n_features, n_causes=grid.n_causes, n_bins=grid.n_bins,
            d_shared=d_shared, d_cause=d_cause, dropout=dropout)

    def forward(self, batch: Dict) -> Tensor:
        return self.head(batch['x'])


def config_grid(cfg: Dict) -> TimeGrid:
    """The experiment config's time grid."""
    return TimeGrid(cfg['TIME_GRID_CUTS_DAYS'], cfg['MODELLED_CAUSES'],
                    cfg['BRIER_INTEGRATION_DAYS'])


def build_model(n_features: int, grid: TimeGrid,
                cfg: Dict) -> TabularDeepHit:
    """A fresh model with the experiment config's head."""
    return TabularDeepHit(
        n_features, grid, d_shared=cfg['DEEPHIT_HEAD_D_SHARED'],
        d_cause=cfg['DEEPHIT_HEAD_D_CAUSE'],
        dropout=cfg['DEEPHIT_HEAD_DROPOUT'])
