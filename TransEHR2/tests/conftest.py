"""A miniature Stage B root, so the unit tests own their inputs.

The end-to-end test in ``test_extract_data.py`` runs against
IBDdataprep's real fixture and the live config, which is what pins
section 4.4's 94 / 37 / 16. Everything else is checked here against a
six-feature cohort the test writes itself, because the claims below need
values the real fixture does not contain -- an out-of-domain category, a
declared level that collides with pandas' NA strings, a drug dispensed at
a timestamp nothing else names.
"""

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from pathlib import Path

from TransEHR2.data.preprocessing import collate_tensorized
from TransEHR2.utils import move_batch_to_device


def collate_for_model(items: list) -> dict:
    """A batch in the form the model sees it.

    The lookup family crosses the worker boundary sparsely (section
    5.1), and ``move_batch_to_device`` is what rebuilds its dense
    tensors -- so a batch that has not been through the move is not the
    batch any model or loss is ever handed. The device is the CPU here,
    which is the only difference from the training loop.
    """
    return move_batch_to_device(collate_tensorized(items),
                                torch.device('cpu'))


# One feature of every type, kept small enough to read an array by eye.
# ORD's levels are numeric-looking strings, like the real BLDUA; UB's
# level 0 is 'None', like the real UBAC.
VARIABLE_PROPERTIES = {
    'NUM': {'type': 'numeric', 'size': 1, 'category_map': {}},
    'CAT': {'type': 'categorical', 'size': 2,
            'category_map': {0: 'L', 1: 'U'}},
    'ORD': {'type': 'ordinal', 'size': 3,
            'category_map': {0: '0', 1: '1-24', 2: '25-50'}},
    'UB': {'type': 'ordinal', 'size': 2,
           'category_map': {0: 'None', 1: 'Few'}},
    'TXT': {'type': 'text', 'size': 1, 'category_map': {}},
    'DRG': {'type': 'drug', 'size': 3, 'category_map': {}},
    'EVT': {'type': 'categorical', 'size': 2,
            'category_map': {0: 0, 1: 1}},
}

CONFIG = {
    'VALUED_FEATS': ['NUM', 'CAT', 'ORD', 'UB'],
    'EVENT_FEATS': ['EVT'],
    'TEXT_FEATS': ['TXT'],
    'DRUG_FEATS': ['DRG'],
    'STATIC_FEATS': [],
    'MAX_EPISODE_LEN_STEPS': 4,
    'MIN_EPISODE_LEN_STEPS': 1,
}

TIMESERIES_COLUMNS = ['TIMESTAMP', 'NUM', 'CAT', 'ORD', 'UB', 'TXT', 'EVT']
STAYS_COLUMNS = ['PATID', 'STAY_INDEX', 'DATA_SOURCE', 'ADMITDATE',
                 'DISDATE', 'INDEX_TIME', 'ADMITCAT', 'ABSTRACT_TYPE',
                 'IBD_RELATED', 'DEATH_DT', 'OBS_END_DT']
DRUGS_COLUMNS = ['PATID', 'TIMESTAMP', 'SLOT', 'DRUG_DIN', 'CLINVEC_INDEX',
                 'CLINVEC_ATC_CODE', 'CLINVEC_ATC_NAME', 'REL_DAILY_QTY',
                 'MAINTENANCE', 'STRD_FLARE_DOSE']

# Four vocabulary rows, so the pad index is 4.
CLINVEC_ROWS = 4
CLINVEC_DIM = 2


def write_csv(path: Path, rows: list, columns: list) -> None:
    """Write like Stage B does: the empty string is the only NA."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, index=False, na_rep='', lineterminator='\n')


class MiniRoot:
    """Builds a Stage B root, labels.csv and a dataset config."""

    def __init__(self, tmp_path: Path):
        self.data_dir = tmp_path / 'data'
        self.root = self.data_dir / 'root'
        self.labels_rows = []
        self.config = dict(CONFIG)
        self.var_properties = {
            k: dict(v) for k, v in VARIABLE_PROPERTIES.items()
        }
        self.tmp_path = tmp_path

    def add_patient(self, patid, timeseries, stays, drugs=()):
        """Add one patient. ``timeseries`` rows are TIMESERIES_COLUMNS."""
        write_csv(self.root / str(patid) / 'timeseries.csv',
                  list(timeseries), TIMESERIES_COLUMNS)
        stay_rows = []
        for stay_index, (source, admit, index_time) in enumerate(stays):
            stay_rows.append([patid, stay_index, source, admit, '',
                              index_time, '', '', '', '',
                              '2020-04-01T05:59:00Z'])
            self.labels_rows.append(
                [patid, stay_index, source, 1000.0, 0]
            )
        write_csv(self.root / str(patid) / 'stays.csv', stay_rows,
                  STAYS_COLUMNS)
        if drugs:
            rows = [[patid, stamp, slot, 12345, index, 'A01', 'name',
                     qty, '', ''] for stamp, slot, index, qty in drugs]
            write_csv(self.root / str(patid) / 'drugs.csv', rows,
                      DRUGS_COLUMNS)

    def add_fold(self, fold_name, train, val=None, test=None):
        """Write section 3's row-index arrays for one fold.

        A fold is ``int64`` positions in ``labels.csv``, not a directory
        of its own arrays, so this is all a fold is on disk. Extraction
        finds ``{fold}_train_rows.npy`` and writes that fold's
        standardization statistics; the loaders find all three.
        """
        fold_dir = self.data_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        for partition, rows in (('train', train), ('val', val),
                                ('test', test)):
            if rows is None:
                continue
            np.save(fold_dir / f'{fold_name}_{partition}_rows.npy',
                    np.asarray(rows, dtype=np.int64))

    def finish(self) -> Path:
        """Write labels.csv, the YAMLs and ClinVec; return the config."""
        write_csv(self.data_dir / 'labels.csv', self.labels_rows,
                  ['PATID', 'STAY_INDEX', 'DATA_SOURCE', 'TIME_TO_EVENT',
                   'EVENT_TYPE'])

        props_path = self.tmp_path / 'variable_properties.yaml'
        props_path.write_text(yaml.safe_dump(self.var_properties))

        clinvec = self.tmp_path / 'ClinVec_atc.csv'
        columns = ['node_id'] + [f'V{i + 1}' for i in range(CLINVEC_DIM)]
        write_csv(clinvec,
                  [[f'A0{i}'] + [0.0] * CLINVEC_DIM
                   for i in range(CLINVEC_ROWS)], columns)

        config = dict(self.config)
        config['DATA_DIR'] = str(self.data_dir)
        config['VARIABLE_PROPERTIES_PATH'] = str(props_path)
        config['CLINVEC_PATH'] = str(clinvec)
        config_path = self.tmp_path / 'dataset.yaml'
        config_path.write_text(yaml.safe_dump(config))
        return config_path

    @property
    def extracted(self) -> Path:
        return self.data_dir / 'extracted'

    @property
    def lookup_tables(self) -> Path:
        """Section 4.4's global tables, the *sibling* of ``extracted/``.

        C4 puts them here rather than among the per-episode arrays,
        because ``save_extracted`` clears its own directory and would
        otherwise delete a table that cost GPU-hours to build.
        """
        path = self.data_dir / 'lookup_tables'
        path.mkdir(exist_ok=True)
        return path

    def load(self, name: str) -> np.ndarray:
        return np.load(self.extracted / f'{name}.npy')


@pytest.fixture
def mini(tmp_path):
    """An empty MiniRoot; each test adds the patients its claim needs."""
    return MiniRoot(tmp_path)


@pytest.fixture
def one_patient(mini):
    """A single patient with four timesteps and one episode.

    Row 3 is the index time. Row 0 carries a drug dispensation and no
    valued feature at all, which section 2.7 requires to still be a
    timestep.
    """
    mini.add_patient(
        1001,
        timeseries=[
            ['2019-01-01T00:00:00Z', '', '', '', '', '', ''],
            ['2019-01-02T00:00:00Z', 1.5, 'L', '0', 'None', 'a note', ''],
            ['2019-01-03T00:00:00Z', 2.5, 'U', '25-50', 'Few', '', 1],
            ['2019-01-04T00:00:00Z', 3.5, 'L', '1-24', '', 'a note', ''],
        ],
        stays=[('DAD', '2019-01-01T00:00:00Z', '2019-01-04T00:00:00Z')],
        drugs=[('2019-01-01T00:00:00Z', 0, 2, 1.0),
               ('2019-01-01T00:00:00Z', 1, 3, 0.5)],
    )
    return mini
