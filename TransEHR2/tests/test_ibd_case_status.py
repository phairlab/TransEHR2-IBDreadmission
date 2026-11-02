"""The IBD case definition over a synthetic root.

Each patient is built so that one rule of the definition decides the outcome:
the per-source rule, the window, the classification thresholds, the CLM
weight, the cut at ``INDEX_TIME`` and the local calendar date.
"""

import csv
import numpy as np
import pandas as pd
import pytest
import yaml

from pathlib import Path

from IBDdataprep.superfeature import dump_superfeature

from ibd_case_status import code_descriptions, main

DEFINITION = (Path(__file__).resolve().parents[1] / 'configs' / 'cohorts' /
              'ibd_case_definition.yaml')

CD10, UC10 = "Crohn's disease of small intestine", 'Ulcerative (chronic) pancolitis'
CD9, UC9 = 'Regional enteritis of small intestine', 'Ulcerative (chronic) enterocolitis'


def _write(path, header, rows, encoding='utf-8'):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding=encoding) as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def diagnoses(*descs):
    return dump_superfeature({'DIAGNOSES': [{'DX_DESC': d, 'DX_TYPE': ['M']}
                                            for d in descs]})


def claims(*descs):
    return dump_superfeature({'CLAIMS': [{'CLAIM_DESC': 'Visit',
                                          'DX_DESC': list(descs)}]})


class Root:
    """A root, labels.csv, extracted_rows.npy and the dictionaries."""

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.data_dir = tmp_path / 'data'
        self.labels, self.drawn = [], []
        resources = self.data_dir / 'resources'
        _write(resources / 'icd10ca_dictionary.csv',
               ['ICD10CA Code', 'Long Description'],
               [['K500', CD10], ['K510', UC10], ['J180', 'Bronchopneumonia']],
               encoding='utf-8-sig')
        _write(resources / 'icd9_dictionary.csv',
               ['icd9code', 'long_description', 'short_description'],
               [['555.0', CD9, ''], ['556.0', UC9, ''], ['486', 'Pneumonia', '']],
               encoding='latin-1')

    def patient(self, patid, events, draw):
        """``events``: (timestamp, source, text); DAD and AMB ones are stays.

        ``draw`` is the stay index ``extracted_rows.npy`` names.
        """

        patient_dir = self.data_dir / 'root' / patid
        _write(patient_dir / 'timeseries.csv', ['TIMESTAMP', 'TEXT_SUPERFEATURE'],
               [[stamp, text] for stamp, _, text in events])
        stays = [(stamp, source) for stamp, source, _ in events
                 if source in ('DAD', 'AMB')]
        _write(patient_dir / 'stays.csv',
               ['PATID', 'STAY_INDEX', 'DATA_SOURCE', 'INDEX_TIME'],
               [[patid, i, source, stamp] for i, (stamp, source) in enumerate(stays)])
        for i, (_, source) in enumerate(stays):
            if i == draw:
                self.drawn.append(len(self.labels))
            self.labels.append([patid, i, source])

    def run(self, order=None, workers=1):
        _write(self.data_dir / 'labels.csv', ['PATID', 'STAY_INDEX', 'DATA_SOURCE'],
               self.labels)
        drawn = self.drawn if order is None else [self.drawn[i] for i in order]
        np.save(self.data_dir / 'extracted_rows.npy', np.asarray(drawn, dtype=np.int64))
        config = self.tmp_path / 'dataset.yaml'
        config.write_text(yaml.safe_dump({'DATA_DIR': str(self.data_dir)}))
        assert main([str(config), str(DEFINITION),
                     '--n-workers', str(workers)]) == 0
        return pd.read_csv(self.data_dir / 'ibd_case_status.csv',
                           dtype={'PATID': str}, keep_default_na=False)


@pytest.fixture
def root(tmp_path):
    return Root(tmp_path)


def _row(status, patid):
    return status[status['PATID'] == patid].iloc[0]


def test_two_dad_stays_make_a_case_and_the_threshold_is_strict(root):
    root.patient('p1', [('2015-01-10T18:00:00Z', 'DAD', diagnoses(CD10)),
                        ('2015-04-10T18:00:00Z', 'DAD', diagnoses(CD10))], draw=1)
    root.patient('p2', [('2015-01-10T18:00:00Z', 'DAD', diagnoses(CD10)),
                        ('2015-02-10T18:00:00Z', 'DAD', diagnoses(CD10)),
                        ('2015-03-10T18:00:00Z', 'DAD', diagnoses(CD10))], draw=2)
    status = root.run()
    # Score -2 is not below -2: IBDU. Score -3 is CD.
    assert _row(status, 'p1')[['IBD_CASE', 'STATUS', 'DAD_CD']].tolist() == [1, 'IBDU', 2]
    assert _row(status, 'p2')[['IBD_CASE', 'STATUS', 'SCORE']].tolist() == [1, 'CD', -3.0]


def test_records_after_the_index_time_are_not_used(root):
    root.patient('p1', [('2015-01-10T18:00:00Z', 'DAD', diagnoses(UC10)),
                        ('2015-04-10T18:00:00Z', 'DAD', diagnoses(UC10)),
                        ('2015-05-10T18:00:00Z', 'DAD', diagnoses(UC10))], draw=0)
    row = _row(root.run(), 'p1')
    assert row['IBD_CASE'] == 0 and row['STATUS'] == '' and row['DAD_UC'] == 1


def test_rules_never_combine_sources(root):
    root.patient('p1', [('2015-01-10T18:00:00Z', 'DAD', diagnoses(CD10)),
                        ('2015-02-10T18:00:00Z', 'AMB', diagnoses(CD10))], draw=1)
    row = _row(root.run(), 'p1')
    assert row['IBD_CASE'] == 0
    assert (row['DAD_CD'], row['AMB_CD']) == (1, 1)


def test_the_window_is_strictly_under_730_days(root):
    root.patient('in', [('2015-01-01T18:00:00Z', 'AMB', diagnoses(UC10)),
                        ('2016-12-30T18:00:00Z', 'AMB', diagnoses(UC10))], draw=1)
    root.patient('out', [('2015-01-01T18:00:00Z', 'AMB', diagnoses(UC10)),
                         ('2016-12-31T18:00:00Z', 'AMB', diagnoses(UC10))], draw=1)
    status = root.run()
    assert _row(status, 'in')['QUALIFIES_AMB'] == 1
    assert _row(status, 'out')['QUALIFIES_AMB'] == 0


def test_claims_need_four_dates_and_weigh_0_4(root):
    def history(n_claims):
        events = [(f'2015-0{m}-01T06:59:00Z', 'CLM', claims(UC9))
                  for m in range(1, n_claims + 1)]
        return events + [('2015-09-01T18:00:00Z', 'AMB', diagnoses('Bronchopneumonia'))]

    root.patient('three', history(3), draw=0)
    root.patient('four', history(4), draw=0)
    root.patient('six', history(6), draw=0)
    status = root.run()
    assert _row(status, 'three')['IBD_CASE'] == 0
    # 4 x 0.4 = 1.6 is not above 2; 6 x 0.4 = 2.4 is.
    assert _row(status, 'four')[['QUALIFIES_CLM', 'STATUS']].tolist() == [1, 'IBDU']
    assert _row(status, 'six')[['CLM_UC', 'STATUS']].tolist() == [6, 'UC']


def test_dates_are_local(root):
    # 11:00 and 23:59 on 1 January in Edmonton: two UTC dates, one local date.
    root.patient('p1', [('2015-01-01T18:00:00Z', 'AMB', diagnoses(CD10)),
                        ('2015-01-02T06:59:00Z', 'AMB', diagnoses(CD10))], draw=1)
    row = _row(root.run(), 'p1')
    assert row['AMB_CD'] == 2 and row['QUALIFIES_AMB'] == 0


def test_a_record_with_both_codes_counts_toward_both(root):
    root.patient('p1', [('2015-01-10T18:00:00Z', 'DAD', diagnoses(CD10, UC10)),
                        ('2015-02-10T18:00:00Z', 'DAD', diagnoses(UC10, 'Bronchopneumonia'))],
                 draw=1)
    row = _row(root.run(), 'p1')
    assert (row['DAD_CD'], row['DAD_UC'], row['STATUS']) == (1, 2, 'IBDU')


def test_diagnoses_on_no_stay_are_counted_and_not_used(root, capsys):
    root.patient('p1', [('2015-01-10T18:00:00Z', 'SCU', diagnoses(CD10)),
                        ('2015-02-10T18:00:00Z', 'DAD', diagnoses(CD10))], draw=0)
    row = _row(root.run(), 'p1')
    assert (row['DAD_CD'], row['AMB_CD']) == (1, 0)
    assert "not used: 1" in capsys.readouterr().out


def test_rows_follow_extracted_rows(root):
    root.patient('a', [('2015-01-10T18:00:00Z', 'DAD', diagnoses(CD10)),
                       ('2015-02-10T18:00:00Z', 'DAD', diagnoses(CD10))], draw=1)
    root.patient('b', [('2015-01-10T18:00:00Z', 'AMB', diagnoses(UC10))], draw=0)
    status = root.run(order=[1, 0], workers=2)
    assert status['PATID'].tolist() == ['b', 'a']
    assert status['EXTRACTED_ROW'].tolist() == [0, 1]
    assert status['LABELS_ROW'].tolist() == [2, 1]
    assert status['STAY_INDEX'].tolist() == [0, 1]


def test_a_description_shared_with_another_code_is_refused(root):
    path = root.data_dir / 'resources' / 'icd10ca_dictionary.csv'
    with path.open('a', newline='', encoding='utf-8') as handle:
        csv.writer(handle).writerow(['K520', CD10])
    codes = yaml.safe_load(DEFINITION.read_text())['CODES']
    with pytest.raises(ValueError, match='do not identify'):
        code_descriptions(root.data_dir / 'resources', codes)
