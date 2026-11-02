"""``ibd_case_status.csv`` -- which drawn episodes meet the IBD case definition.

One row per episode ``extracted_rows.npy`` names, in its order, so a row's
position is the index the fold arrays use. Each says whether the patient's
record up to the episode's ``INDEX_TIME`` meets the case definition, and if so
whether it classifies as Crohn's disease (CD), ulcerative colitis (UC) or IBD
unclassified (IBDU). It is a lookup for stratifying the phase-2 XAI cohort, or
for narrowing it to cases.

The algorithm is ``cohort.py``'s from the ibd-patient-cohort-definition
repository, steps 1 to 3, re-implemented rather than imported because that
repository is private. Its settings are in
``configs/cohorts/ibd_case_definition.yaml``.

Design decisions this script commits to
---------------------------------------

* **It reads the timeseries TransEHR2 reads, not the raw RMT23345 tables**
  (user, 2026-10-28). A patient is a case when what the model is given meets
  the definition. Linking raw records to timesteps would mean reproducing
  every edit Stages A and B make to a timestamp -- local time to UTC, the
  23:59 default for a date-only field, AMB visits shifted off a DAD minute,
  discharges clamped to death, external-cause stays dropped.
* **Every timestep at or before ``INDEX_TIME`` counts**, not only the
  ``MAX_EPISODE_LEN_STEPS`` the model reads (user, 2026-10-28). So the status
  belongs to the episode: no record after its prediction origin is used.
* **Codes are matched by description.** The prepare scripts replaced each
  code with its dictionary description. That loses nothing here as long as
  every CD or UC description belongs to CD or UC codes alone, which is checked
  against the dictionaries on every run. A claim coded to three ICD-9 digits
  had no description, so TransEHR2 never sees it and neither does this.
* **The source is read off the structure.** A diagnosis inside a ``CLAIMS``
  block is CLM. A ``DIAGNOSES`` block is DAD or AMB, whichever stay in
  ``stays.csv`` has its ``INDEX_TIME`` at that timestep: a DAD stay's
  diagnoses sit on its discharge and an AMB visit's on its arrival, and
  ``timeline.py`` never lets the two share a minute. A block on no stay's
  minute is counted in the summary and not used.
* **What counts as a record** is a DAD stay, an AMB visit, or a claim block
  (one per service code per day). ``cohort.py`` counts raw claim rows, so CLM
  counts can be lower here.
* **Dates are local calendar dates**, which undoes the UTC shift of the 23:59
  default. A DAD stay is dated by its discharge, where ``cohort.py`` uses its
  admission.
* **There is no registry step.** ``cohort.py``'s step 4 drops cases with no
  REG row; that is a question for the cohort, not for the case definition.

The output is patient-level and stays on the cluster. Nothing per patient
reaches stdout: the summary is counts.

Usage:
    python scripts/ibd_case_status.py DATASET_CONFIG DEFINITION_CONFIG \\
        [--output DATA_DIR/ibd_case_status.csv] \\
        [--resources DATA_DIR/resources] [--n-workers 8]
"""

from __future__ import annotations

import argparse
import csv
import numpy as np
import pandas as pd
import re
import sys
import yaml

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

from IBDdataprep.code_dictionaries import ICD9, ICD10CA, descriptions_for
from IBDdataprep.superfeature import parse_superfeature

SOURCES = ('DAD', 'AMB', 'CLM')
GROUPS = ('CD', 'UC')
DICTIONARIES = {'ICD10CA': ICD10CA, 'ICD9': ICD9}

TIMESERIES_FILE = 'timeseries.csv'
STAYS_FILE = 'stays.csv'
TIMESTAMP = 'TIMESTAMP'
TEXT_FEATURE = 'TEXT_SUPERFEATURE'

COLUMNS = (['EXTRACTED_ROW', 'LABELS_ROW', 'PATID', 'STAY_INDEX', 'IBD_CASE',
            'STATUS', 'SCORE']
           + [f'{source}_{group}' for source in SOURCES for group in GROUPS]
           + [f'QUALIFIES_{source}' for source in SOURCES])


def code_descriptions(resources: Path, codes: dict) -> dict:
    """``{dictionary: {'CD': set, 'UC': set}}`` of descriptions to match.

    Raises if a description fails to identify its group: one shared by CD and
    UC, or by either and a code outside both.
    """

    found = {}
    for name, groups in codes.items():
        dictionary = DICTIONARIES[name]
        selected = {group: descriptions_for(resources, dictionary,
                                            groups[group])
                    for group in GROUPS}
        prefixes = [code for group in GROUPS for code in groups[group]]
        others = descriptions_for(resources, dictionary, [''], less=prefixes)
        empty = [group for group in GROUPS if not selected[group]]
        if empty:
            raise ValueError(f'{dictionary.filename} has no description for '
                             f'{name} {empty}')
        ambiguous = ((selected['CD'] & selected['UC'])
                     | ((selected['CD'] | selected['UC']) & others))
        if ambiguous:
            raise ValueError(f'{dictionary.filename}: these descriptions do '
                             f'not identify CD or UC: {sorted(ambiguous)}')
        found[name] = selected
    return found


@lru_cache(maxsize=None)
def _matcher(descriptions: tuple) -> re.Pattern:
    """A cheap test that skips the parse on rows naming no CD or UC code."""

    return re.compile('|'.join(re.escape(text) for text in descriptions))


def _flags(values, selected: dict) -> tuple:
    """``(is_cd, is_uc)`` for a list of ``DX_DESC`` values."""

    flat = set()
    for value in values:
        flat.update([value] if isinstance(value, str) else value)
    return tuple(bool(flat & selected[group]) for group in GROUPS)


def ibd_records(path: Path, source_at: dict, descriptions: dict) -> tuple:
    """Every CD- or UC-coded record in one ``timeseries.csv``.

    Returns ``(records, unattributed)``: records are ``(timestamp, source,
    is_cd, is_uc)``, and ``unattributed`` counts ``DIAGNOSES`` blocks carrying
    a code at a minute no stay in ``source_at`` holds.
    """

    every = tuple(sorted(set().union(*(selected[group]
                                       for selected in descriptions.values()
                                       for group in GROUPS))))
    matches = _matcher(every)
    records, unattributed = [], 0
    with path.open(newline='') as handle:
        rows = csv.reader(handle)
        header = next(rows)
        at, text_at = header.index(TIMESTAMP), header.index(TEXT_FEATURE)
        for row in rows:
            if not matches.search(row[text_at]):
                continue
            stamp = pd.Timestamp(row[at])
            parsed = parse_superfeature(row[text_at])
            for block in parsed.get('CLAIMS') or []:
                flags = _flags([block.get('DX_DESC') or []],
                               descriptions['ICD9'])
                if any(flags):
                    records.append((stamp, 'CLM') + flags)
            flags = _flags([item.get('DX_DESC') or []
                            for item in parsed.get('DIAGNOSES') or []],
                           descriptions['ICD10CA'])
            if any(flags):
                if stamp in source_at:
                    records.append((stamp, source_at[stamp]) + flags)
                else:
                    unattributed += 1
    return records, unattributed


def meets_rule(dates: list, min_records: int, window_days: int) -> bool:
    """``min_records`` of the distinct sorted dates, first to last <
    ``window_days`` apart."""

    if len(dates) < max(min_records, 1):
        return False
    if min_records <= 1:
        return True
    k = min_records - 1
    return any((dates[i] - dates[i - k]).days < window_days
               for i in range(k, len(dates)))


def case_status(records: list, index_time: pd.Timestamp,
                definition: dict) -> dict:
    """One episode's row, from the records at or before its ``INDEX_TIME``."""

    row, score = {}, 0.0
    for source in SOURCES:
        mine = [record for record in records
                if record[1] == source and record[0] <= index_time]
        cd = sum(record[2] for record in mine)
        uc = sum(record[3] for record in mine)
        row[f'{source}_CD'], row[f'{source}_UC'] = cd, uc
        score += definition['WEIGHTS'][source] * (uc - cd)
        dates = sorted({record[0].tz_convert(definition['LOCAL_TIMEZONE'])
                        .date() for record in mine})
        rule = definition['RULES'][source]
        row[f'QUALIFIES_{source}'] = int(meets_rule(
            dates, rule['MIN_RECORDS'], rule['WINDOW_DAYS']))
    row['IBD_CASE'] = int(any(row[f'QUALIFIES_{source}']
                              for source in SOURCES))
    row['SCORE'] = round(score, 6)
    if not row['IBD_CASE']:
        row['STATUS'] = ''
    elif score < definition['CD_THRESHOLD']:
        row['STATUS'] = 'CD'
    elif score > definition['UC_THRESHOLD']:
        row['STATUS'] = 'UC'
    else:
        row['STATUS'] = 'IBDU'
    return row


def _patient(task: tuple) -> tuple:
    """``(rows, unattributed)`` for one patient's drawn episodes."""

    patient_dir, stay_indices, descriptions, definition = task
    stays = pd.read_csv(patient_dir / STAYS_FILE, dtype=str,
                        keep_default_na=False)
    stays['STAY_INDEX'] = stays['STAY_INDEX'].astype(int)
    index_times = pd.to_datetime(stays['INDEX_TIME'], utc=True,
                                 format='ISO8601')
    source_at = dict(zip(index_times, stays['DATA_SOURCE']))
    index_of = dict(zip(stays['STAY_INDEX'], index_times))
    records, unattributed = ibd_records(patient_dir / TIMESERIES_FILE,
                                        source_at, descriptions)
    rows = [case_status(records, index_of[stay], definition)
            for stay in stay_indices]
    return rows, unattributed


def case_table(data_dir: Path, descriptions: dict, definition: dict,
               n_workers: int) -> tuple:
    """The output frame, and the unattributed-block count."""

    labels = pd.read_csv(data_dir / 'labels.csv', dtype={'PATID': str},
                         keep_default_na=False,
                         usecols=['PATID', 'STAY_INDEX'])
    extracted = np.load(data_dir / 'extracted_rows.npy')
    drawn = labels.iloc[extracted].reset_index(drop=True)
    drawn.insert(0, 'LABELS_ROW', extracted)
    drawn.insert(0, 'EXTRACTED_ROW', np.arange(len(drawn)))

    patients = list(dict.fromkeys(drawn['PATID']))
    stays_of = drawn.groupby('PATID', sort=False)['STAY_INDEX'].apply(list)
    tasks = [(data_dir / 'root' / patid, stays_of[patid], descriptions,
              definition) for patid in patients]
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_patient, tasks, chunksize=64))
    else:
        results = [_patient(task) for task in tasks]

    by_patient = dict(zip(patients, results))
    taken = {patid: 0 for patid in patients}
    rows = []
    for patid in drawn['PATID']:
        rows.append(by_patient[patid][0][taken[patid]])
        taken[patid] += 1
    status = pd.concat([drawn, pd.DataFrame(rows)], axis=1)[COLUMNS]
    return status, sum(result[1] for result in results)


def _print_summary(status: pd.DataFrame, unattributed: int,
                   output: Path) -> None:
    cases = status[status['IBD_CASE'] == 1]
    print(f'case table:   {output}')
    print(f'episodes:     {len(status)}')
    print(f'  meeting the case definition: {len(cases)}')
    for group in ('CD', 'UC', 'IBDU'):
        print(f'    {group:<5} {int((cases["STATUS"] == group).sum())}')
    print('  meeting each source rule (they overlap):')
    for source in SOURCES:
        print(f'    {source:<5} {int(status[f"QUALIFIES_{source}"].sum())}')
    print(f'DIAGNOSES blocks with a CD or UC code on no stay\'s minute, '
          f'not used: {unattributed}')


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Write ibd_case_status.csv: the IBD case definition and '
                    'CD/UC/IBDU status of every drawn episode.')
    parser.add_argument('dataset_config', type=Path,
                        help='Dataset config; DATA_DIR is read from it')
    parser.add_argument('definition_config', type=Path,
                        help='configs/cohorts/ibd_case_definition.yaml')
    parser.add_argument('--output', type=Path, default=None,
                        help='Default: DATA_DIR/ibd_case_status.csv')
    parser.add_argument('--resources', type=Path, default=None,
                        help='Code dictionaries. Default: DATA_DIR/resources')
    parser.add_argument('--n-workers', type=int, default=1,
                        help='Patients read in parallel [default: %(default)s]')
    args = parser.parse_args(argv)

    data_dir = Path(yaml.safe_load(args.dataset_config.read_text())['DATA_DIR'])
    definition = yaml.safe_load(args.definition_config.read_text())
    output = args.output or data_dir / 'ibd_case_status.csv'
    resources = args.resources or data_dir / 'resources'

    descriptions = code_descriptions(resources, definition['CODES'])
    status, unattributed = case_table(data_dir, descriptions, definition,
                                      max(args.n_workers, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    status.to_csv(output, index=False, lineterminator='\n')
    _print_summary(status, unattributed, output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
