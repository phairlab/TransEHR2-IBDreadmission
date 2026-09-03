"""The global lookup tables and the checks that pair them to an extraction.

The claims here are about the *pairing* between a table and the
extraction that indexes it, which is what nothing else can catch: both
tables are order-sensitive, so a table built against a different
extraction has the right shape and the right dtype and gathers the wrong
rows. Each guard is tested by the failure it exists to catch, not only by
a passing build -- a checker that only ever passes proves nothing.

The text pass is exercised with a stub tokenizer and a stub embedder. The
real ones are a 70B-parameter download; what ``embed.py`` is responsible
for is the pass around them -- the row order, the streaming write, the
mask derivation and the refusals -- and every one of those is independent
of which model is loaded.
"""

import numpy as np
import pandas as pd
import pickle
import pytest
import sys
import yaml

from pathlib import Path

from TransEHR2.data import manifest
from TransEHR2.data.preprocessing import load_dataset

from embed import (
    TABLES_DIRNAME, build_drug_table, build_text_tables, load_text_strings,
    record_pad_token_id
)
from embed import main as embed_main

from .conftest import CLINVEC_DIM, CLINVEC_ROWS
from .test_datasets import run, write_tables


PAD_ID = 128009  # Outside ASCII, as a Llama 3.x added token is.
STUB_DIM = 3
STUB_LEN = 8


class StubTokenizer:
    """One character per token, right-padded -- enough to be a tokenizer.

    The mask is set from the content, exactly as HuggingFace sets it, so
    ``mask == (ids != pad_id)`` holds and the derivation check passes.
    """

    pad_token_id = PAD_ID

    def __call__(self, texts, max_length, padding, truncation,
                 return_attention_mask, return_tensors):
        ids = np.full((len(texts), max_length), PAD_ID, dtype=np.int64)
        mask = np.zeros((len(texts), max_length), dtype=np.int64)
        for row, text in enumerate(texts):
            codes = [ord(c) for c in text][:max_length]
            ids[row, :len(codes)] = codes
            mask[row, :len(codes)] = 1
        return {'input_ids': ids, 'attention_mask': mask}


class PadInContentTokenizer(StubTokenizer):
    """A tokenizer that emits ``pad_id`` inside a sequence.

    This is what a ``TEXT_SUPERFEATURE`` containing the five characters
    ``[PAD]`` would do: HuggingFace splits added tokens out of input
    text, so the id would appear mid-sequence and the derivation section
    4.4 relies on would be wrong. Vanishingly unlikely in ICD/CCI/MIS
    descriptions, which is why it is a checked assumption rather than a
    stored array.
    """

    def __call__(self, texts, **kwargs):
        encoded = super().__call__(texts, **kwargs)
        encoded['input_ids'][0, 0] = PAD_ID
        return encoded


def stub_embed(ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """An embedding that names the string it came from.

    Row *r* is the first token id of string *r*, repeated, so a table row
    identifies its source and a misordered pass is visible by eye.
    """
    return np.repeat(
        ids[:, :1].astype(np.float32), STUB_DIM, axis=1
    )


def build(tmp_path, strings, tokenizer=None, embed=stub_embed):
    """Run the text pass over ``strings`` and return the tables."""
    out_dir = tmp_path / TABLES_DIRNAME
    build_text_tables(
        strings, tokenizer or StubTokenizer(), embed, str(out_dir),
        STUB_DIM, max_length=STUB_LEN, batch_size=2
    )
    return out_dir


def write_clinvec(path: Path, n_rows: int, dim: int) -> np.ndarray:
    """A ClinVec file whose row *r* is filled with *r + 1*."""
    vectors = np.repeat(
        np.arange(1, n_rows + 1, dtype=np.float32)[:, None], dim, axis=1
    )
    frame = pd.DataFrame(vectors, columns=[f'V{i + 1}' for i in range(dim)])
    frame.insert(0, 'node_id', [f'A0{i}' for i in range(n_rows)])
    frame.to_csv(path, index=False)
    return vectors


# --- the drug table ---------------------------------------------------

def test_the_drug_table_is_the_vocabulary_plus_an_all_zero_pad_row(tmp_path):
    """``(V + 1, D)``, where V is the *vocabulary's* row count.

    Deriving V from the cohort would leave the pad colliding with a drug
    no patient in this cohort happened to receive."""
    vectors = write_clinvec(tmp_path / 'ClinVec_atc.csv', 5, 4)
    path = build_drug_table(
        str(tmp_path / 'ClinVec_atc.csv'), str(tmp_path / TABLES_DIRNAME)
    )

    table = np.load(path)
    assert table.shape == (6, 4)
    assert table.dtype == np.float32
    assert np.array_equal(table[:-1], vectors)
    assert not table[-1].any(), "the pad row is all zero"


def test_the_drug_table_needs_no_llm(tmp_path, monkeypatch):
    """``--tables drug`` must not import the LLM stack: the table is a
    read of a CSV, and requiring a 70B download to rebuild it would make
    the cheap table as expensive as the dear one. Poisoning the
    module is what makes this a test rather than a claim -- the import
    is inside the text branch, so only running the drug branch proves
    the branch is where it is.
    """
    write_clinvec(tmp_path / 'ClinVec_atc.csv', 3, 2)
    config = tmp_path / 'dataset.yaml'
    config.write_text(yaml.safe_dump({
        'DATA_DIR': str(tmp_path),
        'CLINVEC_PATH': str(tmp_path / 'ClinVec_atc.csv'),
    }))
    monkeypatch.setitem(sys.modules, 'TransEHR2.modules', None)

    assert embed_main([str(config), '--tables', 'drug']) == 0
    assert (tmp_path / TABLES_DIRNAME / 'drug_embeddings.npy').exists()
    with pytest.raises(ImportError):
        import TransEHR2.modules  # noqa: F401


# --- the text tables --------------------------------------------------

def test_the_text_tables_follow_text_strings_order(tmp_path):
    """The row order is ``text_strings.pkl``'s and nothing else's: that
    correspondence is the whole of what makes the extracted ``int32``
    values valid indices."""
    strings = ['aa', 'b', 'ccc', 'd', 'ee']
    out_dir = build(tmp_path, strings)

    tokens = np.load(out_dir / 'text_tokens.npy')
    embeddings = np.load(out_dir / 'text_embeddings.npy')

    assert tokens.shape == (5, STUB_LEN)
    assert tokens.dtype == np.int32
    assert embeddings.shape == (5, STUB_DIM)
    assert embeddings.dtype == np.float32
    for row, text in enumerate(strings):
        assert tokens[row, 0] == ord(text[0])
        assert (embeddings[row] == ord(text[0])).all()


def test_the_batch_boundary_does_not_reorder_the_table(tmp_path):
    """Five strings at ``batch_size=2`` is three batches, the last one
    short. A pass that wrote batches back at the wrong offset would still
    produce a table of the right shape."""
    strings = [chr(ord('a') + i) * 2 for i in range(5)]
    embeddings = np.load(build(tmp_path, strings) / 'text_embeddings.npy')
    assert [int(row[0]) for row in embeddings] == [
        ord(s[0]) for s in strings
    ]


def test_no_attention_mask_reaches_disk(tmp_path):
    """The mask is a local variable of this pass. It has one consumer,
    once, and is exactly ``ids != pad_id``."""
    out_dir = build(tmp_path, ['aa', 'bb'])
    assert sorted(p.name for p in out_dir.iterdir()) == [
        'text_embeddings.npy', 'text_tokens.npy'
    ]


def test_the_mask_is_derivable_from_the_tokens(tmp_path):
    """The positive half, on the artifact that survives: the stored
    tokens alone recover the mask the embedding was pooled with."""
    strings = ['aa', 'bbbb']
    tokens = np.load(build(tmp_path, strings) / 'text_tokens.npy')
    derived = tokens != PAD_ID
    assert derived.sum(axis=1).tolist() == [len(s) for s in strings]


def test_the_build_fails_when_a_pad_id_appears_in_content(tmp_path):
    """The one assumption the derivation makes about *content* rather
    than about the vocabulary. No mask is stored, so this is its sole
    guard -- it has to fail the build rather than embed a string whose
    mask cannot be recovered."""
    with pytest.raises(AssertionError, match='attention mask'):
        build(tmp_path, ['aa', 'bb'], tokenizer=PadInContentTokenizer())


def test_an_empty_table_still_names_its_width(tmp_path):
    """A cohort with no text at all writes ``(0, D)``, not ``(0,)``: the
    loader reads the table's own width, so a table with no rows still has
    to declare one."""
    out_dir = build(tmp_path, [])
    assert np.load(out_dir / 'text_embeddings.npy').shape == (0, STUB_DIM)
    assert np.load(out_dir / 'text_tokens.npy').shape == (0, STUB_LEN)


# --- what the pass refuses --------------------------------------------

def test_a_blank_string_is_refused_rather_than_skipped(tmp_path):
    """Blanks are kept out of the table, and extraction already does
    that. Skipping one *here* would shift every later row out from under
    indices extraction has already assigned, so the case is an upstream
    bug to report."""
    with open(tmp_path / 'text_strings.pkl', 'wb') as f:
        pickle.dump(['aa', '   ', 'bb'], f)
    with pytest.raises(ValueError, match='blank string'):
        load_text_strings(str(tmp_path))


def test_a_missing_string_table_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match='text_strings.pkl'):
        load_text_strings(str(tmp_path))


def test_the_resolved_pad_token_id_reaches_metadata(tmp_path):
    """Extraction records ``LLM_NAME``, ``TOKENIZER_PAD_TOKEN`` and
    ``MAX_TOKEN_LENGTH`` but loads no tokenizer, so the resolved id is
    established here -- the fourth fact that has to be recorded with the
    other three for a tokenizer change to be detectable."""
    with open(tmp_path / 'metadata.pkl', 'wb') as f:
        pickle.dump({'max_ts_len': 4}, f)
    record_pad_token_id(str(tmp_path), PAD_ID)
    with open(tmp_path / 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    assert metadata['pad_token_id'] == PAD_ID
    assert metadata['max_ts_len'] == 4, "the rest of it survives"


# --- where the tables live --------------------------------------------

def test_a_re_extraction_does_not_delete_the_tables(one_patient):
    """``save_extracted`` clears every ``.npy``, ``.pkl`` and ``.npz``
    in its own directory, so a text table kept there -- tens of gigabytes
    and hours of GPU time -- would not survive the next extraction run.
    In ``lookup_tables/`` it survives by construction."""
    one_patient.add_fold('fold0', train=[0])
    assert run(one_patient) == 0
    write_tables(one_patient)
    tables = sorted(p.name for p in one_patient.lookup_tables.iterdir())

    assert run(one_patient) == 0
    assert sorted(
        p.name for p in one_patient.lookup_tables.iterdir()
    ) == tables


def test_a_table_left_in_extracted_is_not_found(one_patient):
    """The corollary: ``extracted/`` is no longer where the loader
    looks, so a table put there is as good as absent."""
    one_patient.add_fold('fold0', train=[0])
    assert run(one_patient) == 0
    write_tables(one_patient)
    for name in ('text_embeddings.npy', 'drug_embeddings.npy'):
        (one_patient.lookup_tables / name).rename(
            one_patient.extracted / name
        )
    with pytest.raises(FileNotFoundError, match='text_embeddings.npy'):
        load_dataset(str(one_patient.extracted), fold='fold0')


# --- every stored index is a row of its table -------------------------

def test_an_index_past_the_end_of_the_table_is_refused(one_patient):
    one_patient.add_fold('fold0', train=[0])
    assert run(one_patient) == 0
    write_tables(one_patient)
    values = np.load(one_patient.extracted / 'text_values_0.npy')
    values[0] = 99
    np.save(one_patient.extracted / 'text_values_0.npy', values)
    with pytest.raises(ValueError, match='The table and the extraction'):
        load_dataset(str(one_patient.extracted), fold='fold0')


def test_a_negative_index_is_refused(one_patient):
    """The half the gather would not catch: an index past the end raises
    at ``__getitem__``, but a negative one wraps to the far end of the
    table and returns some other string's vector without a word."""
    one_patient.add_fold('fold0', train=[0])
    assert run(one_patient) == 0
    write_tables(one_patient)
    values = np.load(one_patient.extracted / 'text_values_0.npy')
    values[0] = -1
    np.save(one_patient.extracted / 'text_values_0.npy', values)
    with pytest.raises(ValueError, match='table and the extraction'):
        load_dataset(str(one_patient.extracted), fold='fold0')


def test_the_drug_pad_index_is_in_range(one_patient):
    """The pad index is ``V``, the table's last row, so the range check
    must accept it -- a check written as ``< V`` would reject every
    padded slot in the cohort."""
    one_patient.add_fold('fold0', train=[0])
    assert run(one_patient) == 0
    write_tables(one_patient)
    values = np.load(one_patient.extracted / 'drug_values_0.npy')
    assert (values == CLINVEC_ROWS).any(), "the fixture pads some slots"
    load_dataset(str(one_patient.extracted), fold='fold0')


# --- the checksum in manifest.csv -------------------------------------

@pytest.fixture
def registered(tmp_path, monkeypatch):
    """A DATA_ROOT and a manifest that this test owns."""
    monkeypatch.setenv('SHARED_DATA_ROOT', str(tmp_path))
    monkeypatch.setattr(manifest, 'MANIFEST_PATH', str(tmp_path / 'm.csv'))
    (tmp_path / 'm.csv').write_text(
        'path,sha256,source,source_type\n'
        'lookup_tables/drug_embeddings.npy,pending,python embed.py,build\n'
    )
    write_clinvec(tmp_path / 'ClinVec_atc.csv', 3, 2)
    return build_drug_table(
        str(tmp_path / 'ClinVec_atc.csv'), str(tmp_path / TABLES_DIRNAME)
    )


def test_a_checksum_is_recorded_and_then_verifies(registered):
    checksum = manifest.record_checksum(registered)
    assert checksum == manifest.sha256_of(registered)
    manifest.verify(registered)


def test_a_table_that_disagrees_with_the_manifest_is_refused(registered):
    """The tables are order-sensitive, so a table that copied wrong
    keeps its shape and its dtype and gathers the wrong rows. Only the
    checksum notices."""
    manifest.record_checksum(registered)
    table = np.load(registered)
    table[0] += 1.0
    np.save(registered, table)
    with pytest.raises(ValueError, match='does not match'):
        manifest.verify(registered)


def test_an_unrecorded_checksum_does_not_pass_as_verified(
    registered, capsys
):
    """A 'pending' row is a table nobody has checksummed yet. It must
    say so rather than read as a passing check."""
    manifest.verify(registered)
    assert 'not verified' in capsys.readouterr().out


def test_record_checksum_will_not_add_a_row(registered, tmp_path, capsys):
    """``update_manifest.sh`` refuses to add entries so that data which
    cannot be shared is never registered for distribution. A Python path
    into the same file has to refuse too, or it walks around the rule."""
    other = tmp_path / TABLES_DIRNAME / 'text_embeddings.npy'
    np.save(other, np.zeros((2, 2), dtype=np.float32))
    assert manifest.record_checksum(str(other)) is None
    assert 'not in' in capsys.readouterr().out
    assert 'text_embeddings' not in (tmp_path / 'm.csv').read_text()


def test_a_table_outside_data_root_is_reported_not_refused(
    tmp_path, monkeypatch, capsys
):
    """Fixtures and tmp-dir cohorts are not distributed artifacts and
    have no manifest row. Refusing them would make every test carry a
    checksum of a file it just wrote -- but the skip has to be audible,
    or an unregistered table in a real run goes unchecked in silence."""
    monkeypatch.setenv('SHARED_DATA_ROOT', str(tmp_path / 'elsewhere'))
    path = tmp_path / 'table.npy'
    np.save(path, np.zeros((2, 2), dtype=np.float32))
    manifest.verify(str(path))
    assert 'not checksum-verified' in capsys.readouterr().out


def test_the_loader_verifies_the_checksum_at_use(
    one_patient, tmp_path_factory, monkeypatch
):
    """The checksum is verified when the tables are opened, not only
    when they are fetched, and this is the only test that the call site
    is on the path a training run takes.
    The cohort is made to sit under a DATA_ROOT this test owns, which is
    what gives its tables a manifest row at all."""
    one_patient.add_fold('fold0', train=[0])
    assert run(one_patient) == 0
    write_tables(one_patient)

    root = one_patient.data_dir
    manifest_path = tmp_path_factory.mktemp('manifest') / 'm.csv'
    manifest_path.write_text(
        'path,sha256,source,source_type\n'
        'lookup_tables/text_embeddings.npy,pending,python embed.py,build\n'
    )
    monkeypatch.setenv('SHARED_DATA_ROOT', str(root))
    monkeypatch.setattr(manifest, 'MANIFEST_PATH', str(manifest_path))

    table = root / 'lookup_tables' / 'text_embeddings.npy'
    manifest.record_checksum(str(table))
    load_dataset(str(one_patient.extracted), fold='fold0')

    np.save(table, np.load(table) + 1.0)
    with pytest.raises(ValueError, match='does not match'):
        load_dataset(str(one_patient.extracted), fold='fold0')


def test_the_tables_embed_this_extraction_and_load_back(one_patient):
    """The loop closes. The strings come from an
    extraction, ``embed.py`` builds both tables from them and from that
    extraction's own ClinVec file, and the loader accepts the pair --
    row counts, widths, pad row and every index in range."""
    one_patient.add_fold('fold0', train=[0])
    assert run(one_patient) == 0

    strings = load_text_strings(str(one_patient.extracted))
    assert strings, "the fixture patient carries text"
    out_dir = str(one_patient.lookup_tables)
    build_text_tables(
        strings, StubTokenizer(), stub_embed, out_dir, STUB_DIM,
        max_length=STUB_LEN, batch_size=2
    )
    build_drug_table(str(one_patient.tmp_path / 'ClinVec_atc.csv'), out_dir)
    record_pad_token_id(str(one_patient.extracted), PAD_ID)

    dataset = load_dataset(str(one_patient.extracted), fold='fold0')
    assert dataset.lookup_table_dims == [STUB_DIM, CLINVEC_DIM]
    item = dataset[0]
    assert item['val_lookup_sparse'][0]['values'].shape[-1] == STUB_DIM
