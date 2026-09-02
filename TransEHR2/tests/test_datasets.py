"""One test per claim C2 makes (blueprint sections 4.2, 4.3, 5, 5.1).

Every test names the reading it defends, so a future change that breaks
one can tell whether it is breaking a decision or a bug. The inputs are
C1's output: each test runs the extractor over a root the suite writes
itself, then writes the two global lookup tables C4 will build, which is
the parcel boundary -- C2 reads the format C1 produced and nothing else.
"""

import numpy as np
import pickle
import pytest
import torch

from TransEHR2.data.datasets import LazyArray
from TransEHR2.data.preprocessing import (
    collate_tensorized, get_text_counts_from_dataset,
    get_text_counts_from_dataset_vectorized, load_dataset,
    prepare_dataloaders
)

from extract_data import main as extract_main

from .conftest import CLINVEC_DIM, CLINVEC_ROWS, collate_for_model

# Deliberately not CLINVEC_DIM: section 5.1 makes the embedding width
# per-feature, text being 4096 or 8192 beside ClinVec's 128.
TEXT_EMBED_DIM = 5


def run(mini):
    """Run the extractor over a MiniRoot and return the exit code."""
    config_path = mini.finish()
    return extract_main([str(config_path)])


def write_tables(mini, text_dim=TEXT_EMBED_DIM, dtype=np.float32):
    """Write the global lookup tables of section 4.4, as C4 will.

    Row *r* of each table is filled with ``r + 1``, so a gathered vector
    names the row it came from; the drug table's final row is the
    all-zero pad that unused slots index.
    """
    with open(mini.extracted / 'text_strings.pkl', 'rb') as f:
        n_strings = len(pickle.load(f))

    text = np.repeat(
        np.arange(1, n_strings + 1, dtype=dtype)[:, None], text_dim, axis=1
    )
    np.save(mini.extracted / 'text_embeddings.npy', text)

    drug = np.repeat(
        np.arange(1, CLINVEC_ROWS + 1, dtype=np.float32)[:, None],
        CLINVEC_DIM, axis=1
    )
    drug = np.vstack([drug, np.zeros((1, CLINVEC_DIM), dtype=np.float32)])
    np.save(mini.extracted / 'drug_embeddings.npy', drug)


def extracted(mini, fold='fold0', rows=(0,), **kwargs):
    """Extract, write the tables, and load the result as a dataset."""
    mini.add_fold(fold, train=list(rows), val=list(rows), test=list(rows))
    assert run(mini) == 0
    write_tables(mini, **kwargs)
    return load_dataset(str(mini.extracted), fold=fold)


@pytest.fixture
def dataset(one_patient):
    """``one_patient``, extracted and loaded with fold0's statistics."""
    return extracted(one_patient)


# --- the one-hot expansion and the -1 sentinel (section 4.3) ---------

def test_one_hot_width_is_the_declared_category_size(dataset):
    """Section 4.3: ``size`` is the number of categories for a
    categorical feature and the number of levels for an ordinal one, and
    that is the width __getitem__ expands an index to."""
    item = dataset[0]
    # CAT has 2 categories; ORD has 3 levels and UB has 2.
    assert [v.shape[-1] for v in item['val_categorical_values']] == [2]
    assert [v.shape[-1] for v in item['val_ordinal_values']] == [3, 2]


def test_category_index_becomes_a_one_hot_row(dataset):
    """An in-domain index k sets column k and nothing else."""
    item = dataset[0]
    # Section 4.2 right-aligns, so the index time is the final column.
    # one_patient's last row is CAT 'L' (category 0) and ORD '1-24'
    # (level 1).
    assert item['val_categorical_values'][0][-1].tolist() == [1.0, 0.0]
    assert item['val_ordinal_values'][0][-1].tolist() == [0.0, 1.0, 0.0]


def test_unobserved_index_expands_to_an_all_zero_row(dataset):
    """Section 4.3's table, row ``(0, -1)``: not observed. The one-hot
    row is all zero, which np.zeros storage could not have expressed --
    it would have said category 0."""
    item = dataset[0]
    # one_patient's first timestep carries a drug and no valued feature.
    assert item['val_categorical_indicators'][0, 0] == 0.0
    assert item['val_categorical_values'][0][0].sum() == 0.0
    # And its last timestep has no UB value.
    assert item['val_ordinal_indicators'][-1, 1] == 0.0
    assert item['val_ordinal_values'][1][-1].sum() == 0.0


def test_out_of_domain_index_expands_to_an_all_zero_row(mini):
    """Section 4.3's table, row ``(1, -1)``: observed, but the value is
    not in ``category_map``. The indicator stays 1 -- the timestep is
    still a real observation, and the indicator loss still trains on it
    -- while the value row is all zero."""
    mini.add_patient(
        1001,
        timeseries=[
            ['2019-01-01T00:00:00Z', 1.0, 'L', '0', 'Few', '', ''],
            # 'Z' is in no category_map.
            ['2019-01-02T00:00:00Z', 1.0, 'Z', '0', 'Few', '', ''],
        ],
        stays=[('DAD', '2019-01-01T00:00:00Z', '2019-01-02T00:00:00Z')],
    )
    item = extracted(mini)[0]
    assert item['val_categorical_indicators'][-1, 0] == 1.0, "observed"
    assert item['val_categorical_values'][0][-1].sum() == 0.0
    # The timestep before it holds a value that does map.
    assert item['val_categorical_values'][0][-2].tolist() == [1.0, 0.0]


def test_the_losses_guard_still_fires_on_a_zero_row(mini):
    """Section 4.3's reason for all of the above: ``losses.py``'s four
    ``target.sum(dim=-1) > 0`` guards (``:183``, ``:222``, ``:398``,
    ``:443``) exist because softmax cannot emit an all-zero
    distribution, so cross-entropy has no defined target there. This is
    that expression, verbatim, over a collated batch."""
    mini.add_patient(
        1001,
        timeseries=[
            ['2019-01-01T00:00:00Z', 1.0, 'L', '0', 'Few', '', ''],
            ['2019-01-02T00:00:00Z', 1.0, 'Z', '0', 'Few', '', ''],
        ],
        stays=[('DAD', '2019-01-01T00:00:00Z', '2019-01-02T00:00:00Z')],
    )
    dataset = extracted(mini)
    batch = collate_tensorized([dataset[0]])

    target = batch['val_data']['categorical']['values'][0]
    valid = target.sum(dim=-1) > 0
    # Two padded timesteps, then the mapping value, then 'Z'.
    assert valid[0].tolist() == [False, False, True, False]
    assert not valid.all(), "an out-of-domain target must be skipped"

    # The same expression guards the ordinal loss (``:445``), where the
    # cost of getting it wrong is higher: BetaLoss penalizes by ordinal
    # *distance*, so a miss coded as level 0 takes the maximum penalty.
    ordinal = batch['val_data']['ordinal']['values'][0]
    assert (ordinal.sum(dim=-1) > 0)[0].tolist() == [
        False, False, True, True
    ]


# --- time (section 4.2) ----------------------------------------------

def test_minutes_become_days(dataset):
    """Section 4.2: minutes are canonical on disk and days are what the
    model gets, 'where magnitudes are small and the THP wants a sanely
    scaled input'."""
    item = dataset[0]
    minutes = np.asarray(dataset.val_times[0])
    assert item['val_times'].dtype == torch.float32
    np.testing.assert_allclose(
        item['val_times'].numpy(), minutes / 1440.0, rtol=0, atol=0
    )
    # one_patient's timesteps are one day apart, ending at the origin.
    assert item['val_times'].tolist() == [-3.0, -2.0, -1.0, 0.0]
    assert item['event_times'].dtype == torch.float32


def test_the_index_time_is_the_final_column(dataset):
    """Section 4.2: t = 0, the index time and the last record all sit in
    the final column. Nothing here normalizes that to left-aligned."""
    item = dataset[0]
    assert item['val_times'][-1] == 0.0
    assert item['val_masks'][-1] == 1.0
    assert (item['val_times'] <= 0).all()


def test_time_to_event_stays_in_minutes(dataset):
    """A target, not a record timestamp. Section 4.2's conversion is
    about model input; the target's scaling belongs with the prediction
    head, which is section 8 item 3 and still open."""
    item = dataset[0]
    assert item['time_to_event'].item() == pytest.approx(1000.0)
    assert item['time_to_event'].dtype == torch.float32
    assert item['event_type'].dtype == torch.long


# --- fold standardization (section 5) --------------------------------

def test_fold_statistics_are_applied_from_the_npz(one_patient):
    """Section 5 moves standardization out of extraction and into
    __getitem__, reading ``summary_statistics_fold{i}.npz``. The rule is
    the pre-C1 one verbatim: subtract the mean, divide by the p95-p5
    span, over the whole row."""
    dataset = extracted(one_patient)
    stats = np.load(one_patient.extracted / 'summary_statistics_fold0.npz')
    span = stats['p95'][0] - stats['p5'][0]
    assert span > 0

    raw = np.asarray(dataset.val_numeric_values[0][0], dtype=np.float32)
    expected = (raw - stats['means'][0]) / span
    np.testing.assert_array_equal(
        dataset[0]['val_numeric_values'][0].numpy(), expected
    )


def test_scaling_reaches_unobserved_timesteps(one_patient):
    """The same rule's second surprising half, preserved deliberately:
    the pre-C1 code scaled the array in place, so an unobserved or padded
    position became ``-mean / span`` rather than staying at zero."""
    dataset = extracted(one_patient)
    item = dataset[0]
    # one_patient's first timestep has no NUM value.
    assert item['val_numeric_indicators'][0, 0] == 0.0
    assert item['val_numeric_values'][0][0].item() != 0.0


def test_a_degenerate_span_zeroes_the_feature(mini):
    """'If p5 == p95, the feature is zeroed' -- one observed value makes
    the two percentiles coincide, and dividing by the span would be a
    division by zero."""
    mini.add_patient(
        1001,
        timeseries=[
            ['2019-01-01T00:00:00Z', '', '', '', '', '', ''],
            ['2019-01-02T00:00:00Z', 7.0, 'L', '0', 'Few', '', ''],
        ],
        stays=[('DAD', '2019-01-01T00:00:00Z', '2019-01-02T00:00:00Z')],
    )
    dataset = extracted(mini)
    stats = np.load(mini.extracted / 'summary_statistics_fold0.npz')
    assert stats['p5'][0] == stats['p95'][0]
    assert not dataset[0]['val_numeric_values'][0].any()


def test_no_fold_returns_the_values_unscaled(one_patient):
    """``fold=None`` is for inspection and for tests. It must not be a
    silent no-op that a training run could reach by accident, hence the
    argument being required rather than defaulted."""
    one_patient.add_fold('fold0', train=[0])
    assert run(one_patient) == 0
    write_tables(one_patient)
    dataset = load_dataset(str(one_patient.extracted), fold=None)
    np.testing.assert_array_equal(
        dataset[0]['val_numeric_values'][0].numpy(),
        np.asarray(dataset.val_numeric_values[0][0], dtype=np.float32)
    )


def test_the_named_fold_is_the_one_applied(one_patient):
    """Two folds' statistics live side by side in the same directory, so
    the fold name has to select between them rather than the directory
    doing it."""
    one_patient.add_fold('fold0', train=[0])
    assert run(one_patient) == 0
    write_tables(one_patient)
    np.savez(
        one_patient.extracted / 'summary_statistics_fold1.npz',
        means=np.zeros(1, np.float32), p5=np.zeros(1, np.float32),
        p95=np.full(1, 2.0, np.float32)
    )
    raw = np.asarray(
        np.load(one_patient.extracted / 'val_numeric_values_0.npy')[0],
        dtype=np.float32
    )
    fold1 = load_dataset(str(one_patient.extracted), fold='fold1')
    np.testing.assert_array_equal(
        fold1[0]['val_numeric_values'][0].numpy(), raw / 2.0
    )


def test_a_missing_fold_npz_is_refused(one_patient):
    """Standardization is per fold and computed over that fold's
    *training* rows alone; silently returning raw values would leak
    nothing but would feed a model inputs it was not trained on."""
    one_patient.add_fold('fold0', train=[0])
    assert run(one_patient) == 0
    write_tables(one_patient)
    with pytest.raises(FileNotFoundError, match='summary_statistics_fold3'):
        load_dataset(str(one_patient.extracted), fold='fold3')


# --- the lookup family's gather (sections 4.3, 4.4, 5.1) -------------

def test_text_gathers_the_table_row_of_each_timestep_that_has_one(dataset):
    """Section 4.4: ``text_values`` is an int32 row index into the global
    table, and __getitem__ is the only consumer of the CSR arrays. The
    item carries one entry per timestep that holds a record rather than
    the whole timestep axis (section 5.1) -- a single-slot feature's
    entry is (D,), its weight being 1 by definition, so there is nothing
    to slot."""
    item = dataset[0]
    # TXT is lookup feature 0; DRG follows it (section 4.3's ordering,
    # ``TEXT_FEATS + DRUG_FEATS``).
    text = item['val_lookup_sparse'][0]
    # one_patient carries the same note at two timesteps, interned once,
    # so both gather table row 0, whose fill value is 1.
    assert text['timestep_index'].tolist() == [1, 3]
    assert text['values'].shape == (2, TEXT_EMBED_DIM)
    assert text['values'][0].tolist() == [1.0] * TEXT_EMBED_DIM
    assert text['values'][1].tolist() == [1.0] * TEXT_EMBED_DIM
    # The timestep with no text gathers nothing, which under the sparse
    # form means it is simply not named.
    assert 2 not in text['timestep_index'].tolist()
    # One feature axis over the whole family, whatever the on-disk
    # split by type (section 5.1). A single-slot feature carries no
    # dose or mask array: its weight is 1 by definition.
    assert item['val_lookup_indicators'].shape == (4, 2)
    assert text['doses'] is None
    assert text['masks'] is None


def test_drugs_gather_slotted_and_unpooled(dataset):
    """Sections 4.3 and 5.1: 'the pool is part of the forward pass, not
    of __getitem__'. A timestep's drugs arrive as (slots, D) with their
    doses and masks beside them, so a gradient still reaches an
    individual slot and therefore an individual DIN."""
    item = dataset[0]
    drug = item['val_lookup_sparse'][1]
    # one_patient dispenses at its first timestep and nowhere else.
    assert drug['timestep_index'].tolist() == [0]
    assert drug['values'].shape == (1, 3, CLINVEC_DIM)
    assert drug['doses'].shape == (1, 3) and drug['masks'].shape == (1, 3)

    # ClinVec rows 2 and 3, at doses 1.0 and 0.5; the third slot is
    # unused.
    assert drug['values'][0, 0].tolist() == [3.0] * CLINVEC_DIM
    assert drug['values'][0, 1].tolist() == [4.0] * CLINVEC_DIM
    assert drug['doses'][0].tolist() == [1.0, 0.5, 0.0]
    assert drug['masks'][0].tolist() == [1.0, 1.0, 0.0]
    # No mean anywhere: an unpooled slot tensor still names its slots.
    assert drug['values'][0, 0].tolist() != drug['values'][0, 1].tolist()


def test_an_unused_slot_gathers_the_all_zero_pad_row(dataset):
    """Section 4.4 pads with V, the final row of a (V+1, D) table, and
    that row is all zero -- so a padded slot contributes nothing whether
    or not the mask is applied."""
    item = dataset[0]
    assert not item['val_lookup_sparse'][1]['values'][0, 2].any()


def test_the_gather_casts_to_float32(one_patient):
    """Section 5.1: 'cast at the gather, not mid-model'. torch.cat wants
    one dtype across its inputs, and a mismatch introduced later
    surfaces deep in the encoder instead of at the boundary."""
    dataset = extracted(one_patient, dtype=np.float64)
    assert dataset.lookup_tables[0].dtype == np.float64
    item = dataset[0]
    assert item['val_lookup_sparse'][0]['values'].dtype == torch.float32
    assert item['val_lookup_sparse'][1]['values'].dtype == torch.float32


def test_a_missing_table_is_refused(one_patient):
    """Both tables are C4's. A text feature has no declared width until
    then, so there is no shape to fall back to, and dropping the feature
    would train a USE_TEXT model on no text at all."""
    one_patient.add_fold('fold0', train=[0])
    assert run(one_patient) == 0
    with pytest.raises(FileNotFoundError, match='text_embeddings.npy'):
        load_dataset(str(one_patient.extracted), fold='fold0')


def test_a_stale_text_table_is_refused(one_patient):
    """Extraction re-interns the strings on every run (section 4.4), so
    a table built against an earlier extraction indexes a vocabulary
    that no longer exists. The row count is what ties the two."""
    one_patient.add_fold('fold0', train=[0])
    assert run(one_patient) == 0
    write_tables(one_patient)
    np.save(one_patient.extracted / 'text_embeddings.npy',
            np.zeros((99, TEXT_EMBED_DIM), dtype=np.float32))
    with pytest.raises(ValueError, match='text_strings.pkl'):
        load_dataset(str(one_patient.extracted), fold='fold0')


def test_a_drug_table_must_have_the_pad_row(one_patient):
    """Section 4.4: the table is (V+1, D) where V is the vocabulary's row
    count, not the cohort's. A table of exactly V rows would leave the
    pad index colliding with a real drug."""
    one_patient.add_fold('fold0', train=[0])
    assert run(one_patient) == 0
    write_tables(one_patient)
    np.save(one_patient.extracted / 'drug_embeddings.npy',
            np.zeros((CLINVEC_ROWS, CLINVEC_DIM), dtype=np.float32))
    with pytest.raises(ValueError, match='pads unused slots'):
        load_dataset(str(one_patient.extracted), fold='fold0')


# --- the collated batch ----------------------------------------------

def test_the_batch_collates_with_the_expected_shapes(dataset):
    """The model's input format is unchanged by any of this: one
    indicator tensor per family, a per-feature list of values, and the
    stacked (B, T, n_feats, D) text tensor section 5.1 replaces in C3."""
    batch = collate_for_model([dataset[0], dataset[0]])
    val = batch['val_data']

    assert val['times'].shape == (2, 4)
    assert val['masks'].shape == (2, 4)
    assert val['numeric']['indicators'].shape == (2, 4, 1)
    assert [v.shape for v in val['numeric']['values']] == [(2, 4, 1)]
    assert val['categorical']['indicators'].shape == (2, 4, 1)
    assert [v.shape for v in val['categorical']['values']] == [(2, 4, 2)]
    assert val['ordinal']['indicators'].shape == (2, 4, 2)
    assert [v.shape for v in val['ordinal']['values']] == [
        (2, 4, 3), (2, 4, 2)
    ]
    assert val['lookup']['indicators'].shape == (2, 4, 2)
    assert [v.shape for v in val['lookup']['slot_values']] == [
        (2, 4, TEXT_EMBED_DIM), (2, 4, 3, CLINVEC_DIM)
    ]
    assert [None if v is None else v.shape
            for v in val['lookup']['doses']] == [None, (2, 4, 3)]
    assert [None if v is None else v.shape
            for v in val['lookup']['masks']] == [None, (2, 4, 3)]
    # Unpooled: the dose-weighted mean is part of the forward pass.
    assert 'embedded_values' not in val['lookup']
    assert batch['event_data']['indicators'].shape == (2, 4, 1)
    assert batch['targets']['time_to_event'].shape == (2, 1)
    assert batch['targets']['event_type'].shape == (2,)


def test_every_collated_value_is_float32(dataset):
    """Everything that reaches ``torch.cat`` in the encoder has to agree
    on dtype, and the one-hot expansion and the gather are where the
    int16 indices and the table rows stop being integers."""
    batch = collate_for_model([dataset[0], dataset[0]])
    val = batch['val_data']
    tensors = (
        [val['times'], val['masks']] + val['lookup']['slot_values']
        + val['numeric']['values'] + val['categorical']['values']
        + val['ordinal']['values']
        + [v for v in val['lookup']['doses'] if v is not None]
        + [v for v in val['lookup']['masks'] if v is not None]
    )
    assert {t.dtype for t in tensors} == {torch.float32}


def test_static_data_is_absent_from_the_item_and_the_batch(dataset):
    """Section A.3 empties ``STATIC_FEATS`` and section 4.4 omits the
    zero-width file, so ``datasets.py:190``'s unconditional index had
    nothing to index. ``models.py:339`` and ``:521`` both already read
    the key defensively, so absence is the case they handle -- a
    ``(batch, 0)`` tensor is not."""
    assert dataset.static_data is None
    assert 'static_data' not in dataset[0]
    assert 'static_data' not in collate_tensorized([dataset[0]])


def test_the_episode_id_reaches_the_batch(dataset):
    """Row -> (PATID, STAY_INDEX), for mapping an XAI score back to the
    episode it came from."""
    assert dataset[0]['episode_id'] == (1001, 0)
    batch = collate_tensorized([dataset[0]])
    assert batch['episode_id'].tolist() == [[1001, 0]]


# --- the loaders (sections 2, 3) -------------------------------------

@pytest.fixture
def cohort(mini):
    """Three patients, one episode each, so partitions can differ."""
    for patid in (1001, 1002, 1003):
        mini.add_patient(
            patid,
            timeseries=[
                ['2019-01-01T00:00:00Z', 1.0, 'L', '0', 'Few', 'a', ''],
                ['2019-01-02T00:00:00Z', 2.0, 'U', '1-24', 'None', '', ''],
            ],
            stays=[('DAD', '2019-01-01T00:00:00Z',
                    '2019-01-02T00:00:00Z')],
        )
    mini.add_fold('fold0', train=[0], val=[1], test=[2])
    assert run(mini) == 0
    write_tables(mini)
    return mini


def test_prepare_dataloaders_partitions_by_row_index(cohort):
    """Section 2's second decision: a fold is row indices into one
    cohort-wide set of arrays, so the three partitions are Subsets of
    one dataset rather than three directories."""
    train, val, test = prepare_dataloaders(
        str(cohort.data_dir), 'fold0', batch_size=1, num_workers=0
    )
    assert [len(loader.dataset) for loader in (train, val, test)] == [1, 1, 1]
    # One dataset, three views of it.
    assert train.dataset.dataset is val.dataset.dataset


def test_idx_is_the_cohort_row(cohort):
    """Which is what makes ``episode_ids`` and any XAI score resolve
    against ``labels.csv`` without a second mapping."""
    _, _, test = prepare_dataloaders(
        str(cohort.data_dir), 'fold0', batch_size=1, num_workers=0
    )
    batch = next(iter(test))
    assert batch['idx'].tolist() == [2]
    assert batch['episode_id'].tolist() == [[1003, 0]]


def test_a_fold_without_val_rows_yields_two_loaders(mini):
    """Section 3 carves val out of train, but a fold written without it
    is a two-partition fold rather than an error."""
    mini.add_patient(
        1001,
        timeseries=[['2019-01-01T00:00:00Z', 1.0, 'L', '0', 'Few', '', '']],
        stays=[('DAD', '2019-01-01T00:00:00Z', '2019-01-01T00:00:00Z')],
    )
    mini.add_fold('fold0', train=[0], test=[0])
    assert run(mini) == 0
    write_tables(mini)
    loaders = prepare_dataloaders(
        str(mini.data_dir), 'fold0', batch_size=1, num_workers=0
    )
    assert len(loaders) == 2


def test_rows_past_the_cohort_are_refused(cohort):
    """A stale fold directory indexes rows that no longer exist.
    Extraction already refuses one up front; so does loading."""
    np.save(
        cohort.data_dir / 'fold0' / 'fold0_test_rows.npy',
        np.array([99], dtype=np.int64)
    )
    with pytest.raises(ValueError, match='has 3 episode'):
        prepare_dataloaders(
            str(cohort.data_dir), 'fold0', batch_size=1, num_workers=0
        )


def test_a_missing_train_row_array_is_an_error(cohort):
    """Unlike val, train and test are not optional."""
    (cohort.data_dir / 'fold0' / 'fold0_train_rows.npy').unlink()
    with pytest.raises(FileNotFoundError, match='fold0_train_rows.npy'):
        prepare_dataloaders(
            str(cohort.data_dir), 'fold0', batch_size=1, num_workers=0
        )


def test_the_text_counts_are_counted_over_text_features_only(cohort):
    """Text balancing exists to stop one rank receiving every text-heavy
    episode, and it is the LLM-width embeddings that make an episode
    heavy -- not the 128-wide drug ones. Both counters agree."""
    dataset = load_dataset(str(cohort.extracted), fold='fold0')
    counts = get_text_counts_from_dataset_vectorized(dataset)
    np.testing.assert_array_equal(
        counts, get_text_counts_from_dataset(dataset)
    )
    # One text entry per episode, and three episodes.
    assert counts.tolist() == [1, 1, 1]


def test_a_batch_survives_a_dataloader(cohort):
    """End to end through the collate function the loaders install."""
    train, _, _ = prepare_dataloaders(
        str(cohort.data_dir), 'fold0', batch_size=1, num_workers=0
    )
    batch = next(iter(train))
    assert batch['val_data']['numeric']['values'][0].shape == (1, 4, 1)


# --- memory mapping across process boundaries ------------------------

def test_the_dataset_pickles_without_copying_its_arrays(dataset):
    """The reason ``LazyArray`` exists: a ``np.memmap`` pickles *by
    value*, and the loaders start their workers with ``spawn``, so open
    memmaps in the dataset would give every worker a private copy of
    every array. Section 4.5 puts the cohort at ~75 GB."""
    blob = pickle.dumps(dataset, protocol=pickle.HIGHEST_PROTOCOL)
    # A path and a little metadata per array. An open memmap would put
    # every byte of every array in here instead.
    # A path and a little metadata per array, plus the fold's three
    # statistics vectors. An open memmap would put every byte of every
    # array in here instead.
    assert len(blob) < 20_000


def test_an_unpickled_array_is_a_real_mapping(dataset):
    """And the point of not copying: the array each process opens is
    backed by the file, so the page cache is the shared copy."""
    revived = pickle.loads(pickle.dumps(dataset))
    array = revived.val_times.array
    assert isinstance(array, np.memmap)
    assert array.filename is not None
    np.testing.assert_array_equal(array, np.asarray(dataset.val_times))


def test_a_lazy_array_behaves_like_the_array(tmp_path):
    """The surface ``MixedDataset`` actually uses."""
    path = tmp_path / 'a.npy'
    np.save(path, np.arange(6, dtype=np.int16).reshape(3, 2))
    lazy = LazyArray(str(path))
    assert lazy.shape == (3, 2)
    assert lazy.dtype == np.int16
    assert len(lazy) == 3
    assert lazy[1].tolist() == [2, 3]


# --- the model's input format (section 5) ----------------------------

def test_a_batch_is_accepted_by_the_generator_unchanged(dataset):
    """Section 5's claim about all of the above: "the model's input
    format is unchanged by all of this, and neither is losses.py".

    The oracle is the real encoder rather than an assertion about
    shapes: a collated batch goes through ``generate_record_masks`` and
    ``MaskedTokenGenerator`` -- so through ``torch.cat`` over the
    families, ``combine_value_and_lookup_data``, and the timestamp
    encoding -- and every head produces its feature's width back.
    ``norm`` and ``predict_indicators`` are what all four experiment
    configs set.
    """
    from TransEHR2.modules import MaskedTokenGenerator, ValueDataEncoder
    from TransEHR2.utils import generate_record_masks

    batch = collate_for_model([dataset[0], dataset[0]])
    val_data = batch['val_data']
    lookup_dims = [v.shape[-1] for v in val_data['lookup']['slot_values']]

    numeric_dims = [v.shape[-1] for v in val_data['numeric']['values']]
    categorical_classes = [
        v.shape[-1] for v in val_data['categorical']['values']
    ]
    ordinal_levels = [v.shape[-1] for v in val_data['ordinal']['values']]
    n_features = (
        len(numeric_dims) + len(categorical_classes) + len(ordinal_levels)
        + val_data['lookup']['indicators'].shape[-1]
    )
    feat_dim = (
        sum(numeric_dims) + sum(categorical_classes) + sum(ordinal_levels)
        + sum(lookup_dims)
    )

    torch.manual_seed(0)
    generator = MaskedTokenGenerator(
        encoder=ValueDataEncoder(
            n_features=n_features, feat_dim=feat_dim, d_model=16,
            n_heads=2, n_encoder_blocks=1, dim_feedforward=32,
            dropout=0.0, norm='LayerNorm'
        ),
        d_model=16,
        numeric_dims=numeric_dims,
        categorical_classes=categorical_classes,
        ordinal_features=ordinal_levels,
        lookup_dims=lookup_dims,
        predict_indicators=False,
        dim_feedforward=32
    )
    record_masks, _ = generate_record_masks(batch)
    output = generator(val_data, record_masks)

    T = dataset.max_ts_len
    assert [v.shape for v in output['numeric']['values']] == [
        (2, T, dim) for dim in numeric_dims
    ]
    assert [v.shape for v in output['categorical']['values']] == [
        (2, T, n) for n in categorical_classes
    ]
    assert [v.shape for v in output['ordinal']['values']] == [
        (2, T, n) for n in ordinal_levels
    ]
    assert [v.shape for v in output['lookup']['embedded_values']] == [
        (2, T, dim) for dim in lookup_dims
    ]
