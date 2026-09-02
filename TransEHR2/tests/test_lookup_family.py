"""Section 5.1's regression gate, and one test per claim C3 makes.

The gate is the hard condition the section states: **with
``DRUG_FEATS: []``, the refactored model must produce outputs
bitwise-identical to the current one on a fixed batch and seed.**
``fixtures/lookup_family_baseline.npz`` holds those outputs, captured
from the tree as it stood before the refactor, and the ``gate`` fixture
below rebuilds their inputs deterministically -- the fixture root, the
collated batch, the record masks, and every module's initialization
order. Nothing here may be re-recorded to make a
failure go away: the point of the gate is that a text regression and a
drug bug are otherwise indistinguishable.

The fixture is ``conftest``'s, minus the two features that are not about
the lookup family: ``DRG``, because the gate is the no-drug case, and
``UB``, because dlordinal's ``BetaLoss`` needs three levels and it has
two.
"""

import numpy as np
import pickle
import pytest
import torch

from pathlib import Path

from TransEHR2.data.preprocessing import collate_tensorized, load_dataset
from TransEHR2.losses import MaskedDiscriminatorLoss, MaskedGeneratorLoss
from TransEHR2.models import ELECTRA, MixedClassifier
from TransEHR2.modules import (
    EventDataEncoder, MaskedTokenDiscriminator, MaskedTokenGenerator,
    TransformerHawkesProcess, ValueDataEncoder
)
from TransEHR2.utils import generate_record_masks

from extract_data import main as extract_main

from .conftest import MiniRoot

BASELINE = Path(__file__).parent / 'fixtures' / 'lookup_family_baseline.npz'

# Deliberately not ClinVec's 128: section 5.1 makes the embedding width
# per-feature precisely because they differ.
TEXT_EMBED_DIM = 5
D_MODEL = 16


def build_root(tmp_path):
    """``conftest``'s fixture with ``DRUG_FEATS: []`` (the gate's case)."""
    mini = MiniRoot(Path(tmp_path))
    mini.config = dict(mini.config)
    mini.config['DRUG_FEATS'] = []
    mini.config['VALUED_FEATS'] = ['NUM', 'CAT', 'ORD']
    del mini.var_properties['DRG']
    del mini.var_properties['UB']
    mini.add_patient(
        1001,
        timeseries=[
            ['2019-01-01T00:00:00Z', '', '', '', '', '', ''],
            ['2019-01-02T00:00:00Z', 1.5, 'L', '0', '', 'a note', ''],
            ['2019-01-03T00:00:00Z', 2.5, 'U', '25-50', '', '', 1],
            ['2019-01-04T00:00:00Z', 3.5, 'L', '1-24', '', 'other note', ''],
        ],
        stays=[('DAD', '2019-01-01T00:00:00Z', '2019-01-04T00:00:00Z')],
    )
    mini.add_patient(
        1002,
        timeseries=[
            ['2019-02-01T00:00:00Z', 0.5, 'U', '1-24', '', 'a note', 1],
            ['2019-02-02T00:00:00Z', 2.0, 'L', '25-50', '', '', ''],
            ['2019-02-03T00:00:00Z', 4.0, 'U', '0', '', 'third note', 1],
        ],
        stays=[('AMB', '2019-02-01T00:00:00Z', '2019-02-03T00:00:00Z')],
    )
    mini.add_fold('fold0', train=[0, 1], val=[0, 1], test=[0, 1])
    return mini


def extract_and_load(mini):
    """Extract, write C4's lookup tables, and load the result."""
    from .conftest import CLINVEC_DIM, CLINVEC_ROWS

    config_path = mini.finish()
    assert extract_main([str(config_path)]) == 0

    with open(mini.extracted / 'text_strings.pkl', 'rb') as f:
        n_strings = len(pickle.load(f))
    # Row r is filled with r + 1, so a gathered vector names its row.
    table = np.repeat(
        np.arange(1, n_strings + 1, dtype=np.float32)[:, None],
        TEXT_EMBED_DIM, axis=1
    )
    np.save(mini.extracted / 'text_embeddings.npy', table)

    if mini.config['DRUG_FEATS']:
        drug = np.repeat(
            np.arange(1, CLINVEC_ROWS + 1, dtype=np.float32)[:, None],
            CLINVEC_DIM, axis=1
        )
        # The final all-zero row is the pad unused slots index.
        drug = np.vstack([drug, np.zeros((1, CLINVEC_DIM), dtype=np.float32)])
        np.save(mini.extracted / 'drug_embeddings.npy', drug)

    return load_dataset(str(mini.extracted), fold='fold0')


def encoder_dims(val_data):
    """The encoder's two widths, summed over every feature family."""
    lookup_dims = [v.shape[-1] for v in val_data['lookup']['slot_values']]
    numeric = [v.shape[-1] for v in val_data['numeric']['values']]
    categorical = [v.shape[-1] for v in val_data['categorical']['values']]
    ordinal = [v.shape[-1] for v in val_data['ordinal']['values']]
    multilabel = [v.shape[-1] for v in val_data['multilabel']['values']]
    n_features = (len(numeric) + len(categorical) + len(ordinal)
                  + len(multilabel) + len(lookup_dims))
    feat_dim = (sum(numeric) + sum(categorical) + sum(ordinal)
                + sum(multilabel) + sum(lookup_dims))
    return {
        'lookup_dims': lookup_dims, 'numeric': numeric,
        'categorical': categorical, 'ordinal': ordinal,
        'multilabel': multilabel, 'n_features': n_features,
        'feat_dim': feat_dim,
    }


def value_encoder(dims):
    return ValueDataEncoder(
        n_features=dims['n_features'], feat_dim=dims['feat_dim'],
        d_model=D_MODEL, n_heads=2, n_encoder_blocks=1,
        dim_feedforward=32, dropout=0.0, norm='LayerNorm'
    )


def event_encoder(n_event_types):
    return EventDataEncoder(
        num_types=n_event_types, d_model=16, d_inner=32, n_layers=1,
        n_head=2, d_k=8, d_v=8, dropout=0.0
    )


def build_electra(dims, n_event_types, n_lookup):
    """Seeded exactly as the baseline capture seeded it."""
    torch.manual_seed(1)
    generator = MaskedTokenGenerator(
        encoder=value_encoder(dims),
        d_model=D_MODEL,
        numeric_dims=dims['numeric'],
        categorical_classes=dims['categorical'],
        ordinal_features=dims['ordinal'] or None,
        multilabel_classes=dims['multilabel'] or None,
        lookup_dims=dims['lookup_dims'],
        # What all four experiment configs set.
        predict_indicators=False,
        dim_feedforward=32
    )
    discriminator = MaskedTokenDiscriminator(
        encoder=value_encoder(dims),
        d_model=D_MODEL,
        n_numeric_features=len(dims['numeric']),
        n_categorical_features=len(dims['categorical']),
        n_ordinal_features=len(dims['ordinal']),
        n_multilabel_features=len(dims['multilabel']),
        n_lookup_features=n_lookup,
        n_static_features=0,
        dim_feedforward=32
    )
    hawkes = TransformerHawkesProcess(
        encoder=event_encoder(n_event_types), num_types=n_event_types
    )
    electra = ELECTRA(generator, discriminator, hawkes, use_lookup=True)
    electra.eval()
    return electra


def build_classifier(dims, n_event_types):
    torch.manual_seed(3)
    classifier = MixedClassifier(
        event_encoder=event_encoder(n_event_types),
        val_encoder=value_encoder(dims),
        d_event_enc=16, d_val_enc=D_MODEL, d_statics=0, num_classes=2,
        aggr='max', use_lookup=True
    )
    classifier.eval()
    return classifier


def flatten(prefix, obj, into):
    """Name every tensor in a nested output structure, in a fixed order."""
    if isinstance(obj, torch.Tensor):
        into[prefix] = obj.detach()
    elif isinstance(obj, dict):
        for key in sorted(obj):
            flatten(f'{prefix}.{key}', obj[key], into)
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            flatten(f'{prefix}[{i}]', value, into)
    return into


@pytest.fixture
def gate(tmp_path):
    """The batch, the masks and the two models the baseline was taken on.

    The record masks are drawn at a higher rate than training uses so
    that a text record is actually masked: at 0.15 over this cohort's
    handful of observed text positions, ``int(0.15 * n)`` is zero and
    the cosine-distance branch never runs -- a gate that exercises
    nothing.
    """
    dataset = extract_and_load(build_root(tmp_path))
    batch = collate_tensorized([dataset[0], dataset[1]])
    dims = encoder_dims(batch['val_data'])
    n_lookup = batch['val_data']['lookup']['indicators'].shape[-1]
    n_event_types = batch['event_data']['indicators'].shape[-1]

    torch.manual_seed(0)
    record_masks, event_masks = generate_record_masks(
        batch, feature_sample_rate=0.6, obs_unobs_ratio=2.0,
        subsample_rate=0.5
    )
    return {
        'dataset': dataset, 'batch': batch, 'dims': dims,
        'record_masks': record_masks, 'event_masks': event_masks,
        'electra': build_electra(dims, n_event_types, n_lookup),
        'classifier': build_classifier(dims, n_event_types),
        'n_event_types': n_event_types,
    }


# --- the regression gate (section 5.1) -------------------------------

def test_the_refactored_path_reproduces_the_baseline_bitwise(gate, tmp_path):
    """Section 5.1's hard condition. Every tensor the family's rename
    and un-stacking could have moved -- the record masks, the
    generator's and discriminator's outputs, the masked targets, both
    pretraining losses and the finetuning logits -- against the same
    quantity from the pre-refactor tree.

    ``torch.equal``, not ``allclose``: the section asks for bitwise, and
    the two formulations are meant to be the same arithmetic in the same
    order. A concatenation of per-feature tensors reproduces a stacked
    tensor's ``flatten(start_dim=2)`` exactly, and an elementwise
    multiply per feature reproduces one over the stack.
    """
    current = {}
    out = gate['electra'](
        gate['batch'], gate['record_masks'], device='cpu',
        compute_intensities=False
    )
    flatten('masks', gate['record_masks'], current)
    current['event_masks'] = gate['event_masks']
    flatten('gen', out['generator'], current)
    flatten('disc', out['discriminator'], current)
    flatten('tgt', out['masked_targets'], current)

    gen_loss_fn = MaskedGeneratorLoss(
        lookup_weight=0.7, ordinal_features=gate['dims']['ordinal'] or None
    )
    current['gen_loss'] = gen_loss_fn(
        out['generator'], out['masked_targets'], gate['record_masks']
    ).detach()
    current['disc_loss'] = MaskedDiscriminatorLoss()(
        out['discriminator'], gate['record_masks']
    ).detach()

    # A fresh batch: the ELECTRA forward mutates its own in place.
    dataset = extract_and_load(build_root(tmp_path / 'clf'))
    clf_batch = collate_tensorized([dataset[0], dataset[1]])
    current['classifier_logits'] = gate['classifier'](clf_batch).detach()

    baseline = np.load(BASELINE)
    assert sorted(baseline.files) == sorted(current)
    for name in sorted(baseline.files):
        expected = torch.from_numpy(baseline[name])
        actual = current[name]
        assert actual.shape == expected.shape, name
        assert torch.equal(actual, expected), (
            f"{name}: max |delta| = "
            f"{(actual - expected).abs().max().item():.3e}"
        )


def test_the_gate_masks_a_text_record(gate):
    """The gate above is only worth running if the cosine-distance
    branch runs, which needs a masked text position whose target is not
    the zero vector -- ``losses.py`` skips those to keep ``F.normalize``
    off a NaN."""
    assert gate['record_masks']['lookup']['indicators'].sum() > 0
    out = gate['electra'](
        gate['batch'], gate['record_masks'], device='cpu',
        compute_intensities=False
    )
    targets = out['masked_targets']['lookup']['embedded_values'][0]
    assert targets.shape[0] > 0
    assert torch.norm(targets, p=2, dim=-1).min() > 1e-8


# --- the shape of the generalization (section 5.1) -------------------

def test_the_family_reaches_the_model_as_a_per_feature_list(gate):
    """Section 5.1: 'replace the stacked tensor with a list of
    (B, T, D_f), one per lookup feature'. A stacked
    ``(B, T, n_feats, D)`` forces one shared width, which is the one
    thing that genuinely blocks the generalization."""
    from TransEHR2.utils import resolve_lookup_embeddings

    lookup = gate['batch']['val_data']['lookup']
    assert isinstance(lookup['slot_values'], list)
    embeddings = resolve_lookup_embeddings(lookup)
    assert [v.shape for v in embeddings] == [(2, 4, TEXT_EMBED_DIM)]
    assert lookup['indicators'].shape == (2, 4, 1)


def test_ragged_widths_reach_the_encoder(gate):
    """The claim the list exists for: two lookup features of different
    widths concatenate into one encoder input, which
    ``torch.stack(..., dim=2)`` could not have represented at all.

    The second feature is synthesised rather than extracted -- C4 builds
    the drug table, and this is about the model path, not the store.
    """
    from TransEHR2.utils import combine_value_and_lookup_data

    from TransEHR2.utils import resolve_lookup_embeddings

    batch, dims = gate['batch'], gate['dims']
    lookup = batch['val_data']['lookup']
    narrow = resolve_lookup_embeddings(lookup)[0]
    wide = torch.arange(
        2 * 4 * 3, dtype=torch.float32
    ).reshape(2, 4, 3)

    indicators, values = combine_value_and_lookup_data(
        value_assoc_indicators=batch['val_data']['numeric']['indicators'],
        value_assoc_values=batch['val_data']['numeric']['values'][0],
        lookup_assoc_indicators=torch.cat(
            [lookup['indicators'], lookup['indicators']], dim=-1
        ),
        lookup_embeddings=[narrow, wide]
    )
    assert indicators.shape == (2, 4, len(dims['numeric']) + 2)
    assert values.shape == (2, 4, dims['numeric'][0] + TEXT_EMBED_DIM + 3)
    # Feature-major, the layout flatten(start_dim=2) produced.
    assert torch.equal(values[..., -3:], wide)
    assert torch.equal(values[..., -3 - TEXT_EMBED_DIM:-3], narrow)


def test_one_head_per_lookup_feature_at_its_own_width(gate):
    """Section 5.1's table: ``ModuleList([Linear(d_model, text_embed_dim)]
    * n_text)`` becomes one ``Linear(d_model, D_f)`` per lookup feature,
    ``D_f`` per-feature."""
    generator = MaskedTokenGenerator(
        encoder=value_encoder(gate['dims']),
        d_model=D_MODEL,
        numeric_dims=gate['dims']['numeric'],
        categorical_classes=gate['dims']['categorical'],
        lookup_dims=[4096, 128],
        predict_indicators=False,
        dim_feedforward=32
    )
    assert [head.out_features for head in generator.lookup_heads] == [
        4096, 128
    ]
    assert generator.predict_lookup_feats


# --- the drug feature joins the family (sections 4.3, 5.1) -----------

@pytest.fixture
def drug_gate(tmp_path):
    """``conftest``'s fixture unchanged, so both members are present.

    ``TXT`` is one unweighted slot of width 5 and ``DRG`` is three
    dose-weighted slots of width 2, which is the ragged case the family
    exists for.
    """
    from .conftest import CLINVEC_DIM, CLINVEC_ROWS, MiniRoot

    mini = MiniRoot(Path(tmp_path))
    mini.config = dict(mini.config)
    mini.config['VALUED_FEATS'] = ['NUM', 'CAT', 'ORD']
    del mini.var_properties['UB']
    mini.add_patient(
        1001,
        timeseries=[
            ['2019-01-01T00:00:00Z', '', '', '', '', '', ''],
            ['2019-01-02T00:00:00Z', 1.5, 'L', '0', '', 'a note', ''],
            ['2019-01-03T00:00:00Z', 2.5, 'U', '25-50', '', '', 1],
            ['2019-01-04T00:00:00Z', 3.5, 'L', '1-24', '', 'other note', ''],
        ],
        stays=[('DAD', '2019-01-01T00:00:00Z', '2019-01-04T00:00:00Z')],
        drugs=[('2019-01-01T00:00:00Z', 0, 2, 1.0),
               ('2019-01-01T00:00:00Z', 1, 3, 0.5)],
    )
    mini.add_patient(
        1002,
        timeseries=[
            ['2019-02-01T00:00:00Z', 0.5, 'U', '1-24', '', 'a note', 1],
            ['2019-02-02T00:00:00Z', 2.0, 'L', '25-50', '', '', ''],
            ['2019-02-03T00:00:00Z', 4.0, 'U', '0', '', 'third note', 1],
        ],
        stays=[('AMB', '2019-02-01T00:00:00Z', '2019-02-03T00:00:00Z')],
        drugs=[('2019-02-02T00:00:00Z', 0, 1, 2.0)],
    )
    mini.add_fold('fold0', train=[0, 1], val=[0, 1], test=[0, 1])

    dataset = extract_and_load(mini)
    batch = collate_tensorized([dataset[0], dataset[1]])
    dims = encoder_dims(batch['val_data'])
    n_lookup = batch['val_data']['lookup']['indicators'].shape[-1]
    n_event_types = batch['event_data']['indicators'].shape[-1]
    torch.manual_seed(0)
    record_masks, _ = generate_record_masks(
        batch, feature_sample_rate=0.6, obs_unobs_ratio=2.0,
        subsample_rate=0.5
    )
    return {
        'batch': batch, 'dims': dims, 'record_masks': record_masks,
        'clinvec_dim': CLINVEC_DIM, 'pad_index': CLINVEC_ROWS,
        'electra': build_electra(dims, n_event_types, n_lookup),
        'classifier': build_classifier(dims, n_event_types),
    }


def test_both_members_reach_the_encoder_at_their_own_widths(drug_gate):
    """The ragged case on real extracted data rather than a synthesised
    tensor: text at 5 and ClinVec at 2, in one family."""
    assert drug_gate['dims']['lookup_dims'] == [
        TEXT_EMBED_DIM, drug_gate['clinvec_dim']
    ]
    lookup = drug_gate['batch']['val_data']['lookup']
    assert lookup['indicators'].shape == (2, 4, 2)
    assert lookup['doses'][0] is None and lookup['doses'][1] is not None


def test_the_pool_is_the_dose_weighted_mean(drug_gate):
    """Section 5.1: 'dose-scale by ``drug_doses``, zero by
    ``drug_masks``, sum, divide by the mask sum'.

    Patient 1001's first timestep dispenses ClinVec rows 2 and 3 at
    doses 1.0 and 0.5; the table fills row r with r + 1, so the slots
    are 3 and 4 and the third slot is the all-zero pad.
    """
    from TransEHR2.utils import resolve_lookup_embeddings

    lookup = drug_gate['batch']['val_data']['lookup']
    pooled = resolve_lookup_embeddings(lookup)[1]
    assert pooled.shape == (2, 4, drug_gate['clinvec_dim'])
    # (3 * 1.0 + 4 * 0.5 + 0) / 2 unmasked slots.
    expected = (3.0 * 1.0 + 4.0 * 0.5) / 2.0
    assert pooled[0, 0].tolist() == [expected] * drug_gate['clinvec_dim']
    # Text pools as a no-op: one slot, weight 1.
    assert torch.equal(
        resolve_lookup_embeddings(lookup)[0], lookup['slot_values'][0]
    )


def test_the_pool_divides_by_a_clamped_mask_sum(drug_gate):
    """Section 5.1's first numerical detail: a timestep reaching the
    pool with no unmasked slot would otherwise produce 0/0. Patient
    1001's later timesteps carry no dispensation at all, so they are
    that case."""
    from TransEHR2.utils import pool_lookup_slots

    lookup = drug_gate['batch']['val_data']['lookup']
    pooled = pool_lookup_slots(
        lookup['slot_values'][1], lookup['doses'][1], lookup['masks'][1]
    )
    assert lookup['masks'][1][0, 1].sum() == 0
    assert torch.isfinite(pooled).all()
    assert not pooled[0, 1].any()


def test_a_zero_dose_timestep_pools_to_the_zero_vector(drug_gate):
    """Section 5.1's second: a timestep whose real slots all carry
    ``REL_DAILY_QTY == 0`` pools to zero even though its indicator is 1,
    and ``losses.py``'s eps guard then drops it from the reconstruction
    loss. Recorded here as expected behaviour, not a bug."""
    from TransEHR2.utils import pool_lookup_slots

    lookup = drug_gate['batch']['val_data']['lookup']
    zero_doses = torch.zeros_like(lookup['doses'][1])
    pooled = pool_lookup_slots(
        lookup['slot_values'][1], zero_doses, lookup['masks'][1]
    )
    assert not pooled.any()


def test_the_values_are_concatenated_in_a_single_cat(drug_gate):
    """Section 5.1's 'concatenate once, not twice': the nested form
    materializes an intermediate the size of the whole embedded block.
    The oracle is the arithmetic, checked against the nested form."""
    from TransEHR2.utils import (
        combine_value_and_lookup_data, resolve_lookup_embeddings
    )

    val_data = drug_gate['batch']['val_data']
    lookup = val_data['lookup']
    embeds = resolve_lookup_embeddings(lookup)
    values = torch.cat(val_data['numeric']['values'], dim=-1)

    _, combined = combine_value_and_lookup_data(
        value_assoc_indicators=val_data['numeric']['indicators'],
        value_assoc_values=values,
        lookup_assoc_indicators=lookup['indicators'],
        lookup_embeddings=embeds
    )
    nested = torch.cat([values, torch.cat(embeds, dim=-1)], dim=-1)
    assert torch.equal(combined, nested)


def test_a_gradient_reaches_an_individual_drug_slot(drug_gate):
    """The XAI requirement of section 4.3, tested directly: 'the
    dose-weighted mean is applied at the encoder input, which is what
    preserves per-slot gradients'. A gradient that stops at the pooled
    vector cannot attribute a prediction to one DIN.

    The two dispensed slots must receive gradient and the padded third
    must not -- a nonzero gradient there would mean the pad row was
    participating."""
    batch = drug_gate['batch']
    slots = batch['val_data']['lookup']['slot_values'][1]
    slots.requires_grad_(True)

    logits = drug_gate['classifier'](batch)
    logits.sum().backward()

    grad = slots.grad
    assert grad is not None
    # Patient 1001, first timestep: slots 0 and 1 were dispensed.
    assert grad[0, 0, 0].abs().sum() > 0
    assert grad[0, 0, 1].abs().sum() > 0
    assert grad[0, 0, 2].abs().sum() == 0
    # And the two differ, so the attribution separates the two DINs.
    assert not torch.equal(grad[0, 0, 0], grad[0, 0, 1])


def test_the_family_trains_end_to_end_with_a_drug_feature(drug_gate):
    """Section 9's second done-when: forward and backward complete on a
    fixture batch once the drug feature is in the family."""
    out = drug_gate['electra'](
        drug_gate['batch'], drug_gate['record_masks'], device='cpu',
        compute_intensities=False
    )
    dims = drug_gate['dims']
    assert [v.shape[-1] for v in out['generator']['lookup']['embedded_values']] == (
        dims['lookup_dims']
    )
    assert out['discriminator']['lookup'].shape == (2, 4, 2)

    gen_loss = MaskedGeneratorLoss(
        ordinal_features=dims['ordinal'] or None
    )(out['generator'], out['masked_targets'], drug_gate['record_masks'])
    disc_loss = MaskedDiscriminatorLoss()(
        out['discriminator'], drug_gate['record_masks']
    )
    (gen_loss + disc_loss).backward()

    assert torch.isfinite(gen_loss) and torch.isfinite(disc_loss)
    grads = [
        p.grad for p in drug_gate['electra'].generator.lookup_heads.parameters()
    ]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_the_generator_reconstructs_the_pooled_drug_vector(drug_gate):
    """Section 5.1: 'pretraining reconstructs the pooled 128-dim drug
    vector by cosine similarity -- the same treatment text gets'. Not
    the 30 slots individually: the pooled vector is what the encoder
    consumes, and per-slot reconstruction would force an ordering on an
    unordered set."""
    out = drug_gate['electra'](
        drug_gate['batch'], drug_gate['record_masks'], device='cpu',
        compute_intensities=False
    )
    targets = out['masked_targets']['lookup']['embedded_values']
    assert targets[1].shape[-1] == drug_gate['clinvec_dim']
    assert drug_gate['electra'].generator.lookup_heads[1].out_features == (
        drug_gate['clinvec_dim']
    )


def test_an_empty_family_leaves_the_model_path_alone(gate):
    """``TEXT_FEATS: []`` and ``DRUG_FEATS: []`` together: the batch
    carries no ``lookup`` key at all, which is what keeps the gate above
    a statement about text and not about the family's plumbing."""
    batch = gate['batch']
    del batch['val_data']['lookup']
    dims = gate['dims']
    dims = dict(dims, lookup_dims=[],
                n_features=dims['n_features'] - 1,
                feat_dim=dims['feat_dim'] - TEXT_EMBED_DIM)

    torch.manual_seed(1)
    generator = MaskedTokenGenerator(
        encoder=value_encoder(dims),
        d_model=D_MODEL,
        numeric_dims=dims['numeric'],
        categorical_classes=dims['categorical'],
        ordinal_features=dims['ordinal'] or None,
        lookup_dims=[],
        predict_indicators=False,
        dim_feedforward=32
    )
    assert not generator.predict_lookup_feats
    record_masks, _ = generate_record_masks(batch)
    assert 'lookup' not in record_masks
    output = generator(batch['val_data'], record_masks)
    assert 'lookup' not in output
