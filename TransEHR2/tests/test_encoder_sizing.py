"""The value encoder is sized over the whole lookup family, drugs included.

``DRUG_FEATS`` is the half of the family that is easy to lose. It has no
``VALUED_FEATS`` entry and no ``timeseries.csv`` column, so a sizing pass
that walks the config's feature lists never meets it -- but extraction
writes ``val_drug_indicators.npy`` all the same, ``MixedDataset`` builds
its column list from every entry of ``lookup_feat_types`` with no filter,
and ``MixedClassifier`` concatenates whatever the batch carries. The
three entry points used to count ``VALUED_FEATS + TEXT_FEATS``, which is
one feature and one ClinVec width short of that, and the first forward
died in ``indicator_input_projection_layer``.

So the condition here is a forward pass, not an arithmetic identity:
these build the model the way the entry points build it, from a config
whose ``DRUG_FEATS`` is non-empty, and push one collated batch through
both the pretraining stack and the classifier. The text table is
deliberately a different width from ClinVec's, so a sizing that happened
to double-count text would not pass either.
"""

import numpy as np
import pickle
import pytest
import torch

from pathlib import Path

from TransEHR2.data.preprocessing import (
    load_dataset, lookup_feat_widths, value_encoder_dims
)
from TransEHR2.models import ELECTRA, MixedClassifier
from TransEHR2.modules import (
    EventDataEncoder, MaskedTokenDiscriminator, MaskedTokenGenerator,
    ValueDataEncoder
)
from TransEHR2.utils import generate_record_masks

from extract_data import main as extract_main

from .conftest import CLINVEC_DIM, CLINVEC_ROWS, MiniRoot, collate_for_model

# Not ClinVec's width: the two halves of the family are sized
# per-feature, and equal widths would hide a sizing that used one for
# both.
TEXT_EMBED_DIM = 5
D_MODEL = 16

# ``conftest``'s config minus ``UB``, whose two ordinal levels are not
# what this is about.
VALUED_FEATS = ['NUM', 'CAT', 'ORD']


@pytest.fixture
def drug_root(tmp_path):
    """An extracted cohort carrying both a text and a drug feature."""
    mini = MiniRoot(Path(tmp_path))
    mini.config = dict(mini.config)
    mini.config['VALUED_FEATS'] = list(VALUED_FEATS)
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
        drugs=[('2019-01-02T00:00:00Z', 0, 2, 1.0),
               ('2019-01-02T00:00:00Z', 1, 3, 0.5)],
    )
    mini.add_patient(
        1002,
        timeseries=[
            ['2019-02-01T00:00:00Z', 0.5, 'U', '1-24', '', 'a note', 1],
            ['2019-02-02T00:00:00Z', 2.0, 'L', '25-50', '', '', ''],
            ['2019-02-03T00:00:00Z', 4.0, 'U', '0', '', 'third note', 1],
        ],
        stays=[('AMB', '2019-02-01T00:00:00Z', '2019-02-03T00:00:00Z')],
        drugs=[('2019-02-01T00:00:00Z', 0, 1, 2.0)],
    )
    mini.add_fold('fold0', train=[0, 1], val=[0, 1], test=[0, 1])

    config_path = mini.finish()
    assert extract_main([str(config_path)]) == 0

    with open(mini.extracted / 'text_strings.pkl', 'rb') as f:
        n_strings = len(pickle.load(f))
    np.save(
        mini.lookup_tables / 'text_embeddings.npy',
        np.repeat(np.arange(1, n_strings + 1, dtype=np.float32)[:, None],
                  TEXT_EMBED_DIM, axis=1)
    )
    drug = np.repeat(
        np.arange(1, CLINVEC_ROWS + 1, dtype=np.float32)[:, None],
        CLINVEC_DIM, axis=1
    )
    # The final all-zero row is the pad index unused slots carry.
    np.save(mini.lookup_tables / 'drug_embeddings.npy',
            np.vstack([drug, np.zeros((1, CLINVEC_DIM), dtype=np.float32)]))

    dataset = load_dataset(str(mini.extracted), fold='fold0')
    return {
        'mini': mini,
        'dataset': dataset,
        'batch': collate_for_model([dataset[0], dataset[1]]),
    }


def value_encoder(n_features, feat_dim):
    return ValueDataEncoder(
        n_features=n_features, feat_dim=feat_dim, d_model=D_MODEL, n_heads=2,
        n_encoder_blocks=1, dim_feedforward=32, dropout=0.0, norm='LayerNorm'
    )


def event_encoder(n_event_types):
    return EventDataEncoder(
        num_types=n_event_types, d_model=16, d_inner=32, n_layers=1,
        n_head=2, d_k=8, d_v=8, dropout=0.0
    )


def test_the_widths_are_the_ones_the_batch_carries(drug_root):
    """``lookup_feat_widths`` is the family's own order and widths.

    The pooled embedding a lookup feature contributes is as wide as the
    table it is gathered from, so the per-feature widths and the slot
    tensors' last axis are the same list -- read here off the batch
    rather than off the config, because the batch is what the encoder
    meets.
    """
    widths = lookup_feat_widths(str(drug_root['mini'].extracted))
    slots = drug_root['batch']['val_data']['lookup']['slot_values']

    assert widths == [TEXT_EMBED_DIM, CLINVEC_DIM]
    assert widths == [t.shape[-1] for t in slots]


def test_the_encoder_dims_cover_every_concatenated_column(drug_root):
    """``value_encoder_dims`` is what ``MixedClassifier`` concatenates.

    The indicator axis is one column per valued feature plus one per
    lookup feature, which is where the drug column used to go missing.
    """
    val_data = drug_root['batch']['val_data']
    widths = lookup_feat_widths(str(drug_root['mini'].extracted))
    n_features, feat_dim = value_encoder_dims(
        drug_root['mini'].var_properties, VALUED_FEATS, widths
    )

    observed_features = sum(
        val_data[family]['indicators'].shape[-1]
        for family in ('numeric', 'categorical', 'ordinal', 'multilabel',
                       'lookup')
    )
    observed_dim = sum(
        t.shape[-1]
        for family in ('numeric', 'categorical', 'ordinal', 'multilabel')
        for t in val_data[family]['values']
    ) + sum(widths)

    assert n_features == observed_features
    assert feat_dim == observed_dim


def test_the_classifier_runs_a_forward_with_a_drug_feature(drug_root):
    """The finetuning and inference path, sized as the entry points
    size it. Before the fix this raised ``mat1 and mat2 shapes cannot be
    multiplied`` on the indicator projection."""
    batch = drug_root['batch']
    widths = lookup_feat_widths(str(drug_root['mini'].extracted))
    n_features, feat_dim = value_encoder_dims(
        drug_root['mini'].var_properties, VALUED_FEATS, widths
    )
    n_event_types = batch['event_data']['indicators'].shape[-1]

    torch.manual_seed(0)
    model = MixedClassifier(
        event_encoder=event_encoder(n_event_types),
        val_encoder=value_encoder(n_features, feat_dim),
        d_event_enc=16, d_val_enc=D_MODEL, d_statics=0, num_classes=2,
        aggr='mean', use_lookup=True
    )
    model.eval()

    with torch.no_grad():
        logits = model(batch)
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()


def test_pretraining_runs_a_forward_with_a_drug_feature(drug_root):
    """The pretraining path, whose heads are per lookup *feature*.

    ``MaskedTokenGenerator`` takes one head per entry of ``lookup_dims``
    and ``MaskedTokenDiscriminator`` one output per lookup feature, so a
    text-only count mis-sizes both on top of the encoder.
    """
    batch = drug_root['batch']
    widths = lookup_feat_widths(str(drug_root['mini'].extracted))
    n_features, feat_dim = value_encoder_dims(
        drug_root['mini'].var_properties, VALUED_FEATS, widths
    )

    torch.manual_seed(0)
    generator = MaskedTokenGenerator(
        encoder=value_encoder(n_features, feat_dim), d_model=D_MODEL,
        numeric_dims=[1], categorical_classes=[2], ordinal_features=[3],
        lookup_dims=widths, predict_indicators=False, dim_feedforward=32
    )
    discriminator = MaskedTokenDiscriminator(
        encoder=value_encoder(n_features, feat_dim), d_model=D_MODEL,
        n_numeric_features=1, n_categorical_features=1, n_ordinal_features=1,
        n_multilabel_features=0, n_lookup_features=len(widths),
        n_static_features=0, dim_feedforward=32
    )
    electra = ELECTRA(generator, discriminator, hawkes=None, use_lookup=True)
    electra.eval()

    torch.manual_seed(0)
    record_masks, _ = generate_record_masks(
        batch, feature_sample_rate=0.6, obs_unobs_ratio=2.0,
        subsample_rate=0.5
    )
    out = electra(batch, record_masks, device='cpu',
                  compute_intensities=False)

    predicted = out['generator']['lookup']['embedded_values']
    assert [t.shape[-1] for t in predicted] == widths


def test_the_family_follows_the_data(tmp_path):
    """``use_lookup`` is read off the extraction, not off a flag.

    Text and drugs are used whenever the dataset config declares them,
    so the only thing that turns the family off is a cohort extracted
    without it -- and then the widths are empty, which is what the
    entry points take ``use_lookup`` from. An experiment-level switch
    would be a fourth place for the count to drift from the data.
    """
    mini = MiniRoot(Path(tmp_path))
    mini.config = dict(mini.config)
    mini.config['VALUED_FEATS'] = list(VALUED_FEATS)
    mini.config['TEXT_FEATS'] = []
    mini.config['DRUG_FEATS'] = []
    for name in ('UB', 'TXT', 'DRG'):
        del mini.var_properties[name]
    mini.add_patient(
        1001,
        timeseries=[
            ['2019-01-01T00:00:00Z', 1.5, 'L', '0', '', '', ''],
            ['2019-01-02T00:00:00Z', 2.5, 'U', '25-50', '', '', 1],
        ],
        stays=[('DAD', '2019-01-01T00:00:00Z', '2019-01-02T00:00:00Z')],
    )
    mini.add_fold('fold0', train=[0], val=[0], test=[0])
    assert extract_main([str(mini.finish())]) == 0

    widths = lookup_feat_widths(str(mini.extracted))
    assert widths == []
    assert not bool(widths)

    n_features, _ = value_encoder_dims(
        mini.var_properties, VALUED_FEATS, widths
    )
    assert n_features == len(VALUED_FEATS)
