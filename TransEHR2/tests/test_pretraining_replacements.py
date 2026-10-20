"""What the discriminator is shown, and what it is asked, after the 2026-10-19 fixes.

`ELECTRA._prepare_discriminator_input_inplace` writes the generator's guesses into the masked
cells and returns the discriminator's targets. These pin the five properties it now has --
sampled replacements, detached replacements, a missing state wherever the written indicator is 0,
a whole predicted row for a fake presence, and labels that say whether a cell *changed* rather
than whether it was masked -- plus the mask rules `generate_record_masks` follows when the
generator predicts indicators, the repaired indicator heads, the event-encoder switch, and
seeding.

The replacement tests call the method on a stand-in for ``self``: it reads nothing from the model
but ``generator.predict_indicators`` and the sampling helper, so a hand-built batch pins it
exactly, without a generator in the way.
"""

from types import SimpleNamespace

import torch

from TransEHR2.models import ELECTRA, MixedClassifier
from TransEHR2.modules import MaskedTokenGenerator, ValueDataEncoder
from TransEHR2.test_encoder_simulated_training import (
    D_MODEL, DIM_FF, N_HEADS, NUMERIC_DIMS, N_VAL_FEATURES, TOTAL_FEAT_DIM,
    _build_classifier, _simulate_cohort
)
from TransEHR2.utils import _gen_val_assoc_feat_mask, generate_record_masks, seed_everything

T = 6
BIG = 50.0  # a logit that makes a Bernoulli or softmax draw certain


def _owner(predict_indicators):
    return SimpleNamespace(generator=SimpleNamespace(predict_indicators=predict_indicators),
                           _sample_replacement=ELECTRA._sample_replacement)


def _prepare(owner, value_data, gen_output, record_masks):
    return ELECTRA._prepare_discriminator_input_inplace(owner, value_data, gen_output, record_masks)


def _families(observed):
    """One numeric (width 2), one categorical (3 classes) and one lookup (width 4) feature.

    ``observed`` is a (T,) 0/1 tensor shared by all three. Observed cells hold distinctive values;
    unobserved cells hold the all-zero missing state, as a gated batch does.
    """
    ind = observed.view(1, T, 1).float()
    numeric = torch.tensor([[1.0, 2.0]]).expand(T, 2).reshape(1, T, 2) * ind
    categorical = torch.nn.functional.one_hot(torch.zeros(1, T, dtype=torch.long), 3).float() * ind
    lookup = torch.full((1, T, 4), 3.0) * ind
    return {
        'numeric': {'indicators': ind.clone(), 'values': [numeric.clone()]},
        'categorical': {'indicators': ind.clone(), 'values': [categorical.clone()]},
        'lookup': {'indicators': ind.clone(), 'embedded_values': [lookup.clone()]},
    }


def _all_masked(value_data, value_rows=None):
    """Every cell masked; component masks set where ``value_rows`` (default: all) says."""
    masks = {}
    for family, entry in value_data.items():
        key = 'embedded_values' if family == 'lookup' else 'values'
        rows = torch.ones(1, T) if value_rows is None else value_rows.view(1, T).float()
        masks[family] = {
            'indicators': torch.ones_like(entry['indicators']),
            key: [rows.unsqueeze(-1).expand_as(v).clone() for v in entry[key]],
        }
    return masks


def _guesses(indicator_logit, categorical_logits=(BIG, -BIG, -BIG), requires_grad=False):
    def leaf(t):
        return t.requires_grad_(requires_grad)
    return {
        'numeric': {'indicators': leaf(torch.full((1, T, 1), indicator_logit)),
                    'values': [leaf(torch.full((1, T, 2), 9.0))]},
        'categorical': {'indicators': leaf(torch.full((1, T, 1), indicator_logit)),
                        'values': [leaf(torch.tensor(categorical_logits).expand(1, T, 3).clone())]},
        'lookup': {'indicators': leaf(torch.full((1, T, 1), indicator_logit)),
                   'embedded_values': [leaf(torch.full((1, T, 4), 7.0))]},
    }


OBSERVED = torch.tensor([1, 1, 1, 0, 0, 0])


def test_a_written_absence_leaves_the_missing_state_in_every_family():
    value_data = _families(OBSERVED)
    labels = _prepare(_owner(True), value_data, _guesses(-BIG), _all_masked(value_data))
    for family, entry in value_data.items():
        key = 'embedded_values' if family == 'lookup' else 'values'
        assert not entry['indicators'].any(), family
        assert not entry[key][0].any(), family
        # Absent cells written absent are untouched, so real; observed ones changed.
        assert labels[family]['indicators'][0, :, 0].tolist() == [1, 1, 1, 0, 0, 0], family


def test_a_fake_presence_gets_the_whole_predicted_row():
    """With no component masked at the absent cells, as with indicators predicted, the row
    written is still the generator's full guess, never a partial vector."""
    value_data = _families(OBSERVED)
    masks = _all_masked(value_data, value_rows=OBSERVED)
    labels = _prepare(_owner(True), value_data, _guesses(BIG), masks)
    absent = slice(3, 6)
    assert torch.all(value_data['numeric']['values'][0][0, absent] == 9.0)
    assert torch.all(value_data['lookup']['embedded_values'][0][0, absent] == 7.0)
    assert torch.all(value_data['categorical']['values'][0][0, absent, 0] == 1.0)
    for family in value_data:
        assert torch.all(value_data[family]['indicators'][0, absent] == 1.0)
        assert torch.all(labels[family]['indicators'][0, absent] == 1.0)


def test_an_unchanged_masked_cell_is_labelled_real():
    """Without indicator prediction, a categorical draw that reproduces the truth leaves the
    cell identical to the original; ELECTRA calls that real, and so does this."""
    value_data = _families(torch.ones(T, dtype=torch.long))
    labels = _prepare(_owner(False), value_data, _guesses(0.0, (BIG, -BIG, -BIG)),
                      _all_masked(value_data))
    assert not labels['categorical']['indicators'].any()
    value_data = _families(torch.ones(T, dtype=torch.long))
    labels = _prepare(_owner(False), value_data, _guesses(0.0, (-BIG, BIG, -BIG)),
                      _all_masked(value_data))
    assert labels['categorical']['indicators'].all()


def test_categorical_replacements_are_sampled_not_argmax():
    torch.manual_seed(0)
    n = 400
    value_data = {'categorical': {
        'indicators': torch.ones(1, n, 1),
        'values': [torch.nn.functional.one_hot(torch.zeros(1, n, dtype=torch.long), 3).float()]}}
    masks = {'categorical': {'indicators': torch.ones(1, n, 1), 'values': [torch.ones(1, n, 3)]}}
    gen = {'categorical': {'indicators': None,
                           'values': [torch.tensor([0.1, 0.0, 0.0]).expand(1, n, 3).clone()]}}
    _prepare(_owner(False), value_data, gen, masks)
    drawn = value_data['categorical']['values'][0][0].argmax(dim=-1)
    assert len(drawn.unique()) == 3


def test_replacements_carry_no_gradient_to_the_generator():
    value_data = _families(OBSERVED)
    gen = _guesses(BIG, requires_grad=True)
    _prepare(_owner(True), value_data, gen, _all_masked(value_data))
    for family, entry in value_data.items():
        key = 'embedded_values' if family == 'lookup' else 'values'
        assert not entry[key][0].requires_grad, family
        assert not entry['indicators'].requires_grad, family


def _mask_batch(observed, family='numeric'):
    """A one-feature batch for `_gen_val_assoc_feat_mask`: observed cells and padding-free."""
    n = observed.numel()
    entry = {'indicators': observed.view(1, n, 1).float()}
    values = [torch.ones(1, n, 3)]
    if family == 'lookup':
        entry['slot_values'] = values
        val_masks = {'lookup': {'indicators': torch.zeros(1, n, 1),
                                'embedded_values': [torch.zeros(1, n, 3)]}}
    else:
        entry['values'] = values
        val_masks = {family: {'indicators': torch.zeros(1, n, 1), 'values': [torch.zeros(1, n, 3)]}}
    data = {'val_data': {family: entry, 'masks': torch.ones(1, n)}}
    return data, val_masks


def test_with_indicators_predicted_masked_absent_cells_get_no_value_mask():
    torch.manual_seed(0)
    observed = (torch.arange(200) % 2).long()
    data, val_masks = _mask_batch(observed)
    _gen_val_assoc_feat_mask(data, 'numeric', val_masks, 0.5, 1.0, 1.0, predict_indicators=True)
    indicator = val_masks['numeric']['indicators'][0, :, 0].bool()
    value = val_masks['numeric']['values'][0][0].any(dim=-1)
    absent = observed == 0
    assert (indicator & absent).any()          # absent cells are masked for their indicator
    assert not (value & absent).any()          # but never for a value
    assert torch.equal(value, indicator & ~absent)


def test_without_indicator_prediction_value_masks_are_as_before():
    torch.manual_seed(0)
    observed = (torch.arange(200) % 2).long()
    data, val_masks = _mask_batch(observed)
    _gen_val_assoc_feat_mask(data, 'numeric', val_masks, 0.5, 1.0, 1.0, predict_indicators=False)
    indicator = val_masks['numeric']['indicators'][0, :, 0].bool()
    value = val_masks['numeric']['values'][0][0].any(dim=-1)
    assert torch.equal(value, indicator)


def test_lookup_samples_absent_cells_for_its_indicator_only_when_predicted():
    observed = (torch.arange(200) % 2).long()
    absent = observed == 0
    for predict in (False, True):
        torch.manual_seed(0)
        data, val_masks = _mask_batch(observed, family='lookup')
        _gen_val_assoc_feat_mask(data, 'lookup', val_masks, 0.5, 1.0, 1.0,
                                 predict_indicators=predict)
        indicator = val_masks['lookup']['indicators'][0, :, 0].bool()
        value = val_masks['lookup']['embedded_values'][0][0].any(dim=-1)
        assert (indicator & absent).any() == predict
        assert not (value & absent).any()


def test_indicator_heads_read_the_encoder_and_train_it():
    """They used to read the masked indicator vector through Linear(d_model, ...), which ran only
    where d_model equalled the feature count. Here it does not (2 features, d_model 32)."""
    assert N_VAL_FEATURES != D_MODEL
    torch.manual_seed(0)
    encoder = ValueDataEncoder(
        n_features=N_VAL_FEATURES, feat_dim=TOTAL_FEAT_DIM, d_model=D_MODEL, n_heads=N_HEADS,
        n_encoder_blocks=1, dim_feedforward=DIM_FF, dropout=0.0, activation='gelu',
        norm='LayerNorm', normalize_before=True)
    generator = MaskedTokenGenerator(encoder=encoder, d_model=D_MODEL, numeric_dims=NUMERIC_DIMS,
                                     categorical_classes=[], predict_indicators=True,
                                     dim_feedforward=DIM_FF)
    batch, _ = _simulate_cohort(all_padding_episode=False)
    record_masks, _ = generate_record_masks(batch, predict_indicators=True)
    out = generator(batch['val_data'], record_masks)
    logits = out['numeric']['indicators']
    assert logits.shape == batch['val_data']['numeric']['indicators'].shape
    logits.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())


def test_without_an_event_encoder_the_event_stream_is_ignored():
    torch.manual_seed(0)
    full = _build_classifier()
    classifier = MixedClassifier(event_encoder=None, val_encoder=full.val_encoder, d_event_enc=0,
                                 d_val_enc=D_MODEL, d_statics=0, num_classes=1, aggr='mean')
    classifier.eval()
    batch, _ = _simulate_cohort()
    with torch.no_grad():
        before = classifier(batch)
        batch['event_data']['indicators'] = torch.rand_like(batch['event_data']['indicators'])
        after = classifier(batch)
    assert torch.equal(before, after)
    assert classifier.head[0].in_features == D_MODEL


def test_an_event_width_without_an_event_encoder_is_refused():
    full = _build_classifier()
    try:
        MixedClassifier(event_encoder=None, val_encoder=full.val_encoder, d_event_enc=D_MODEL,
                        d_val_enc=D_MODEL, d_statics=0, num_classes=1)
    except ValueError:
        return
    raise AssertionError('a nonzero d_event_enc with no event encoder was accepted')


def test_seeding_repeats_masks_and_replacements_and_a_new_seed_does_not():
    def draw(seed):
        seed_everything(seed)
        batch, _ = _simulate_cohort(all_padding_episode=False)
        masks, _ = generate_record_masks(batch, predict_indicators=True)
        value_data = _families(OBSERVED)
        _prepare(_owner(True), value_data, _guesses(0.0, (0.0, 0.0, 0.0)), _all_masked(value_data))
        return (masks['numeric']['indicators'], value_data['categorical']['values'][0],
                value_data['numeric']['indicators'])
    first, again, other = draw(7), draw(7), draw(8)
    assert all(torch.equal(a, b) for a, b in zip(first, again))
    assert not all(torch.equal(a, b) for a, b in zip(first, other))
