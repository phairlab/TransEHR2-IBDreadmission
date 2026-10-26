"""Text-field units: masking a field in extraction's dialect, and scoring it inside TSR.

The encoder is replaced by a stand-in that maps each string to a fixed vector, so the TSR
arithmetic is checked on the deployed classifier without loading a language model. What the
stand-in cannot check -- that re-embedding an unmasked stored string reproduces its table row --
is a property of the real encoder and table, and belongs to a run on the cluster.
"""

import hashlib

import pytest
import torch

from IBDdataprep.superfeature import dump_superfeature, parse_superfeature

from TransEHR2.tests.test_tsr_units import TEXT_WIDTH, _model_and_batch, _target
from TransEHR2.text_fields import (drop_fields, keep_fields, present_fields,
                                   text_field_units)
from TransEHR2.tsr import outputs_many, tsr_scores

FIELDS = ('MIS_CODE', 'PROVIDER_SVC', 'PROCEDURES', 'DIAGNOSES', 'CLAIMS')
TEXT_COLUMN = 2

STRING = dump_superfeature({
    'DIAGNOSES': [{'DX_DESC': 'Crohn disease of small intestine', 'DX_TYPE': 'M'},
                  {'DX_DESC': 'Iron deficiency anaemia', 'DX_TYPE': '1'}],
    'PROCEDURES': ['Colonoscopy'],
    'PROVIDER_SVC': ['Gastroenterology'],
})
LONE = dump_superfeature({'CLAIMS': [{'CLAIM_DESC': 'Office visit', 'DX_DESC': 'K50',
                                      'PERS_CAPB_PRVD_SPEC_AD': 'GP', 'SECTOR': 'Community'}]})


def _embed(strings):
    """A fixed vector per string, so the same string always embeds the same way."""
    out = []
    for text in strings:
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        out.append(torch.randn(TEXT_WIDTH, generator=torch.Generator().manual_seed(seed)))
    return torch.stack(out)


def _episode(text):
    model, batch = _model_and_batch()
    batch['val_data']['lookup']['slot_values'][0][0, 2] = _embed([text])[0]
    return model, batch


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def test_dropping_nothing_gives_back_the_stored_string():
    assert drop_fields(STRING, []) == STRING


def test_a_dropped_field_is_gone_and_the_rest_is_untouched():
    masked = parse_superfeature(drop_fields(STRING, ['PROCEDURES']))
    original = parse_superfeature(STRING)
    assert list(masked) == ['DIAGNOSES', 'PROVIDER_SVC']
    assert masked['DIAGNOSES'] == original['DIAGNOSES']


def test_keeping_fields_is_the_complement_of_dropping_them():
    assert keep_fields(STRING, ['DIAGNOSES']) == drop_fields(STRING,
                                                           ['PROCEDURES', 'PROVIDER_SVC'])


def test_dropping_the_only_field_leaves_nothing():
    assert drop_fields(LONE, ['CLAIMS']) == ''


def test_present_fields_follow_the_given_order():
    assert present_fields(STRING, FIELDS) == ['PROVIDER_SVC', 'PROCEDURES', 'DIAGNOSES']


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def test_field_scores_split_their_cell_and_sum_back_to_it():
    model, batch = _episode(STRING)
    result = tsr_scores(model, batch, target=_target())
    fields = text_field_units(model, batch, result, [STRING], _embed, FIELDS,
                              target=_target())
    assert fields['cells'] == [(2, TEXT_COLUMN)]
    assert fields['field'] == ['PROVIDER_SVC', 'PROCEDURES', 'DIAGNOSES']
    assert fields['share'].sum().item() == pytest.approx(1.0, abs=1e-6)
    assert fields['score'].sum().item() == pytest.approx(
        float(result['scores'][0, 2, TEXT_COLUMN]), rel=1e-5, abs=1e-9)
    assert torch.isfinite(fields['baseline']).all()


def test_a_field_deletion_is_the_model_on_the_re_embedded_string():
    model, batch = _episode(STRING)
    masked = drop_fields(STRING, ['DIAGNOSES'])
    vector = _embed([masked])[0]
    perturbed = outputs_many(model, batch, [(2, TEXT_COLUMN, vector)])[0]

    _, by_hand = _episode(masked)
    with torch.no_grad():
        expected = model(by_hand)
    assert torch.allclose(perturbed, expected, atol=1e-6)


def test_a_cell_with_one_field_deletes_the_cell_for_it():
    model, batch = _episode(LONE)
    result = tsr_scores(model, batch, target=_target())
    fields = text_field_units(model, batch, result, [LONE], _embed, FIELDS,
                              target=_target())
    assert fields['field'] == ['CLAIMS']
    assert fields['embeddings'] == [None]
    assert fields['share'].item() == pytest.approx(1.0)


def test_only_the_top_cells_are_opened():
    model, batch = _episode(STRING)
    result = tsr_scores(model, batch, target=_target())
    fields = text_field_units(model, batch, result, [STRING], _embed, FIELDS, top_cells=0,
                              target=_target())
    assert fields['cells'] == [] and fields['field'] == []


def test_more_than_one_episode_is_refused():
    model, batch = _episode(STRING)
    result = tsr_scores(model, batch, target=_target())
    result = dict(result, scores=result['scores'].repeat(2, 1, 1))
    with pytest.raises(ValueError, match='one episode'):
        text_field_units(model, batch, result, [STRING], _embed, FIELDS)
