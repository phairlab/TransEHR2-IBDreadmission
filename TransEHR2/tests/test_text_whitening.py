"""Whitening inside the encoder: one transform for the table, every re-embed and attribution.

`GradientTraceableLLM` applies a ContrastiveBMMB whitening file after pooling, so the ways this
can go wrong are a transform fitted on another encoder, one applied twice, and a table built
with a different one from the encoder compared with it. Each gives vectors of the right shape.
"""

import json
import os

import pytest
import torch

from TransEHR2.attribution import token_attributions
from TransEHR2.modules import GradientTraceableLLM, TEXT_TABLE_PROVENANCE
from TransEHR2.tests.test_llm_compat import HIDDEN, _encode, tiny_model  # noqa: F401
from TransEHR2.tests.test_two_stage_attribution import (LENGTH, STRINGS,  # noqa: F401
                                                        downstream)

import embed  # scripts/, on sys.path via conftest


def _transform(path, checkpoint, pooling='cls', seed=1):
    """A whitening file in run_arms.py's format: symmetric W, a mean, and what it fitted on."""
    torch.manual_seed(seed)
    a = torch.randn(HIDDEN, HIDDEN)
    state = {'mean': torch.randn(1, HIDDEN), 'transform': a @ a.t() / HIDDEN,
             'epsilon': 1e-5, 'arm': 'A4w',
             'checkpoint': os.path.realpath(str(checkpoint)), 'pooling': pooling}
    torch.save(state, path)
    return state


def _llm(path, whitening=None, pooling='cls'):
    llm = GradientTraceableLLM(str(path), max_length=LENGTH, pooling=pooling,
                               use_gradient_checkpointing=False, whitening=whitening)
    llm.model.eval()
    return llm


def test_both_forwards_whiten_identically(tiny_model, tmp_path):
    state = _transform(tmp_path / 'w.pt', tiny_model)
    plain, whitened = _llm(tiny_model), _llm(tiny_model, tmp_path / 'w.pt')
    encoded = _encode(plain, STRINGS, LENGTH)

    with torch.no_grad():
        raw = plain(encoded['input_ids'], attention_mask=encoded['attention_mask'])
        z = whitened(encoded['input_ids'], attention_mask=encoded['attention_mask'])
        z_embeds = whitened.forward_from_embeds(
            whitened.model.get_input_embeddings()(encoded['input_ids']),
            encoded['attention_mask'])

    expected = (raw.float() - state['mean']) @ state['transform']
    assert torch.allclose(z, expected, atol=1e-5)
    assert torch.allclose(z_embeds, expected, atol=1e-5)


def test_attribution_differentiates_through_the_encoders_whitening(tiny_model, tmp_path,
                                                                     downstream):
    """The same scores as handing W in by hand, which test_two_stage_attribution checks
    against end-to-end backpropagation. The mean shifts no gradient."""
    state = _transform(tmp_path / 'w.pt', tiny_model)
    plain, whitened = _llm(tiny_model), _llm(tiny_model, tmp_path / 'w.pt')
    encoded = _encode(plain, STRINGS, LENGTH)

    with torch.no_grad():
        stored = whitened(encoded['input_ids'], attention_mask=encoded['attention_mask'])
    stored = stored.clone().requires_grad_(True)
    downstream(stored).sum().backward()

    inside = token_attributions(whitened, encoded['input_ids'], encoded['attention_mask'],
                                stored.grad)
    by_hand = token_attributions(plain, encoded['input_ids'], encoded['attention_mask'],
                                 stored.grad, whitening_transform=state['transform'])
    assert torch.allclose(inside, by_hand, atol=1e-5)

    with pytest.raises(ValueError, match='applied twice'):
        token_attributions(whitened, encoded['input_ids'], encoded['attention_mask'],
                           stored.grad, whitening_transform=state['transform'])


def test_a_transform_fitted_elsewhere_is_refused(tiny_model, tmp_path):
    _transform(tmp_path / 'other.pt', tmp_path / 'some_other_checkpoint')
    with pytest.raises(ValueError, match='was fitted on'):
        _llm(tiny_model, tmp_path / 'other.pt')

    _transform(tmp_path / 'mean.pt', tiny_model, pooling='mean')
    with pytest.raises(ValueError, match='pooling'):
        _llm(tiny_model, tmp_path / 'mean.pt')


def test_the_table_records_its_encoder_and_is_checked(tiny_model, tmp_path):
    _transform(tmp_path / 'w.pt', tiny_model)
    _transform(tmp_path / 'w2.pt', tiny_model, seed=2)
    whitened = _llm(tiny_model, tmp_path / 'w.pt')
    tables = tmp_path / 'lookup_tables'
    tables.mkdir()

    # A table with no record predates it, and was built unwhitened.
    _llm(tiny_model).check_table(str(tables))
    with pytest.raises(ValueError, match='is missing'):
        whitened.check_table(str(tables))

    embed.record_text_provenance(str(tables), whitened.provenance())
    whitened.check_table(str(tables))
    with pytest.raises(ValueError, match='whitening'):
        _llm(tiny_model).check_table(str(tables))
    with pytest.raises(ValueError, match='whitening'):
        _llm(tiny_model, tmp_path / 'w2.pt').check_table(str(tables))

    recorded = json.load(open(tables / TEXT_TABLE_PROVENANCE))
    assert recorded['whitening'] == os.path.realpath(tmp_path / 'w.pt')
    assert recorded['pooling'] == 'cls'
