"""Temporal Saliency Rescaling: the arithmetic, the masking, and what it costs.

Checked against the reference implementation's behaviour (`getTwoStepRescaling` in
`ayaabdelsalam91/TS-Interpretability-Benchmark`), not against the paper's pseudocode -- the two
disagree about whether the feature term is per-cell or per-feature, which is a factor of T in
cost. `test_tsr_recomputes_the_feature_profile_at_every_passing_timestep` is the test that pins
which one this module implements.

The model is a stub whose saliency has a closed form, so every number below is checked against
what it must be rather than against whatever the code produced first. That matters more here
than usual: TSR's output is a product of two scaled aggregates over a perturbed saliency map,
and a wrong mask -- one that zeroed a value but left its indicator saying "observed", or that
deleted a feature across all time instead of in one minute -- changes the numbers without
changing the shapes.
"""

import pytest
import torch

from TransEHR2.tsr import FAMILIES, feature_index, saliency, tsr_scores

BATCH, STEPS, WIDTH = 2, 5, 3
N_NUMERIC, N_LOOKUP = 2, 1
N_FEATURES = N_NUMERIC + N_LOOKUP
WEIGHTS = {'numeric': [1.0, 2.0], 'lookup': [5.0]}


class StubModel(torch.nn.Module):
    """Sums each feature's values, scaled by a per-feature weight and gated by its indicator.

    Deliberately linear and separable: the saliency of cell (t, i) is then exactly
    ``w_i * indicator[t, i] * sum_d value[t, i, d]``, and deleting a cell changes the map only
    at that cell -- so a perturbation that moves anything else is a bug in the masking, not an
    artefact of the model.
    """

    def forward(self, batch):
        val_data = batch['val_data']
        total = None
        for family in FAMILIES:
            if family not in val_data:
                continue
            entry = val_data[family]
            tensors = (entry['embedded_values'] if family == 'lookup'
                       else entry['values'])
            for position, tensor in enumerate(tensors):
                contribution = (WEIGHTS[family][position]
                                * (tensor.sum(dim=-1)
                                   * entry['indicators'][..., position]).sum(dim=-1))
                total = contribution if total is None else total + contribution
        return total


@pytest.fixture
def batch():
    """A collated batch in the documented shape: dense, no sparse blocks left to densify."""
    torch.manual_seed(0)
    return {
        'val_data': {
            'times': torch.arange(STEPS).float().repeat(BATCH, 1),
            'numeric': {
                'indicators': torch.ones(BATCH, STEPS, N_NUMERIC),
                'values': [torch.randn(BATCH, STEPS, WIDTH)
                           for _ in range(N_NUMERIC)],
            },
            'lookup': {
                'indicators': torch.ones(BATCH, STEPS, N_LOOKUP),
                # One slot, no doses and no masks -- the text case, where the pooled
                # embedding is the slot tensor itself.
                'slot_values': [torch.randn(BATCH, STEPS, WIDTH)],
                'doses': [None],
                'masks': [None],
            },
        }
    }


def _expected(batch):
    """The closed form of R(X) for StubModel."""
    val_data = batch['val_data']
    columns = []
    for family, tensors in (('numeric', val_data['numeric']['values']),
                            ('lookup', val_data['lookup']['slot_values'])):
        for position, tensor in enumerate(tensors):
            columns.append(WEIGHTS[family][position] * tensor.sum(dim=-1)
                           * val_data[family]['indicators'][..., position])
    return torch.stack(columns, dim=-1)


def test_the_saliency_is_gradient_times_input(batch):
    scores, lookup_grads = saliency(StubModel(), batch)
    assert scores.shape == (BATCH, STEPS, N_FEATURES)
    assert torch.allclose(scores, _expected(batch), atol=1e-6)
    assert len(lookup_grads) == N_LOOKUP
    assert lookup_grads[0].shape == (BATCH, STEPS, WIDTH)


def test_the_columns_are_in_the_order_feature_index_reports(batch):
    assert feature_index(batch) == [('numeric', 0), ('numeric', 1), ('lookup', 0)]


def test_masking_a_timestep_blanks_that_row_and_nothing_else(batch):
    reference, _ = saliency(StubModel(), batch)
    scores, _ = saliency(StubModel(), batch, timestep=1, index=feature_index(batch))
    assert (scores[:, 1] == 0).all()
    untouched = [t for t in range(STEPS) if t != 1]
    assert torch.allclose(scores[:, untouched], reference[:, untouched], atol=1e-6)


def test_masking_a_cell_leaves_the_rest_of_its_timestep_alone(batch):
    """The reference deletes one (timestep, feature) cell, not a feature across all time."""
    index = feature_index(batch)
    reference, _ = saliency(StubModel(), batch)
    scores, _ = saliency(StubModel(), batch, timestep=2, feature=2, index=index)

    assert (scores[:, 2, 2] == 0).all()
    # The same feature at other timesteps survives -- this is what separates the reference's
    # reading from the pseudocode's ``X_i,: = 0``.
    others = [t for t in range(STEPS) if t != 2]
    assert torch.allclose(scores[:, others, 2], reference[:, others, 2], atol=1e-6)
    assert torch.allclose(scores[:, 2, :2], reference[:, 2, :2], atol=1e-6)


def test_masking_clears_the_indicator_with_the_value(batch):
    """An observed zero is not a deleted feature, and the model reads the two differently."""
    index = feature_index(batch)
    model = StubModel()
    captured = {}
    original_forward = model.forward

    def spy(inner_batch):
        captured['indicators'] = inner_batch['val_data']['numeric']['indicators'].clone()
        return original_forward(inner_batch)

    model.forward = spy
    saliency(model, batch, timestep=2, index=index)
    assert (captured['indicators'][:, 2] == 0).all()
    assert (captured['indicators'][:, 0] == 1).all()
    # And the caller's batch is untouched, so a perturbed pass cannot leak into the next one.
    assert (batch['val_data']['numeric']['indicators'] == 1).all()


def test_the_open_gate_measures_every_cell(batch):
    """The default cost model: 1 + T + T*N, every cell measured, nothing tied at the floor."""
    result = tsr_scores(StubModel(), batch)
    assert result['gate'].all()
    assert result['passes'] == 1 + STEPS + STEPS * N_FEATURES


def test_tsr_recomputes_the_feature_profile_at_every_passing_timestep(batch):
    """With a gate, 1 + T + (passing timesteps) x N.

    Not 1 + T + N, which is what the pseudocode's feature mask would give: the reference masks
    one cell inside the time loop, so the profile is recomputed per timestep.
    """
    result = tsr_scores(StubModel(), batch, quantile=0.55)
    passing = int(result['gate'].any(dim=0).sum())
    assert result['passes'] == 1 + STEPS + passing * N_FEATURES
    assert 0 < passing < STEPS, 'the 0.55 gate should admit some timesteps and reject some'


def test_the_score_is_the_scaled_time_term_times_the_raw_feature_term(batch):
    result = tsr_scores(StubModel(), batch)
    expected = (result['time_contribution'].unsqueeze(-1)
                * result['feature_contribution'])
    assert result['scores'].shape == (BATCH, STEPS, N_FEATURES)
    assert torch.allclose(result['scores'], expected, atol=1e-6)
    # The feature term is the raw delta, not scaled within the timestep: nothing is pinned to
    # zero by construction, and the magnitudes stay comparable between timesteps.
    assert (result['feature_contribution'] > 0).all()


def test_the_time_term_is_divided_by_its_maximum_not_min_max_scaled(batch):
    """Bounded above by 1, but the origin and the ratios between timesteps are left alone."""
    result = tsr_scores(StubModel(), batch)
    delta, scaled = result['delta_time'], result['time_contribution']
    assert torch.allclose(scaled.max(dim=-1).values, torch.ones(BATCH), atol=1e-6)
    assert (scaled.min(dim=-1).values > 0).all(), 'min-max would pin the weakest step at 0'
    assert torch.allclose(scaled, delta / delta.max(dim=-1, keepdim=True).values, atol=1e-6)


def test_gated_timesteps_are_erased_not_floored(batch):
    """The reference's 0.1 floor was calibrated against a [0, 1] feature term, which this is not."""
    result = tsr_scores(StubModel(), batch, quantile=0.55)
    gated = ~result['gate']
    assert gated.any(), 'fixture should produce at least one gated timestep'
    assert (result['feature_contribution'][gated] == 0).all()


def test_the_gate_quantile_sets_the_cost(batch):
    """Raising the quantile to 1.0 admits nothing, so no feature pass is taken at all."""
    result = tsr_scores(StubModel(), batch, quantile=1.0)
    assert result['passes'] == 1 + STEPS
    assert (result['feature_contribution'] == 0).all()


def test_the_lookup_gradient_is_what_the_token_pass_consumes(batch):
    """TSR's unmasked pass already produces stage 1's gradient, so nothing recomputes it."""
    result = tsr_scores(StubModel(), batch)
    grad = result['lookup_grads'][0]
    # One row per (record, timestep); token_attributions takes (N, embed_dim) for the cells
    # being rendered.
    assert grad.shape == (BATCH, STEPS, WIDTH)
    assert grad[0, :2].shape == (2, WIDTH)


# ---------------------------------------------------------------------------
# Batching, and the real model
# ---------------------------------------------------------------------------

def test_chunking_is_exact_not_an_approximation(batch):
    """One perturbation per forward and all of them in one forward must agree.

    The claim batching rests on is that no item's output depends on another's inputs, so a
    summed target backpropagated once gives every replica the gradient it would have had alone.
    If that were wrong anywhere -- a normalization over the batch, a pooled statistic -- this is
    where it would show.
    """
    serial = tsr_scores(StubModel(), batch, chunk=1)
    parallel = tsr_scores(StubModel(), batch, chunk=1024)
    for key in ('scores', 'delta_time', 'feature_contribution'):
        assert torch.allclose(serial[key], parallel[key], atol=1e-6), key
    assert serial['passes'] == parallel['passes']


def _real_model_and_batch(steps=4, n_numeric=2, width=3, text_width=8,
                          d_model=16, n_causes=2, n_bins=3, batch_size=1):
    """A real `MixedClassifier` with a DeepHit head, and a batch in the collated shape.

    Small, but every piece is the deployed one: the value encoder, the event branch, the lookup
    family's pooled slots and the competing-risks head. The stub above pins TSR's arithmetic
    against a closed form; this pins that TSR runs against the thing it will actually explain.
    """
    from TransEHR2.models import MixedClassifier
    from TransEHR2.modules import DeepHitHead, EventDataEncoder, ValueDataEncoder

    features = n_numeric + 1
    feat_dim = n_numeric * width + text_width
    encoding = MixedClassifier.encoding_width(d_event_enc=8, d_val_enc=d_model,
                                              d_statics=0)
    model = MixedClassifier(
        event_encoder=EventDataEncoder(num_types=2, d_model=8, d_inner=16, n_layers=1,
                                       n_head=2, d_k=4, d_v=4, dropout=0.0),
        val_encoder=ValueDataEncoder(
            n_features=features, feat_dim=feat_dim, d_model=d_model, n_heads=2,
            n_encoder_blocks=1, dim_feedforward=d_model, dropout=0.0,
            norm='LayerNorm'),
        d_event_enc=8, d_val_enc=d_model, d_statics=0,
        num_classes=n_causes * n_bins, aggr='mean', use_lookup=True,
        head=DeepHitHead(d_in=encoding, n_causes=n_causes, n_bins=n_bins,
                         d_shared=8, d_cause=4, dropout=0.0))
    model.eval()

    collated = {
        'event_data': {
            'indicators': torch.ones(batch_size, steps, 2),
            'times': torch.arange(steps).float().repeat(batch_size, 1),
            'masks': torch.ones(batch_size, steps),
        },
        'val_data': {
            'times': torch.arange(steps).float().repeat(batch_size, 1),
            'masks': torch.ones(batch_size, steps),
            'numeric': {
                'indicators': torch.ones(batch_size, steps, n_numeric),
                'values': [torch.randn(batch_size, steps, width)
                           for _ in range(n_numeric)],
            },
            'lookup': {
                'indicators': torch.ones(batch_size, steps, 1),
                'slot_values': [torch.randn(batch_size, steps, text_width)],
                'doses': [None], 'masks': [None],
            },
        },
    }
    return model, collated, features


def test_tsr_runs_against_the_real_classifier_and_the_readmission_target():
    """End to end on a MixedClassifier: the deployed scalar, the deployed head, real shapes."""
    from TransEHR2.survival import cif_logit_target

    torch.manual_seed(0)
    steps, n_causes, n_bins = 4, 2, 3
    model, collated, features = _real_model_and_batch(steps=steps, n_causes=n_causes,
                                                      n_bins=n_bins)
    target = cif_logit_target(0, n_bins - 1, n_causes, n_bins)

    result = tsr_scores(model, collated, target=target)
    assert result['scores'].shape == (1, steps, features)
    assert torch.isfinite(result['scores']).all()
    assert result['passes'] == 1 + steps + steps * features
    # The event branch is tiled alongside the value branch; a mismatch there would have raised
    # on the concatenation rather than reaching here.
    assert result['lookup_grads'][0].shape == (1, steps, 8)


def test_the_real_classifier_gives_the_same_scores_batched_or_not():
    """The equivalence that matters, on a model with attention across the time axis."""
    from TransEHR2.survival import cif_logit_target

    torch.manual_seed(0)
    model, collated, _ = _real_model_and_batch()
    target = cif_logit_target(0, 2, 2, 3)
    serial = tsr_scores(model, collated, target=target, chunk=1)
    parallel = tsr_scores(model, collated, target=target, chunk=256)
    assert torch.allclose(serial['scores'], parallel['scores'], atol=1e-5)
