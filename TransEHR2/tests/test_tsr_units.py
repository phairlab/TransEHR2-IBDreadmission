"""The phase-2 additions to TSR: what a deletion is, the drug units, and the target.

`test_tsr.py` pins TSR's arithmetic against a closed-form stub. This file pins the pieces the
faithfulness experiments rest on, against the deployed classifier where the question is what the
model sees:

* a deleted timestep is *masked*, and a timestep a deletion empties is masked too;
* a drug is deleted by its ATC code, which must equal the record with that drug never
  dispensed -- duplicate slots included;
* a drug cell's score splits among its codes and sums back to the cell;
* a timestep with one occupied cell takes `delta_time` without a pass, which is only right if
  the pass would have returned exactly that;
* the empty input (every timestep masked) runs, forward and backward, under every aggregation;
* cropping the left padding changes nothing the model computes;
* the interpolated-CIF log-odds is the cut log-odds at a cut, and its gradient is right.
"""

import pytest
import torch

from TransEHR2.tsr import (_deltas, _with_input_gradients, _zero_many, crop_to_length,
                          feature_index, outputs_many, saliency_many,
                          tsr_scores)
from TransEHR2.utils import pool_lookup_slots

STEPS, WIDTH, TEXT_WIDTH, DRUG_WIDTH, SLOTS = 5, 3, 8, 5, 4
N_CAUSES, N_BINS = 2, 3
# The drug table: row k is code k's vector. Codes repeat across slots by sharing a row, which
# is how one ATC code in two slots of a dispensation reaches the model.
DRUG_TABLE = torch.randn(10, DRUG_WIDTH, generator=torch.Generator().manual_seed(1))


def _drug_slots(codes, doses):
    """(T, S) codes and doses to the dense slot tensors, -1 marking an empty slot."""
    codes = torch.tensor(codes)
    filled = (codes >= 0).float()
    values = DRUG_TABLE[codes.clamp(min=0)] * filled.unsqueeze(-1)
    return values, torch.tensor(doses, dtype=torch.float32) * filled, filled, codes


def _model_and_batch(aggr='mean', seed=0):
    """A small MixedClassifier without the event branch, and an episode carrying every family.

    Timestep layout, oldest first:
      0: padding (masked)
      1: one numeric cell alone -- the single-cell bypass
      2: two numerics and a text cell
      3: a drug cell holding codes 3, 3 (two slots) and 7, plus a numeric
      4: a drug cell holding code 5 alone
    """
    from TransEHR2.models import MixedClassifier
    from TransEHR2.modules import DeepHitHead, ValueDataEncoder

    torch.manual_seed(seed)
    d_model = 16
    n_features = 2 + 2
    model = MixedClassifier(
        event_encoder=None,
        val_encoder=ValueDataEncoder(
            n_features=n_features, feat_dim=2 * WIDTH + TEXT_WIDTH + DRUG_WIDTH,
            d_model=d_model, n_heads=2, n_encoder_blocks=1, dim_feedforward=d_model,
            dropout=0.0, norm='LayerNorm'),
        d_event_enc=0, d_val_enc=d_model, d_statics=0,
        num_classes=N_CAUSES * N_BINS, aggr=aggr, use_lookup=True,
        head=DeepHitHead(d_in=d_model, n_causes=N_CAUSES, n_bins=N_BINS,
                         d_shared=[8], d_cause=[4], dropout=0.0))
    model.eval()

    numeric_ind = torch.zeros(1, STEPS, 2)
    numeric_ind[0, 1, 0] = 1
    numeric_ind[0, 2, :] = 1
    numeric_ind[0, 3, 1] = 1
    numeric_values = [torch.randn(1, STEPS, WIDTH) * numeric_ind[..., f:f + 1]
                      for f in range(2)]

    text_ind = torch.zeros(1, STEPS)
    text_ind[0, 2] = 1
    text = torch.randn(1, STEPS, TEXT_WIDTH) * text_ind.unsqueeze(-1)

    codes = [[-1] * SLOTS] * 3 + [[3, 3, 7, -1], [5, -1, -1, -1]]
    doses = [[0.0] * SLOTS] * 3 + [[1.0, 0.5, 2.0, 0.0], [1.5, 0.0, 0.0, 0.0]]
    values, dose, filled, slot_codes = _drug_slots(codes, doses)
    drug_ind = (filled.sum(dim=-1) > 0).float()

    masks = torch.ones(1, STEPS)
    masks[0, 0] = 0
    batch = {
        'event_data': {'indicators': torch.zeros(1, STEPS, 1),
                       'times': torch.zeros(1, STEPS), 'masks': torch.zeros(1, STEPS)},
        'val_data': {
            'times': torch.arange(STEPS).float().unsqueeze(0),
            'masks': masks,
            'numeric': {'indicators': numeric_ind, 'values': numeric_values},
            'lookup': {
                'indicators': torch.stack([text_ind, drug_ind.unsqueeze(0)], dim=-1),
                'slot_values': [text, values.unsqueeze(0)],
                'doses': [None, dose.unsqueeze(0)],
                'masks': [None, filled.unsqueeze(0)],
                'slot_indices': [torch.where(text_ind > 0, 0, -1).long(),
                                 slot_codes.unsqueeze(0)],
            },
        },
    }
    return model, batch


def _target():
    from TransEHR2.survival import cif_logit_target
    return cif_logit_target(0, N_BINS - 1, N_CAUSES, N_BINS)


DRUG = 3  # the drug column: numeric 0, numeric 1, text, drug


# ---------------------------------------------------------------------------
# What a deletion is
# ---------------------------------------------------------------------------

def test_deleting_a_timestep_masks_it_and_leaves_the_batch_alone():
    _, batch = _model_and_batch()
    before = batch['val_data']['masks'].clone()
    masked, leaves = _with_input_gradients(batch, copies=1)
    _zero_many(masked, leaves, feature_index(batch), [(2, None)], 1)
    assert masked['val_data']['masks'][0, 2] == 0
    # At one copy `_repeat` returns the caller's tensor; the clone is what keeps this true.
    assert torch.equal(batch['val_data']['masks'], before)


def test_a_cell_deletion_masks_its_timestep_only_when_it_empties_it():
    _, batch = _model_and_batch()
    index = feature_index(batch)
    masked, leaves = _with_input_gradients(batch, copies=2)
    # Replica 0 deletes the lone cell at t=1; replica 1 one of three cells at t=2.
    _zero_many(masked, leaves, index, [(1, 0), (2, 0)], 1)
    masks = masked['val_data']['masks']
    assert masks[0, 1] == 0
    assert masks[1, 2] == 1


def test_deleting_a_drug_is_the_record_without_it():
    """Code 3 sits in two slots; deleting it must clear both and re-pool over what is left."""
    _, batch = _model_and_batch()
    index = feature_index(batch)
    masked, leaves = _with_input_gradients(batch, copies=1)
    _zero_many(masked, leaves, index, [(3, DRUG, 3)], 1)

    lookup = batch['val_data']['lookup']
    kept = lookup['masks'][1].clone()
    kept[0, 3, :2] = 0
    expected = pool_lookup_slots(lookup['slot_values'][1], lookup['doses'][1], kept)
    assert torch.allclose(leaves['lookup'][1][0, 3], expected[0, 3])
    assert masked['val_data']['lookup']['indicators'][0, 3, 1] == 1
    assert masked['val_data']['masks'][0, 3] == 1


def test_deleting_a_timesteps_only_drug_clears_the_cell_and_masks_the_timestep():
    _, batch = _model_and_batch()
    masked, leaves = _with_input_gradients(batch, copies=1)
    _zero_many(masked, leaves, feature_index(batch), [(4, DRUG, 5)], 1)
    assert not leaves['lookup'][1][0, 4].any()
    assert masked['val_data']['lookup']['indicators'][0, 4, 1] == 0
    assert masked['val_data']['masks'][0, 4] == 0


def test_a_drug_absent_from_the_timestep_changes_nothing():
    _, batch = _model_and_batch()
    masked, leaves = _with_input_gradients(batch, copies=1)
    original = leaves['lookup'][1].detach().clone()
    _zero_many(masked, leaves, feature_index(batch), [(3, DRUG, 9)], 1)
    assert torch.equal(leaves['lookup'][1], original)


# ---------------------------------------------------------------------------
# TSR with the new units
# ---------------------------------------------------------------------------

def test_a_lone_cell_takes_delta_time_which_is_what_its_pass_would_give():
    model, batch = _model_and_batch()
    result = tsr_scores(model, batch, target=_target())
    alphas = [1.0]
    direct = _deltas(model, batch, [(1, 0)], result['baseline'], _target(),
                     result['index'], 64, alphas)
    assert torch.allclose(direct[0, 0], result['delta_time'][0, 1], rtol=1e-5, atol=1e-7)
    assert result['feature_contribution'][0, 1, 0] == pytest.approx(1.0)


def test_drug_scores_split_their_cell_and_sum_back_to_it():
    model, batch = _model_and_batch()
    result = tsr_scores(model, batch, target=_target(), ig_steps=4)
    drugs = result['drugs']
    assert set(zip(drugs['step'].tolist(), drugs['code'].tolist())) == {(3, 3), (3, 7), (4, 5)}
    for step in (3, 4):
        mine = drugs['step'] == step
        assert drugs['share'][mine].sum() == pytest.approx(1.0, abs=1e-6)
        assert drugs['score'][mine].sum() == pytest.approx(
            float(result['scores'][0, step, DRUG]), rel=1e-5, abs=1e-9)
        # The pool is linear in its slots, so the codes' parts of R(X) are exact.
        assert drugs['baseline'][mine].sum() == pytest.approx(
            float(result['baseline'][0, step, DRUG]), rel=1e-4, abs=1e-7)
    # The lone code takes the whole cell without a pass.
    lone = drugs['step'] == 4
    assert drugs['share'][lone].item() == pytest.approx(1.0)


def test_the_bypass_and_the_drug_passes_are_counted():
    model, batch = _model_and_batch()
    result = tsr_scores(model, batch, target=_target())
    # 1 baseline + 5 timesteps; cells in multi-cell timesteps: t=2 has 3, t=3 has 2; the drug
    # cell at t=3 holds two codes. The lone cells at t=1 and t=4 take no pass.
    assert result['passes'] == 1 + STEPS + 3 + 2 + 2


# ---------------------------------------------------------------------------
# The empty input and cropping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('aggr', ['mean', 'max', 'none'])
def test_the_empty_input_runs_forward_and_backward(aggr):
    model, batch = _model_and_batch(aggr=aggr)
    empty = [(t, None) for t in range(STEPS)]
    out = outputs_many(model, batch, [empty, None])
    assert torch.isfinite(out).all()

    scores, grads = saliency_many(model, batch, [empty], target=_target(),
                                  index=feature_index(batch))
    assert torch.isfinite(scores).all()
    assert all(torch.isfinite(g).all() for g in grads)


def test_the_empty_input_is_one_constant_whatever_the_episode():
    model, first = _model_and_batch(seed=0)
    _, second = _model_and_batch(seed=1)
    empty = [(t, None) for t in range(STEPS)]
    a = outputs_many(model, first, [empty])
    b = outputs_many(model, second, [empty])
    assert torch.allclose(a, b, atol=1e-6)


def test_the_empty_input_runs_in_a_mixed_batch():
    """Two episodes per replica, one of them emptied by a whole-timestep sweep."""
    model, batch = _model_and_batch()
    two = {key: value for key, value in batch.items()}

    def stack(obj):
        if torch.is_tensor(obj):
            return torch.cat([obj, obj], dim=0)
        if isinstance(obj, dict):
            return {k: stack(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [stack(v) for v in obj]
        return obj

    two = stack(two)
    two['val_data']['masks'][1] = 0
    out = outputs_many(model, two, [None])
    assert torch.isfinite(out).all()


def test_cropping_drops_the_padding_and_changes_nothing():
    model, batch = _model_and_batch()
    full = outputs_many(model, batch, [None])
    cropped, start = crop_to_length(batch)
    assert start == 1
    assert cropped['val_data']['masks'].shape == (1, STEPS - 1)
    assert cropped['val_data']['lookup']['slot_values'][1].shape[1] == STEPS - 1
    assert torch.allclose(outputs_many(model, cropped, [None]), full, atol=1e-6)


def test_outputs_many_is_the_model_on_the_perturbed_episode():
    """Deleting t=2 forward-only equals the model on an episode built with t=2 masked."""
    model, batch = _model_and_batch()
    perturbed = outputs_many(model, batch, [(2, None)])[0]

    _, by_hand = _model_and_batch()
    val = by_hand['val_data']
    val['masks'][0, 2] = 0
    val['numeric']['indicators'][0, 2] = 0
    for values in val['numeric']['values']:
        values[0, 2] = 0
    val['lookup']['indicators'][0, 2] = 0
    val['lookup']['slot_values'][0][0, 2] = 0
    with torch.no_grad():
        expected = model(by_hand)
    assert torch.allclose(perturbed, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# The interpolated target
# ---------------------------------------------------------------------------

def _grid():
    from TransEHR2.survival import TimeGrid
    return TimeGrid(cuts_days=[10.0, 20.0, 40.0], causes=['readmission', 'death'],
                    brier_integration_days=40.0)


def test_the_interpolated_logit_is_the_cut_logit_at_a_cut():
    from TransEHR2.survival import cif_logit_target, interpolated_cif_logit_target

    grid = _grid()
    logits = torch.randn(4, grid.n_causes * grid.n_bins, dtype=torch.float64)
    for bin_index, cut in enumerate(grid.edges_days[1:]):
        exact = cif_logit_target(0, bin_index, grid.n_causes, grid.n_bins)(logits)
        read = interpolated_cif_logit_target(0, float(cut), grid)(logits)
        assert read.item() == pytest.approx(exact.item(), rel=1e-9, abs=1e-9)


def test_the_interpolated_logit_has_the_right_gradient_between_cuts():
    from TransEHR2.survival import interpolated_cif_logit_target

    grid = _grid()
    target = interpolated_cif_logit_target(0, 15.0, grid)
    logits = torch.randn(3, grid.n_causes * grid.n_bins, dtype=torch.float64,
                         requires_grad=True)
    assert torch.autograd.gradcheck(target, (logits,))
    target(logits).backward()
    assert logits.grad.abs().sum() > 0


def test_the_interpolated_logit_rejects_a_horizon_off_the_grid():
    from TransEHR2.survival import interpolated_cif_logit_target

    grid = _grid()
    with pytest.raises(ValueError, match='horizon'):
        interpolated_cif_logit_target(0, 50.0, grid)
    with pytest.raises(ValueError, match='horizon'):
        interpolated_cif_logit_target(0, 0.0, grid)
