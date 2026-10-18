"""The rotary arm, and what keeps it comparable to the additive one.

The two arms exist to be compared, so most of what matters here is not
that RoPE works but that it differs from the additive encoding in the
*mechanism* and nothing else: the same ladder, the same band count, the
same dropout and padding mask.

Timestamps are days in this fork, non-positive, zero at the index time.
MIMIC's are hours, so its four ladder constants are out by 24 and none of
them appear here.
"""

import ast
import math
import os

import pytest
import torch
import yaml

from TransEHR2.layers import (RotaryTemporalEncoding, TemporalPositionEncoding,
                              build_frequency_ladder)
from TransEHR2.modules import (EventDataEncoder, ValueDataEncoder,
                               _resolve_position_encoding)


# The measured bounds, as the configs carry them.
P_MIN, P_MAX = 0.00138889, 412357.0
BOUNDS = {'ladder_p_min': P_MIN, 'ladder_p_max': P_MAX}

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
CONFIGS = os.path.join(REPO, 'TransEHR2', 'configs', 'experiments')


def value_encoder(**kwargs):
    base = dict(n_features=3, feat_dim=12, d_model=256, n_heads=2,
                n_encoder_blocks=1, dim_feedforward=256, norm='LayerNorm',
                normalize_before=True)
    return ValueDataEncoder(**{**base, **kwargs})


def event_encoder(**kwargs):
    base = dict(num_types=1, d_model=256, d_inner=256, n_layers=1, n_head=2,
                d_k=128, d_v=64, dropout=0.1, normalize_before=True)
    return EventDataEncoder(**{**base, **kwargs})


# --- the ladder ---------------------------------------------------------

def test_the_ladder_spans_the_bounds_it_is_given():
    ladder = build_frequency_ladder(128, P_MIN, P_MAX)
    assert ladder.numel() == 128
    assert ladder[0].item() == pytest.approx(P_MIN / (2 * math.pi), rel=1e-6)
    assert ladder[-1].item() == pytest.approx(P_MAX / (2 * math.pi), rel=1e-6)
    assert torch.all(ladder[1:] > ladder[:-1]), 'ascending'


def test_the_ladder_is_log_uniform_in_period():
    """Equal ratios between neighbours, not equal differences."""
    ladder = build_frequency_ladder(8, 1.0, 1000.0)
    ratios = (ladder[1:] / ladder[:-1]).tolist()
    assert ratios == pytest.approx([ratios[0]] * len(ratios), rel=1e-5)


def test_a_one_band_ladder_sits_at_p_min():
    """No range to span, and dividing by n_bands - 1 would be a zero div."""
    assert build_frequency_ladder(1, 2.0, 99.0).item() == pytest.approx(
        2.0 / (2 * math.pi), rel=1e-6)


@pytest.mark.parametrize('n_bands,p_min,p_max', [
    (0, 1.0, 10.0), (4, 0.0, 10.0), (4, -1.0, 10.0), (4, 10.0, 10.0),
    (4, 10.0, 1.0),
])
def test_a_malformed_ladder_is_refused(n_bands, p_min, p_max):
    with pytest.raises(ValueError):
        build_frequency_ladder(n_bands, p_min, p_max)


# --- what RoPE is for ---------------------------------------------------

def test_the_score_depends_on_the_gap_not_the_origin():
    """The whole reason for the arm.

    Two records a fixed distance apart must score the same whether they
    sit near the index time or eighteen years before it.
    """
    rope = RotaryTemporalEncoding(2, 128, P_MIN, P_MAX)
    torch.manual_seed(0)
    q, k = torch.randn(1, 2, 4, 128), torch.randn(1, 2, 4, 128)

    def scores(times):
        rq, rk = rope(q, k, torch.tensor([times]))
        return torch.einsum('bhid,bhjd->bhij', rq, rk)

    here = scores([-900.0, -400.0, -30.0, 0.0])
    long_ago = scores([-6400.0, -5900.0, -5530.0, -5500.0])
    assert torch.allclose(here, long_ago, atol=5e-3)


def test_a_different_gap_gives_a_different_score():
    """Lest the invariance above be satisfied by scoring nothing."""
    rope = RotaryTemporalEncoding(2, 128, P_MIN, P_MAX)
    torch.manual_seed(0)
    q, k = torch.randn(1, 2, 4, 128), torch.randn(1, 2, 4, 128)

    def scores(times):
        rq, rk = rope(q, k, torch.tensor([times]))
        return torch.einsum('bhid,bhjd->bhij', rq, rk)

    assert not torch.allclose(scores([-900.0, -400.0, -30.0, 0.0]),
                              scores([-900.0, -400.0, -30.0, -10.0]),
                              atol=1e-3)


def test_rotation_preserves_the_norm():
    """It is a rotation, so it may move a vector but not resize it."""
    rope = RotaryTemporalEncoding(2, 128, P_MIN, P_MAX)
    torch.manual_seed(1)
    q = torch.randn(2, 2, 5, 128)
    rotated, _ = rope(q, q.clone(), torch.randn(2, 5) * 1000.0)
    assert torch.allclose(rotated.norm(dim=-1), q.norm(dim=-1), atol=1e-4)


def test_the_phase_survives_a_timestamp_decades_out():
    """``t / lambda`` reaches order 1e7 radians against the fastest band.

    Dividing there in float32 loses the phase outright, which is why the
    angle is reduced modulo the period in float64 first. The reduction
    preserves whatever precision the timestamp arrived with -- it cannot
    recover more, and ``positions`` is float32 by the time it gets here --
    so the claim is against a float64 reference taken from the *same*
    stored value, not against the unrounded one.
    """
    rope = RotaryTemporalEncoding(1, 4, P_MIN, P_MAX)
    lam = rope.lambdas[0, 0].item()
    period = lam * 2 * math.pi

    positions = torch.tensor([[-period * 1e6 - lam * (math.pi / 2)]])
    stored = float(positions.item())
    assert abs(stored / lam) > 1e6, 'the regime this is about'

    # `%` follows the divisor's sign, as torch.remainder does, so a
    # negative timestamp reduces into [0, period) the same way.
    reference = math.cos((stored % period) / lam)
    reduced = rope._sinusoids(positions, torch.float32)[0][0, 0, 0, 0].item()
    assert reduced == pytest.approx(reference, abs=1e-5)

    # And the thing it is protecting against: the same division in float32.
    naive = math.cos(float(torch.tensor(stored / lam).item()))
    assert abs(naive - reference) > 1e-2, 'the reduction is what saves it'


def test_positions_are_required_and_must_match_the_sequence():
    rope = RotaryTemporalEncoding(2, 128, P_MIN, P_MAX)
    q = k = torch.randn(1, 2, 4, 128)
    with pytest.raises(ValueError, match='timestamps'):
        rope(q, k, None)
    with pytest.raises(ValueError, match='sequence length'):
        rope(q, k, torch.zeros(1, 7))


def test_an_odd_head_width_is_refused():
    """Each band owns a channel pair."""
    with pytest.raises(ValueError, match='even'):
        RotaryTemporalEncoding(2, 127, P_MIN, P_MAX)


# --- parity with the additive arm ---------------------------------------

def test_the_bands_are_partitioned_across_heads_not_repeated():
    """One ladder cut into contiguous slices, head 0 fastest.

    Repeating the ladder per head would leave the rotary arm with a
    spectral deficit, and an additive win could be read as a win for more
    bands rather than for the mechanism.
    """
    rope = RotaryTemporalEncoding(2, 128, P_MIN, P_MAX)
    assert rope.lambdas.shape == (2, 64)
    flat = rope.lambdas.flatten()
    assert torch.all(flat[1:] > flat[:-1]), 'one ascending ladder'
    assert rope.lambdas[0].max() < rope.lambdas[1].min(), 'head 0 is fastest'


def test_the_band_count_matches_the_additive_arm():
    """``n_head * d_head == d_model`` is what buys parity.

    THP_ENCODER_D_K is 128 rather than 64 for this reason; lowering it
    reopens the deficit in silence.
    """
    d_model = 256
    additive = TemporalPositionEncoding(d_model=d_model, p_min=P_MIN,
                                        p_max=P_MAX)
    rope = RotaryTemporalEncoding(2, 128, P_MIN, P_MAX)
    assert rope.band_count == additive.position_encoding.numel() == d_model // 2


def test_the_ladder_buffer_is_not_persistent():
    """Re-spanning changes the values but not the shape.

    A persistent buffer would let a checkpoint written before a re-measure
    overwrite the new ladder with the old one -- no error, and the arm
    running is not the arm intended.
    """
    rope = RotaryTemporalEncoding(2, 128, P_MIN, P_MAX)
    additive = TemporalPositionEncoding(d_model=256, p_min=P_MIN, p_max=P_MAX)
    assert 'lambdas' not in rope.state_dict()
    assert 'position_encoding' not in additive.state_dict()


def test_the_additive_layer_goes_passive_but_keeps_its_mask():
    """Under rope the rotation happens in the attention, but the dropout
    and the zeroing of padded positions have to stay, or the two arms
    differ in more than their encoding."""
    passive = TemporalPositionEncoding(d_model=4, dropout=None, p_min=P_MIN,
                                       p_max=P_MAX, additive=False)
    x = torch.ones(1, 3, 4)
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    out = passive(x, torch.tensor([[-5.0, -2.0, 0.0]]), mask)
    assert torch.equal(out[0, :2], x[0, :2]), 'nothing added'
    assert torch.equal(out[0, 2], torch.zeros(4)), 'padding still zeroed'


def test_one_bound_without_the_other_is_refused():
    with pytest.raises(ValueError, match='together'):
        TemporalPositionEncoding(d_model=8, p_min=P_MIN)


# --- the arm switch -----------------------------------------------------

def test_additive_passes_a_caller_transform_through_untouched():
    marker = torch.nn.Identity()
    assert _resolve_position_encoding('additive', 2, 128, None, None,
                                      marker) is marker


def test_rope_will_not_share_the_seam():
    with pytest.raises(ValueError, match='one or the other'):
        _resolve_position_encoding('rope', 2, 128, P_MIN, P_MAX,
                                   torch.nn.Identity())


def test_rope_without_bounds_is_refused_rather_than_defaulted():
    """'There is no defensible default': the bounds follow from the gaps
    the encoder actually sees."""
    with pytest.raises(ValueError, match='ladder_p_min'):
        _resolve_position_encoding('rope', 2, 128, None, None, None)


def test_an_unknown_scheme_does_not_fall_back_to_additive():
    with pytest.raises(ValueError, match='expected one of'):
        _resolve_position_encoding('alibi', 2, 128, P_MIN, P_MAX, None)


@pytest.mark.parametrize('build', [value_encoder, event_encoder])
def test_both_encoders_carry_both_arms(build):
    assert build(position_encoding='additive').query_key_transform is None
    roped = build(position_encoding='rope', **BOUNDS)
    assert isinstance(roped.query_key_transform, RotaryTemporalEncoding)
    assert roped.query_key_transform.band_count == 128


def test_rope_under_batchnorm_is_refused():
    """That path uses torch.nn.MultiheadAttention, which never exposes q
    and k, so the rotation would silently not happen."""
    with pytest.raises(ValueError, match='LayerNorm'):
        value_encoder(norm='BatchNorm', position_encoding='rope', **BOUNDS)


# --- the configs and the construction sites -----------------------------

@pytest.mark.parametrize('name', ['experiment1_baseline', 'experiment2_text',
                                  'experiment3_nohistory'])
def test_every_config_declares_an_arm_and_four_float_bounds(name):
    """A bound written in unsigned scientific notation parses as a *string*
    under YAML 1.1 and reaches the ladder builder as one."""
    config = yaml.safe_load(open(os.path.join(CONFIGS, f'{name}.yaml')))
    assert config['POSITION_ENCODING'] in ('additive', 'rope')
    for key in ('VALUE_LADDER_P_MIN', 'VALUE_LADDER_P_MAX',
                'EVENT_LADDER_P_MIN', 'EVENT_LADDER_P_MAX'):
        assert isinstance(config[key], float), f'{key} is not a float'
    assert config['VALUE_LADDER_P_MAX'] > config['VALUE_LADDER_P_MIN']
    assert config['EVENT_LADDER_P_MAX'] > config['EVENT_LADDER_P_MIN']


@pytest.mark.parametrize('script', ['fit_transehr2_deephit.py',
                                    'dump_transehr2_deephit_predictions.py',
                                    'render_token_attributions.py'])
def test_each_construction_site_takes_its_own_streams_ladder(script):
    """A value encoder on the event ladder would be wrong by three orders
    of magnitude and would still train.

    Read off the syntax tree rather than by grep, so a site added later
    cannot pass by sitting on the wrong line.
    """
    path = os.path.join(REPO, 'scripts', script)
    tree = ast.parse(open(path).read())
    wanted = {'ValueDataEncoder': 'VALUE_LADDER', 'EventDataEncoder': 'EVENT_LADDER'}

    seen = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, 'id', None) or getattr(node.func, 'attr', None)
        if name not in wanted:
            continue
        seen += 1
        source = ast.dump(node)
        assert 'position_encoding' in source, f'{script}: {name} has no arm'
        assert wanted[name] in source, (
            f'{script}: {name} does not take {wanted[name]}_*')
        other = wanted['EventDataEncoder' if name == 'ValueDataEncoder'
                       else 'ValueDataEncoder']
        assert other not in source, f'{script}: {name} takes the wrong ladder'
    assert seen == 2, f'{script}: expected two encoder sites, found {seen}'
