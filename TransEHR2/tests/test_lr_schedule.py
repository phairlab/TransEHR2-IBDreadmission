"""The learning rate schedule, expressed as a half-life in epochs.

The decay factor used to be applied on a fixed multi-epoch cadence -- fifty
epochs pretraining, twenty finetuning -- recorded nowhere, so the factor alone
did not describe a schedule. A half-life is in the same unit as the run length
it has to be judged against, and the scheduler now steps every epoch, so the
two statements are the same one.
"""

import numpy as np
import pytest
import yaml

from pathlib import Path

# routines imports tensorboard at module scope, which is absent from some
# environments. It surfaces as an AttributeError rather than an ImportError,
# so importorskip does not catch it. The config checks below need none of this
# and run either way, which is the half that would otherwise go unchecked
# wherever the dependency is missing.
try:
    from TransEHR2.routines import (
        IMPROVEMENT_THRESHOLD,
        is_improvement,
        resolve_decay_factor,
    )
except (ImportError, AttributeError) as exc:  # pragma: no cover
    ROUTINES_IMPORT_ERROR = exc
    IMPROVEMENT_THRESHOLD = None
else:
    ROUTINES_IMPORT_ERROR = None

needs_routines = pytest.mark.skipif(
    ROUTINES_IMPORT_ERROR is not None,
    reason=f'TransEHR2.routines unimportable: {ROUTINES_IMPORT_ERROR}'
)

CONFIGS = Path(__file__).resolve().parents[1] / 'configs' / 'experiments'


@needs_routines
def test_the_factor_halves_the_rate_over_the_half_life():
    gamma = resolve_decay_factor(100)
    assert gamma ** 100 == pytest.approx(0.5)


@needs_routines
def test_no_half_life_leaves_the_rate_constant():
    """None and non-positive both mean 'no decay' rather than an error: a flat
    schedule is a legitimate arm of a sweep, not a misconfiguration."""
    for value in (None, 0, -1):
        assert resolve_decay_factor(value) == 1.0


@needs_routines
def test_an_improvement_has_to_clear_a_relative_threshold():
    """Without one, movement in the fifth decimal place holds a run open and
    the best epoch tracks numeric noise rather than convergence."""
    assert is_improvement(0.5, np.inf)
    assert is_improvement(0.4, 0.5)
    assert not is_improvement(0.5, 0.5)
    assert not is_improvement(0.5 - 0.5 * IMPROVEMENT_THRESHOLD / 2, 0.5)
    assert is_improvement(0.5 - 0.5 * IMPROVEMENT_THRESHOLD * 2, 0.5)


@needs_routines
def test_a_negative_incumbent_does_not_invert_the_test():
    """Scaled by |best| rather than by best: the plain `best * (1 - t)` form
    inverts for a negative loss and would accept a worse value."""
    assert not is_improvement(-0.5, -0.5)
    assert is_improvement(-0.6, -0.5)
    assert not is_improvement(-0.4, -0.5)


@needs_routines
def test_a_non_finite_loss_is_never_an_improvement():
    assert not is_improvement(np.nan, 0.5)
    assert not is_improvement(np.inf, 0.5)


def test_every_shipped_config_sets_a_half_life_and_no_stale_factor():
    """The same numeric value means a different schedule under the two forms,
    so a config carrying the old key is refused at the entry points rather than
    reinterpreted. None may carry one."""
    checked = 0
    for path in sorted(CONFIGS.glob('*.yaml')):
        config = yaml.safe_load(path.read_text())
        if 'PRETRAIN_LEARNING_RATE' not in config:
            continue
        checked += 1
        assert 'PRETRAIN_LEARNING_RATE_DECAY' not in config, path.name
        assert 'FINETUNE_LEARNING_RATE_DECAY' not in config, path.name
        assert 'PRETRAIN_LR_HALF_LIFE' in config, path.name
    assert checked > 0


def test_the_converted_half_lives_reproduce_the_old_schedules():
    """A factor g applied every I epochs is a half-life of I * ln(0.5)/ln(g).
    Pretraining stepped 0.9 every 50 epochs, finetuning 0.9 every 20."""
    for interval, expected in ((50, 329), (20, 132)):
        half_life = interval * np.log(0.5) / np.log(0.9)
        assert round(half_life) == expected
