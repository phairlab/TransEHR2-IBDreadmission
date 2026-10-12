"""What an epoch's losses summarize, and what an outlying step reports.

Two statistics were once printed side by side under one heading: the generator term was
the median of its steps, the total was the mean of generator plus discriminator. They
therefore disagreed by whatever the right tail held, and on this cohort the tail is the
whole story -- a training panel showed a generator of 8.66 beside a total of 144,800.
Numeric features are scored by squared error over values standardized by a robust scale
that does not clip, so one sentinel laboratory value standardizes into the thousands and
squares into the millions, and the median is precisely the summary that hides it.

So the mean is reported under the heading that says mean, and the tail is described where
it can be acted on: ``describe_loss_spike`` names the step and the largest prediction and
target magnitudes in it, which is what separates a standardization problem from a model
one.
"""

import numpy as np
import pytest
import torch

from TransEHR2.routines import _mean_losses
from TransEHR2.utils import describe_loss_spike


def test_the_generator_term_is_a_mean():
    """The regression this file exists for: no term is summarized by its median."""
    records = {'Generator_Loss': [1.0, 1.0, 1.0, 1.0, 1_000_000.0],
               'Discriminator_Loss': [0.2, 0.2, 0.2, 0.2, 0.2]}
    out = _mean_losses(records)
    assert out['Generator_Loss'] == float(np.mean(records['Generator_Loss']))
    assert out['Generator_Loss'] > 100_000.0


def test_the_total_and_the_generator_term_stay_commensurable():
    """``Optimization_Loss`` is the mean of the per-step sums, so the rows must add up."""
    gen = [1.0, 2.0, 500.0]
    disc = [0.1, 0.2, 0.3]
    records = {'Optimization_Loss': [g + d for g, d in zip(gen, disc)],
               'Generator_Loss': gen, 'Discriminator_Loss': disc}
    out = _mean_losses(records)
    assert out['Optimization_Loss'] == pytest.approx(out['Generator_Loss']
                                                     + out['Discriminator_Loss'])


def test_an_unrecorded_term_is_nan_not_zero():
    """A term that never ran has no mean; zero would read as a term that scored perfectly."""
    assert np.isnan(_mean_losses({'THP_Loss': []})['THP_Loss'])


def _one_feature(values, mask_values):
    """A single-feature batch of one episode and one timestep."""
    return [torch.tensor([[values]], dtype=torch.float32)], \
        [torch.tensor([[mask_values]], dtype=torch.float32)]


def test_a_sentinel_target_is_named_against_an_ordinary_prediction():
    """The asymmetry the report is for: a huge target beside a small prediction."""
    predictions, _ = _one_feature([0.5, 0.25], [1.0, 1.0])
    masks = {'numeric': {'values': [torch.tensor([[[1.0, 1.0]]])],
                         'indicators': torch.tensor([[[1.0]]])}}
    report = describe_loss_spike(
        predictions={'numeric': {'values': predictions}},
        masked_targets={'numeric': {'values': [torch.tensor([9_400.0, 3.0])]}},
        record_masks=masks, gen_loss=8.8e7, reference=6.2)
    assert 'numeric' in report
    assert '9,400.0' in report      # the target, which standardization let through
    assert '0.5' in report          # the prediction, which is ordinary


def test_a_whole_record_family_is_masked_by_its_indicator_column():
    """Categorical masks index the indicator column, as the generator loss does."""
    report = describe_loss_spike(
        predictions={'categorical': {'values': [torch.tensor([[[0.1, 0.9]]])]}},
        masked_targets={'categorical': {'values': [torch.tensor([[0.0, 1.0]])]}},
        record_masks={'categorical': {'indicators': torch.tensor([[[1.0]]])}},
        gen_loss=50.0, reference=5.0)
    assert 'categorical' in report


def test_a_family_the_batch_does_not_carry_is_skipped():
    """Absent families leave no row rather than raising."""
    report = describe_loss_spike(predictions={}, masked_targets={}, record_masks={},
                                 gen_loss=50.0, reference=5.0)
    assert 'numeric' not in report
    assert '10x' in report.replace(',', '')
