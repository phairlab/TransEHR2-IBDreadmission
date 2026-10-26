"""The Cox head: its loss, its fine-tuning, its dump, and encoder reuse."""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from TransEHR2.csc import read_csc_baseline, read_csc_lp
from TransEHR2.losses import CoxPartialLikelihoodLoss
from TransEHR2.models import MixedClassifier
from TransEHR2.modules import DeepHitHead
from TransEHR2.routines import (
    cox_scores, evaluate_finetuned_model, finetune_model
)
from TransEHR2.survival import (
    CENSORED, DEATH, MINUTES_PER_DAY, READMISSION, TimeGrid
)

from .test_routines_survival import (
    CUTS, D_EVENT, D_MODEL, DEVICE, _event_encoder, _value_encoder, loaders  # noqa: F401
)


def naive_cox_loss(eta, days, cause_index, is_event):
    """The joint negative log partial likelihood per event, by loops."""
    total, n = 0.0, 0
    for i in range(len(days)):
        if not is_event[i]:
            continue
        k = cause_index[i]
        at_risk = days >= days[i]
        total -= eta[i, k] - np.log(np.exp(eta[at_risk, k]).sum())
        n += 1
    return total / n if n else 0.0


def cox_predictor(grid):
    torch.manual_seed(0)
    return MixedClassifier(
        event_encoder=_event_encoder(), val_encoder=_value_encoder(),
        d_event_enc=D_EVENT, d_val_enc=D_MODEL, d_statics=0,
        num_classes=grid.n_causes, aggr='mean',
        head=DeepHitHead(d_in=D_EVENT + D_MODEL, n_causes=grid.n_causes,
                         n_bins=1, d_shared=[16, 16], d_cause=[8, 8],
                         dropout=0.0))


# -- the loss -----------------------------------------------------------------

def test_the_loss_matches_the_partial_likelihood_with_ties():
    rng = np.random.default_rng(0)
    n = 40
    eta = rng.normal(size=(n, 2))
    days = rng.choice([5.0, 10.0, 20.0, 40.0, 80.0], size=n)   # many ties
    cause = rng.integers(-1, 2, size=n)
    event = cause >= 0
    loss, per_cause = CoxPartialLikelihoodLoss(2)(
        torch.tensor(eta), torch.tensor(days), torch.tensor(cause),
        torch.tensor(event))
    assert float(loss) == pytest.approx(
        naive_cox_loss(eta, days, cause, event), rel=1e-12)
    assert float(per_cause.sum()) / event.sum() == pytest.approx(float(loss))


def test_a_cause_absent_from_the_batch_adds_nothing():
    eta = torch.randn(6, 3, dtype=torch.float64)
    days = torch.tensor([1., 2., 3., 4., 5., 6.], dtype=torch.float64)
    cause = torch.tensor([0, -1, 0, 1, -1, 0])          # no cause 2
    _, per_cause = CoxPartialLikelihoodLoss(3)(eta, days, cause, cause >= 0)
    assert float(per_cause[2]) == 0.0


def test_a_batch_without_events_is_zero_and_differentiable():
    eta = torch.randn(5, 2, requires_grad=True)
    cause = torch.full((5,), -1)
    loss, _ = CoxPartialLikelihoodLoss(2)(eta, torch.arange(5.), cause,
                                          cause >= 0)
    loss.backward()
    assert float(loss) == 0.0 and eta.grad is not None


def test_the_loss_ignores_row_order():
    rng = np.random.default_rng(1)
    eta = torch.tensor(rng.normal(size=(30, 2)))
    days = torch.tensor(rng.uniform(1, 100, size=30))
    cause = torch.tensor(rng.integers(-1, 2, size=30))
    perm = torch.randperm(30)
    fn = CoxPartialLikelihoodLoss(2)
    assert float(fn(eta, days, cause, cause >= 0)[0]) == pytest.approx(
        float(fn(eta[perm], days[perm], cause[perm], cause[perm] >= 0)[0]),
        rel=1e-12)


def test_minimizing_the_loss_recovers_the_coefficients():
    """On data from a proportional-hazards model, the loss's minimizer is
    the true log hazard ratio, up to sampling error."""
    rng = np.random.default_rng(2)
    n, beta = 4000, np.array([[0.8, -0.5]])            # one covariate
    x = rng.normal(size=(n, 1))
    rates = np.exp(x @ beta)                           # cause-specific rates
    t_k = rng.exponential(1.0 / rates)                 # latent times
    t = t_k.min(axis=1)
    cause = t_k.argmin(axis=1)
    censor = rng.exponential(2.0, size=n)
    event = t <= censor
    days = np.minimum(t, censor)
    cause = np.where(event, cause, -1)

    b = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([b], max_iter=100)
    fn = CoxPartialLikelihoodLoss(2)
    xt, dt = torch.tensor(x), torch.tensor(days)
    ct, et = torch.tensor(cause), torch.tensor(event)

    def closure():
        opt.zero_grad()
        loss, _ = fn(xt @ b, dt, ct, et)
        loss.backward()
        return loss
    opt.step(closure)
    np.testing.assert_allclose(b.detach().numpy(), beta, atol=0.08)


# -- fine-tuning ---------------------------------------------------------------

def test_a_cox_head_finetunes_and_selects_on_the_partial_likelihood(
        loaders, tmp_path):  # noqa: F811
    grid = TimeGrid(CUTS)
    model = cox_predictor(grid)
    head_before = {k: v.clone() for k, v in model.state_dict().items()
                   if k.startswith('head.')}
    train_scores, val_scores = finetune_model(
        model=model, save_path=str(tmp_path / 'cox.pt'), loaders=loaders,
        grid=grid, writer=None, learning_rate=1e-2, device=DEVICE,
        total_epoch=2, selection_metric='partial_likelihood',
        checkpoint_dir=None, resume_from_checkpoint=False, head='cox')
    assert (tmp_path / 'cox.pt').exists()
    for scores in (train_scores, val_scores):
        assert np.isfinite(scores['Loss_CoxPL'])
        assert 'Readmission_Loss_CoxPL' in scores
        assert 'Loss_DeepHit' not in scores
    after = model.state_dict()
    assert any(not torch.allclose(v, after[k]) for k, v in
               head_before.items()), 'the Cox head did not train'


@pytest.mark.parametrize('head, metric', [('cox', 'brier'),
                                          ('deephit', 'partial_likelihood')])
def test_a_metric_from_the_other_head_is_refused(loaders, tmp_path,  # noqa: F811
                                                 head, metric):
    grid = TimeGrid(CUTS)
    with pytest.raises(ValueError, match='does not apply'):
        finetune_model(
            model=cox_predictor(grid), save_path=str(tmp_path / 'x.pt'),
            loaders=loaders, grid=grid, writer=None, learning_rate=1e-3,
            device=DEVICE, total_epoch=1, selection_metric=metric,
            checkpoint_dir=None, resume_from_checkpoint=False, head=head)


def test_the_split_score_uses_full_risk_sets(loaders):  # noqa: F811
    """Scored over the split at once, not as a mean of batch losses."""
    grid = TimeGrid(CUTS)
    model = cox_predictor(grid).eval()
    val = loaders[1]
    scores = evaluate_finetuned_model(
        model, val, grid, CoxPartialLikelihoodLoss(grid.n_causes), DEVICE)
    model.eval()                     # evaluate hands it back in train mode
    with torch.no_grad():
        eta = torch.cat([model(b) for b in val])
    tte = torch.cat([b['targets']['time_to_event'] for b in val])
    etype = torch.cat([b['targets']['event_type'] for b in val])
    assert scores['Loss_CoxPL'] == pytest.approx(
        cox_scores(eta, tte, etype, grid)['Loss_CoxPL'], rel=1e-6)
    _, ci, ev = grid.discretize(tte, etype)
    days = np.minimum(tte.reshape(-1).numpy() / MINUTES_PER_DAY,
                      grid.horizon_days)
    assert scores['Loss_CoxPL'] == pytest.approx(naive_cox_loss(
        eta.double().numpy(), days, ci.numpy(), ev.numpy()), rel=1e-6)


# -- the dump ---------------------------------------------------------------------

def test_breslow_matches_a_direct_computation():
    from dump_transehr2_csc_predictions import breslow

    grid = TimeGrid(CUTS)
    rng = np.random.default_rng(3)
    n = 60
    lp = rng.normal(size=(n, grid.n_causes))
    days = rng.choice([3.0, 7.0, 15.0, 50.0, 120.0, 400.0], size=n)
    etype = rng.choice([CENSORED, READMISSION, DEATH], size=n)
    times, cumhaz = breslow(lp, days * MINUTES_PER_DAY, etype, grid)

    horizon = min(days.max(), grid.horizon_days)
    d = np.minimum(days, grid.horizon_days)
    code = {READMISSION: 0, DEATH: 1}
    ev = (etype != CENSORED) & (days < grid.horizon_days)
    expected_times = np.unique(d[ev])
    np.testing.assert_array_equal(times, expected_times)
    for k in range(grid.n_causes):
        steps = [sum(1.0 for i in range(n)
                     if ev[i] and code[etype[i]] == k and d[i] == t)
                 / np.exp(lp[d >= t, k]).sum() for t in expected_times]
        np.testing.assert_allclose(cumhaz[k], np.cumsum(steps), rtol=1e-12)
    assert horizon > 0


def test_with_zero_lp_breslow_is_nelson_aalen():
    from dump_transehr2_csc_predictions import breslow

    grid = TimeGrid(CUTS, ['readmission'])
    days = np.array([1., 2., 2., 3., 5., 8.])
    etype = np.array([READMISSION, READMISSION, CENSORED, READMISSION,
                      CENSORED, READMISSION])
    times, cumhaz = breslow(np.zeros((6, 1)), days * MINUTES_PER_DAY, etype,
                            grid)
    np.testing.assert_allclose(times, [1., 2., 3., 8.])
    np.testing.assert_allclose(cumhaz[0], np.cumsum([1 / 6, 1 / 5, 1 / 3, 1]))


def test_the_files_round_trip_through_the_csc_readers(tmp_path):
    from dump_transehr2_csc_predictions import (breslow, save_baseline,
                                                save_lp)

    grid = TimeGrid(CUTS)
    rng = np.random.default_rng(4)
    n = 50
    data = {'idx': rng.permutation(n),
            'time_to_event': rng.uniform(1, 500, n) * MINUTES_PER_DAY,
            'event_type': rng.choice([0, 1, 2, 3], n),
            'lp': rng.normal(size=(n, grid.n_causes))}
    save_lp(str(tmp_path / 'lp.csv'), data, grid)
    back = read_csc_lp(str(tmp_path / 'lp.csv'), grid)
    order = np.argsort(data['idx'])
    np.testing.assert_array_equal(back['idx'], data['idx'][order])
    np.testing.assert_array_equal(back['lp'], data['lp'][order])

    times, cumhaz = breslow(data['lp'], data['time_to_event'],
                            data['event_type'], grid)
    save_baseline(str(tmp_path / 'base.csv'), times, cumhaz, grid)
    t_back, c_back = read_csc_baseline(str(tmp_path / 'base.csv'), grid)
    np.testing.assert_array_equal(t_back, times)
    np.testing.assert_array_equal(c_back, cumhaz)


def test_the_dump_is_scored_as_a_csc_model(tmp_path, monkeypatch):
    """lp and Breslow files -> grid dumps -> recalibration -> evaluation,
    which must take the exact-curve path."""
    import evaluate_finetuned_predictions
    import recalibrate_predictions
    from dump_baseline_csc_predictions import write_fold_dumps
    from dump_transehr2_csc_predictions import (breslow, save_baseline,
                                                save_lp)
    from TransEHR2.csc import BASELINE_FILE, LP_FILE

    grid = TimeGrid(CUTS)
    folder = tmp_path / 'exp' / 'fold1' / 'competing_risks'
    folder.mkdir(parents=True)
    rng = np.random.default_rng(5)
    splits = {}
    for s, split in enumerate(('train', 'val', 'test')):
        n = 400
        splits[split] = {
            'idx': np.arange(s * 1000, s * 1000 + n),
            'time_to_event': rng.uniform(1, 500, n) * MINUTES_PER_DAY,
            'event_type': rng.choice([0, 1, 1, 2, 3], n),
            'lp': rng.normal(scale=0.3, size=(n, grid.n_causes))}
        save_lp(str(folder / LP_FILE.format(task='competing_risks',
                                            split=split)),
                splits[split], grid)
    train = splits['train']
    save_baseline(str(folder / BASELINE_FILE.format(task='competing_risks')),
                  *breslow(train['lp'], train['time_to_event'],
                           train['event_type'], grid), grid)
    write_fold_dumps(str(folder), 'fold1', grid)

    config = tmp_path / 'exp.yaml'
    config.write_text(yaml.dump({
        'TIME_GRID_CUTS_DAYS': CUTS, 'MODELLED_CAUSES': list(grid.cause_names),
        'BRIER_INTEGRATION_DAYS': 365, 'CR_DCAL_N_RHO': 100,
        'EVAL_HORIZONS_DAYS': [30, 90]}))
    for module in (recalibrate_predictions, evaluate_finetuned_predictions):
        monkeypatch.setattr(sys, 'argv', [
            'x', 'exp', '--model_dir', str(tmp_path), '--experiment_config',
            str(config)])
        module.main()
    for split in ('test', 'test_recal_aj'):
        with open(tmp_path / 'exp' / f'{split}_evaluation.yaml') as f_in:
            evaluation = yaml.safe_load(f_in)
        assert evaluation['csc_curve']
        for cause in ('Readmission', 'Death', 'Out_Migration'):
            assert np.isfinite(evaluation[f'{cause}_CR_DCal'][0])


# -- encoder reuse ---------------------------------------------------------------

def test_encoders_are_copied_once_and_never_over_a_copy(tmp_path):
    from fit_transehr2_deephit import adopt_pretrained_encoders

    source, target = tmp_path / 'src', tmp_path / 'dst'
    (source / 'evaluation').mkdir(parents=True)
    (source / 'value_encoder.pt').write_bytes(b'encoder')
    (source / 'evaluation' / 'evaluation_pretrained.yaml').write_text('a: 1')
    target.mkdir()
    adopt_pretrained_encoders(str(target), str(source))
    assert (target / 'value_encoder.pt').read_bytes() == b'encoder'
    assert (target / 'evaluation' / 'evaluation_pretrained.yaml').exists()
    assert yaml.safe_load((target / 'pretrained_from.yaml').read_text())[
        'source_dir'] == str(source)
    assert not (target / 'event_encoder.pt').exists()

    (source / 'value_encoder.pt').write_bytes(b'changed')
    adopt_pretrained_encoders(str(target), str(source))
    assert (target / 'value_encoder.pt').read_bytes() == b'encoder'


def test_a_missing_source_encoder_is_refused(tmp_path):
    from fit_transehr2_deephit import adopt_pretrained_encoders

    (tmp_path / 'dst').mkdir()
    with pytest.raises(FileNotFoundError, match='Pretrain'):
        adopt_pretrained_encoders(str(tmp_path / 'dst'),
                                  str(tmp_path / 'src'))


def test_a_source_that_pretrained_differently_is_refused(tmp_path):
    from fit_transehr2_deephit import check_pretraining_source

    base = {'EXPERIMENT_NAME': 'a', 'MODEL_DIR': '/m', 'BATCH_SIZE': 200,
            'CMPNT_MASK_RATIO': 0.5, 'DEEPHIT_RANK_WEIGHT': 1.0}
    (tmp_path / 'a.yaml').write_text(yaml.dump(base))
    ours = {**base, 'EXPERIMENT_NAME': 'b', 'PRETRAINED_FROM': 'a',
            'DEEPHIT_RANK_WEIGHT': 0.0, 'SURVIVAL_HEAD': 'cox'}
    check_pretraining_source(str(tmp_path / 'b.yaml'), ours, 'a')
    with pytest.raises(ValueError, match='CMPNT_MASK_RATIO'):
        check_pretraining_source(str(tmp_path / 'b.yaml'),
                                 {**ours, 'CMPNT_MASK_RATIO': 0.3}, 'a')
    with pytest.raises(ValueError, match='EXTRA'):
        check_pretraining_source(str(tmp_path / 'b.yaml'),
                                 {**ours, 'EXTRA': 1}, 'a')
    with pytest.raises(FileNotFoundError):
        check_pretraining_source(str(tmp_path / 'b.yaml'), ours, 'missing')


def test_the_ablation_configs_differ_only_where_intended():
    configs = os.path.join(os.path.dirname(__file__), os.pardir, 'configs',
                           'experiments')

    def load(name):
        with open(os.path.join(configs, f'{name}.yaml')) as f_in:
            return yaml.safe_load(f_in)

    def diff(a, b):
        return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}

    te2, tab = load('part1_transehr2'), load('part1_tabdh')
    assert diff(te2, load('part1_transehr2_norank')) == {
        'EXPERIMENT_NAME', 'PRETRAINED_FROM', 'DEEPHIT_RANK_WEIGHT'}
    assert diff(te2, load('part1_transehr2_csc')) == {
        'EXPERIMENT_NAME', 'PRETRAINED_FROM', 'SURVIVAL_HEAD',
        'FINETUNE_SELECTION_METRIC'}
    assert diff(tab, load('part1_tabdh_norank')) == {
        'EXPERIMENT_NAME', 'DEEPHIT_RANK_WEIGHT'}
    assert load('part1_transehr2_norank')['DEEPHIT_RANK_WEIGHT'] == 0
    assert load('part1_tabdh_norank')['DEEPHIT_RANK_WEIGHT'] == 0


def test_rank_weight_zero_is_the_likelihood_alone():
    """The no-rank ablation: the DeepHit loss is its NLL term, exactly."""
    from TransEHR2.losses import DeepHitLoss

    grid = TimeGrid(CUTS)
    logits = torch.randn(12, grid.n_causes * grid.n_bins)
    tte = torch.rand(12) * 400 * MINUTES_PER_DAY
    etype = torch.tensor([0, 1, 2, 3] * 3)
    bins, cause, event = grid.discretize(tte, etype)
    total, (nll, ranking) = DeepHitLoss(grid.n_causes, grid.n_bins,
                                        rank_weight=0.0)(logits, bins, cause,
                                                         event)
    assert torch.equal(total, nll) and float(ranking) == 0.0
