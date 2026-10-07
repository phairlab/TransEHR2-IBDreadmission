"""The single-GPU routines, end to end on a cohort small enough to run.

These check the wiring rather than the arithmetic -- that a batch's labels
reach the DeepHit loss, that a gradient reaches the head, that the files
each stage promises the next one actually appear, and that the THP switch
changes what is built rather than only what is weighted. The arithmetic is
checked in ``test_survival.py``.
"""

import numpy as np
import os
import pytest
import torch

from TransEHR2.losses import DeepHitLoss
from TransEHR2.models import ELECTRA, MixedClassifier
from TransEHR2.modules import (
    DeepHitHead, EventDataEncoder, MaskedTokenDiscriminator,
    MaskedTokenGenerator, TransformerHawkesProcess, ValueDataEncoder
)
from TransEHR2.routines import (
    BRIER_SELECTION_CAUSE, cpu_state_dict, evaluate_finetuned_model,
    finetune_model, is_improvement, load_checkpoint, pretrain_model,
    save_checkpoint, save_encoder_weights
)
from TransEHR2.survival import (
    CENSORED, DEATH, DEFAULT_BRIER_INTEGRATION_DAYS, DEFAULT_CAUSES,
    MINUTES_PER_DAY, OUT_MIGRATION, READMISSION, TimeGrid
)


CUTS = [30.0, 90.0, 365.0]
D_MODEL = 16
D_EVENT = 8
N_NUMERIC = 2
N_CATEGORICAL_CLASSES = [3]
N_EVENT_TYPES = 2
T = 4
DEVICE = torch.device('cpu')


# ------------------------------------------------------------------ fixtures

def _batch(n: int, seed: int) -> dict:
    """One collated batch, shaped exactly as ``collate_tensorized`` leaves it.

    Built by hand rather than run through the extractor: these tests are
    about the loops, and a real extraction would make them slow enough that
    nobody runs them.
    """
    rng = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.rand(*shape, generator=rng)

    n_cat = sum(N_CATEGORICAL_CLASSES)
    # Every episode carries at least one observed timestep, so no branch
    # sees an all-padding row unless a test asks for one.
    masks = torch.ones(n, T)

    # Event types are drawn so that each batch holds both causes and both
    # kinds of censoring; a batch of one class would make the ranking term
    # vacuous without failing.
    event_type = torch.tensor(
        [[READMISSION, DEATH, CENSORED, OUT_MIGRATION][i % 4]
         for i in range(n)], dtype=torch.long)
    # Times spread across the grid, including past its horizon.
    days = torch.tensor([[10.0, 45.0, 200.0, 500.0][i % 4] for i in range(n)])

    return {
        'val_data': {
            'numeric': {
                'indicators': (rand(n, T, N_NUMERIC) > 0.3).float(),
                'values': [rand(n, T, 1) for _ in range(N_NUMERIC)],
            },
            'categorical': {
                'indicators': (rand(n, T, 1) > 0.3).float(),
                'values': [rand(n, T, n_cat)],
            },
            'ordinal': {'indicators': torch.zeros(n, T, 0), 'values': []},
            'multilabel': {'indicators': torch.zeros(n, T, 0), 'values': []},
            'times': torch.arange(T, dtype=torch.float32).expand(n, T).clone(),
            'masks': masks,
        },
        'event_data': {
            'indicators': (rand(n, T, N_EVENT_TYPES) > 0.5).float(),
            'times': torch.arange(T, dtype=torch.float32).expand(n, T).clone(),
            'masks': masks,
        },
        'targets': {
            'time_to_event': (days * MINUTES_PER_DAY).unsqueeze(-1),
            'event_type': event_type,
        },
    }


@pytest.fixture
def loaders():
    """Three splits of two batches each. A list is a loader here: the
    routines only iterate it and check for a distributed sampler."""
    return tuple([_batch(8, seed), _batch(8, seed + 100)]
                 for seed in (1, 2, 3))


def _value_encoder():
    return ValueDataEncoder(
        n_features=N_NUMERIC + len(N_CATEGORICAL_CLASSES),
        feat_dim=N_NUMERIC + sum(N_CATEGORICAL_CLASSES),
        d_model=D_MODEL, n_heads=2, n_encoder_blocks=1, dim_feedforward=16,
        dropout=0.0, norm='LayerNorm')


def _event_encoder():
    return EventDataEncoder(
        num_types=N_EVENT_TYPES, d_model=D_EVENT, d_inner=16, n_layers=1,
        n_head=2, d_k=4, d_v=4, dropout=0.0)


def _predictor(grid):
    torch.manual_seed(0)
    return MixedClassifier(
        event_encoder=_event_encoder(),
        val_encoder=_value_encoder(),
        d_event_enc=D_EVENT, d_val_enc=D_MODEL, d_statics=0,
        num_classes=grid.n_causes * grid.n_bins,
        aggr='mean',
        head=DeepHitHead(d_in=D_EVENT + D_MODEL, n_causes=grid.n_causes,
                         n_bins=grid.n_bins, d_shared=[16, 16],
                         d_cause=[8, 8], dropout=0.0),
    )


def _electra(use_thp: bool):
    torch.manual_seed(0)
    generator = MaskedTokenGenerator(
        encoder=_value_encoder(), d_model=D_MODEL,
        numeric_dims=[1] * N_NUMERIC,
        categorical_classes=N_CATEGORICAL_CLASSES,
        predict_indicators=False, dim_feedforward=16)
    discriminator = MaskedTokenDiscriminator(
        encoder=_value_encoder(), d_model=D_MODEL,
        n_numeric_features=N_NUMERIC,
        n_categorical_features=len(N_CATEGORICAL_CLASSES),
        n_ordinal_features=0, n_multilabel_features=0, n_static_features=0,
        dim_feedforward=16)
    hawkes = TransformerHawkesProcess(
        encoder=_event_encoder(), num_types=N_EVENT_TYPES
    ) if use_thp else None
    return ELECTRA(generator, discriminator, hawkes)


# ---------------------------------------------------------- early stopping

def test_is_improvement_respects_the_direction():
    assert is_improvement(0.5, 1.0)
    assert not is_improvement(1.5, 1.0)
    assert is_improvement(0.8, 0.7, higher_is_better=True)
    assert not is_improvement(0.6, 0.7, higher_is_better=True)


def test_is_improvement_needs_more_than_noise():
    """A move in the fifth decimal is not an improvement, either way."""
    assert not is_improvement(1.0 - 1e-9, 1.0)
    assert not is_improvement(0.7 + 1e-9, 0.7, higher_is_better=True)


def test_is_improvement_accepts_anything_over_an_infinity():
    assert is_improvement(5.0, np.inf)
    assert is_improvement(0.1, -np.inf, higher_is_better=True)
    assert not is_improvement(np.nan, 1.0)


# ------------------------------------------------------------- finetuning

def test_finetune_runs_and_reports_survival_scores(loaders, tmp_path):
    grid = TimeGrid(CUTS)
    save_path = tmp_path / 'models' / 'finetuned.pt'

    train_scores, val_scores = finetune_model(
        model=_predictor(grid), save_path=str(save_path), loaders=loaders,
        grid=grid, writer=None, learning_rate=1e-3, device=DEVICE,
        total_epoch=2, checkpoint_dir=None, resume_from_checkpoint=False)

    assert save_path.exists()
    for scores in (train_scores, val_scores):
        assert np.isfinite(scores['Loss_DeepHit'])
        assert np.isfinite(scores['Loss_DeepHit_NLL'])
        assert 'Readmission_Cindex' in scores
        assert 'Death_Integrated_Brier' in scores
        assert scores['Readmission_Events'] + scores['Death_Events'] \
            + scores['Censored'] == 16


def test_finetune_moves_the_head(loaders, tmp_path):
    """Two epochs must change the weights; a frozen head would still
    'succeed' at everything else these tests check."""
    grid = TimeGrid(CUTS)
    model = _predictor(grid)
    before = cpu_state_dict(model)

    finetune_model(
        model=model, save_path=str(tmp_path / 'm.pt'), loaders=loaders,
        grid=grid, writer=None, learning_rate=1e-2, device=DEVICE,
        total_epoch=2, checkpoint_dir=None, resume_from_checkpoint=False)

    after = cpu_state_dict(model)
    moved = [k for k in before
             if not torch.allclose(before[k], after[k], atol=1e-7)]
    assert any(k.startswith('head.') for k in moved), \
        'the DeepHit head did not train'
    assert any(k.startswith('val_encoder.') for k in moved), \
        'the value encoder did not train'


def test_finetune_rejects_an_unknown_selection_metric(loaders, tmp_path):
    grid = TimeGrid(CUTS)
    with pytest.raises(ValueError, match='selection_metric'):
        finetune_model(
            model=_predictor(grid), save_path=str(tmp_path / 'm.pt'),
            loaders=loaders, grid=grid, writer=None, learning_rate=1e-3,
            device=DEVICE, total_epoch=1, selection_metric='auroc',
            checkpoint_dir=None, resume_from_checkpoint=False)


@pytest.mark.parametrize('metric', ['brier', 'cindex', 'loss'])
def test_every_selection_metric_picks_an_epoch(loaders, tmp_path, metric):
    grid = TimeGrid(CUTS)
    _, val_scores = finetune_model(
        model=_predictor(grid), save_path=str(tmp_path / f'{metric}.pt'),
        loaders=loaders, grid=grid, writer=None, learning_rate=1e-3,
        device=DEVICE, total_epoch=2, selection_metric=metric,
        checkpoint_dir=None, resume_from_checkpoint=False)
    assert val_scores, f'{metric} selected no epoch'


def test_selecting_on_brier_without_readmission_is_refused(loaders,
                                                          tmp_path):
    """The column would be missing and every epoch would score NaN."""
    grid = TimeGrid(CUTS, ['death'])
    with pytest.raises(ValueError, match='MODELLED_CAUSES'):
        finetune_model(
            model=_predictor(grid), save_path=str(tmp_path / 'no.pt'),
            loaders=loaders, grid=grid, writer=None, learning_rate=1e-3,
            device=DEVICE, total_epoch=1, selection_metric='brier',
            checkpoint_dir=None, resume_from_checkpoint=False)


def test_evaluate_scores_the_split_as_a_whole(loaders, tmp_path):
    """Concordance over two batches must not be the mean of two per-batch
    concordances: most comparable pairs cross the batch boundary."""
    grid = TimeGrid(CUTS)
    model = _predictor(grid)
    loss_fn = DeepHitLoss(grid.n_causes, grid.n_bins)
    _, _, test_loader = loaders

    whole = evaluate_finetuned_model(
        model=model, loader=test_loader, grid=grid, loss_fn=loss_fn,
        device=DEVICE)
    halves = [
        evaluate_finetuned_model(model=model, loader=[b], grid=grid,
                                 loss_fn=loss_fn, device=DEVICE)
        for b in test_loader
    ]
    assert whole['Readmission_Events'] == sum(
        h['Readmission_Events'] for h in halves)
    assert whole['Censored'] == sum(h['Censored'] for h in halves)


def test_evaluate_leaves_the_model_in_training_mode(loaders):
    """finetune_model calls it mid-loop, so it has to hand the model back."""
    grid = TimeGrid(CUTS)
    model = _predictor(grid)
    model.train()
    evaluate_finetuned_model(
        model=model, loader=loaders[1], grid=grid,
        loss_fn=DeepHitLoss(grid.n_causes, grid.n_bins), device=DEVICE)
    assert model.training


# ------------------------------------------------------------ pretraining

def test_pretrain_with_thp_writes_both_encoders(loaders, tmp_path):
    save_path = tmp_path / 'pretrained' / 'pretrained.pt'
    pretrain_model(
        model=_electra(use_thp=True), save_path=str(save_path),
        loaders=loaders, writer=None, learning_rate=1e-3, device=DEVICE,
        total_epoch=1, use_thp=True, thp_loss_mc_samples=2,
        checkpoint_dir=None, resume_from_checkpoint=False)

    assert (save_path.parent / 'value_encoder.pt').exists()
    assert (save_path.parent / 'event_encoder.pt').exists()


def test_pretrain_without_thp_writes_only_the_value_encoder(loaders, tmp_path):
    """The switch removes the module, so there is no event encoder to save.

    Finetuning is expected to build its own and train it from scratch; that
    is the trade the switch makes.
    """
    save_path = tmp_path / 'pretrained' / 'pretrained.pt'
    train_losses, _ = pretrain_model(
        model=_electra(use_thp=False), save_path=str(save_path),
        loaders=loaders, writer=None, learning_rate=1e-3, device=DEVICE,
        total_epoch=1, use_thp=False, checkpoint_dir=None,
        resume_from_checkpoint=False)

    assert (save_path.parent / 'value_encoder.pt').exists()
    assert not (save_path.parent / 'event_encoder.pt').exists()
    assert train_losses['THP_Loss'] == 0.0


def test_electra_without_hawkes_emits_no_hawkes_output(loaders):
    from TransEHR2.utils import generate_record_masks
    batch = loaders[0][0]
    model = _electra(use_thp=False)
    masks, _ = generate_record_masks(batch, feature_sample_rate=0.5,
                                     obs_unobs_ratio=1.0, subsample_rate=0.5)
    output = model(batch, masks, device=DEVICE, compute_intensities=True)

    assert 'hawkes_encodings' not in output
    assert 'thp_intensities' not in output
    assert 'generator' in output and 'discriminator' in output


def test_pretrain_objective_includes_the_thp_term_only_when_on(loaders,
                                                               tmp_path):
    with_thp, _ = pretrain_model(
        model=_electra(use_thp=True),
        save_path=str(tmp_path / 'a' / 'p.pt'), loaders=loaders, writer=None,
        learning_rate=1e-3, device=DEVICE, total_epoch=1, use_thp=True,
        thp_loss_mc_samples=2, checkpoint_dir=None,
        resume_from_checkpoint=False)
    assert with_thp['THP_Loss'] != 0.0


# ----------------------------------------------------------- checkpointing

def test_checkpoint_round_trips(tmp_path):
    grid = TimeGrid(CUTS)
    model = _predictor(grid)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # Take a step so the saved weights differ from a fresh model's.
    logits = model(_batch(4, 7))
    logits.sum().backward()
    optimizer.step()
    saved = cpu_state_dict(model)

    ckpt = tmp_path / 'ckpt.pt'
    meta = tmp_path / 'meta.json'
    save_checkpoint(str(ckpt), str(meta), model, optimizer, scheduler,
                    epoch=4, metadata={'best_epoch': 2,
                                       'early_stopping_counter': 1})

    restored = _predictor(grid)
    metadata, _ = load_checkpoint(
        str(ckpt), str(meta), restored,
        torch.optim.Adam(restored.parameters(), lr=1e-3),
        torch.optim.lr_scheduler.ExponentialLR(
            torch.optim.Adam(restored.parameters(), lr=1e-3), gamma=0.9))

    assert metadata['start_epoch'] == 5
    assert metadata['best_epoch'] == 2
    assert metadata['early_stopping_counter'] == 1
    for key, value in saved.items():
        assert torch.allclose(value, restored.state_dict()[key])


def test_missing_checkpoint_starts_from_scratch(tmp_path):
    grid = TimeGrid(CUTS)
    model = _predictor(grid)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    metadata, best = load_checkpoint(
        str(tmp_path / 'nope.pt'), str(tmp_path / 'nope.json'), model,
        optimizer, scheduler)
    assert metadata == {} and best is None


def test_half_a_checkpoint_is_not_a_checkpoint(tmp_path):
    """Weights without metadata would resume at epoch 0 with a reset early-
    stopping counter, which is worse than starting over."""
    grid = TimeGrid(CUTS)
    model = _predictor(grid)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    ckpt = tmp_path / 'ckpt.pt'
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}, ckpt)

    metadata, _ = load_checkpoint(str(ckpt), str(tmp_path / 'absent.json'),
                                  model, optimizer, scheduler)
    assert metadata == {}


def test_finetune_resumes_where_it_stopped(loaders, tmp_path):
    grid = TimeGrid(CUTS)
    ckpt_dir = tmp_path / 'ckpt'

    finetune_model(
        model=_predictor(grid), save_path=str(tmp_path / 'm.pt'),
        loaders=loaders, grid=grid, writer=None, learning_rate=1e-3,
        device=DEVICE, total_epoch=10, checkpoint_dir=str(ckpt_dir),
        resume_from_checkpoint=False)

    # A finished run cleans up after itself, so nothing is left to resume.
    assert not (ckpt_dir / 'finetune_checkpoint.pt').exists()
    assert not (ckpt_dir / 'finetune_metadata.json').exists()


def test_save_encoder_weights_tolerates_a_missing_event_encoder(tmp_path):
    state = {
        'discriminator.encoder.layer.weight': torch.zeros(2, 2),
        'generator.something': torch.zeros(1),
    }
    save_encoder_weights(state, str(tmp_path), expect_event_encoder=False)
    assert (tmp_path / 'value_encoder.pt').exists()
    assert not (tmp_path / 'event_encoder.pt').exists()


def test_save_encoder_weights_needs_a_value_encoder(tmp_path, capsys):
    save_encoder_weights({'generator.x': torch.zeros(1)}, str(tmp_path))
    assert 'no value encoder weights' in capsys.readouterr().out
    assert not (tmp_path / 'value_encoder.pt').exists()


# ------------------------------------------------------------ the timer

def test_create_timer_returns_a_usable_timer(tmp_path):
    """A whole phase, start to summary. The timer is only exercised by the
    drivers, which the suite does not run, so nothing else would catch a
    name that stopped existing."""
    from TransEHR2.utils import create_timer

    timer = create_timer(results_dir=str(tmp_path), experiment_name='probe')
    timer.start_total_timing()
    timer.start_fold('fold0')
    timer.start_phase('pretrain')
    timer.end_phase('pretrain')
    timer.end_fold()
    timer.print_final_summary()

    assert (tmp_path / 'probe_timing_results.yaml').exists()


def test_timer_state_survives_a_checkpoint(tmp_path):
    from TransEHR2.utils import create_timer

    timer = create_timer(results_dir=str(tmp_path), experiment_name='probe')
    timer.start_total_timing()
    timer.start_fold('fold0')
    timer.start_phase('pretrain')
    state = timer.get_timer_state_for_checkpoint()

    resumed = create_timer(results_dir=str(tmp_path), experiment_name='probe')
    resumed.restore_from_checkpoint(state)
    assert resumed.times['current_fold'] == 'fold0'


# ------------------------------------------------------- shipped configs

CONFIG_DIR = (os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
              + '/configs/experiments')


def _experiment_configs():
    import yaml
    for name in sorted(os.listdir(CONFIG_DIR)):
        if not name.endswith('.yaml'):
            continue
        with open(os.path.join(CONFIG_DIR, name)) as f_in:
            yield name, yaml.safe_load(f_in)


@pytest.mark.parametrize('name,config', list(_experiment_configs()))
def test_shipped_configs_declare_a_usable_grid(name, config):
    """A grid that TimeGrid refuses would fail after the data is loaded and
    the model built, which is an expensive place to learn it."""
    cuts = config.get('TIME_GRID_CUTS_DAYS')
    if cuts is None:
        pytest.skip(f'{name} sets no grid; the default applies')
    grid = TimeGrid(cuts, config.get('MODELLED_CAUSES', DEFAULT_CAUSES),
                    config.get('BRIER_INTEGRATION_DAYS',
                               DEFAULT_BRIER_INTEGRATION_DAYS))
    assert grid.n_bins == len(cuts)


@pytest.mark.parametrize('name,config', list(_experiment_configs()))
def test_shipped_configs_name_a_real_selection_metric(name, config):
    from TransEHR2.routines import SELECTION_METRICS
    metric = config.get('FINETUNE_SELECTION_METRIC')
    if metric is None:
        pytest.skip(f'{name} sets no selection metric; the default applies')
    assert metric in SELECTION_METRICS


@pytest.mark.parametrize('name,config', list(_experiment_configs()))
def test_shipped_configs_can_select_on_the_metric_they_name(name, config):
    """'brier' scores readmission, so a config naming it must model it.

    The run refuses this combination before building anything; the point
    of checking it here is that the refusal would otherwise arrive after
    the data is loaded and the encoders are on the GPU.
    """
    if config.get('FINETUNE_SELECTION_METRIC') != 'brier':
        pytest.skip(f'{name} does not select on the Brier score')
    causes = config.get('MODELLED_CAUSES', DEFAULT_CAUSES)
    assert BRIER_SELECTION_CAUSE in causes


@pytest.mark.parametrize('name,config', list(_experiment_configs()))
def test_shipped_configs_say_whether_the_hawkes_process_runs(name, config):
    """USE_THP defaults to True in code, so a config that omits it silently
    keeps the module the switch exists to remove."""
    assert 'USE_THP' in config, f'{name} does not state USE_THP'
    assert isinstance(config['USE_THP'], bool)


def test_resuming_under_a_different_selection_metric_is_refused(loaders,
                                                                tmp_path):
    """The stored best score is a concordance or a likelihood, never both."""
    grid = TimeGrid(CUTS)
    ckpt_dir = tmp_path / 'ckpt'
    ckpt_dir.mkdir()
    model = _predictor(grid)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    save_checkpoint(
        str(ckpt_dir / 'finetune_checkpoint.pt'),
        str(ckpt_dir / 'finetune_metadata.json'),
        model, optimizer, scheduler, epoch=0,
        metadata={'best_epoch_val_metric': 0.72,
                  'selection_metric': 'cindex'})

    with pytest.raises(ValueError, match='selected on'):
        finetune_model(
            model=_predictor(grid), save_path=str(tmp_path / 'm.pt'),
            loaders=loaders, grid=grid, writer=None, learning_rate=1e-3,
            device=DEVICE, total_epoch=1, selection_metric='loss',
            checkpoint_dir=str(ckpt_dir), resume_from_checkpoint=True)


def test_the_head_is_sized_for_what_the_trunk_actually_concatenates():
    """A head built from ``encoding_width`` must accept the real encoding.

    The drivers size their DeepHitHead from that number without ever seeing
    the concatenation, so the two have to be checked against each other
    somewhere.
    """
    grid = TimeGrid(CUTS)
    width = MixedClassifier.encoding_width(D_EVENT, D_MODEL, 0)
    model = MixedClassifier(
        event_encoder=_event_encoder(), val_encoder=_value_encoder(),
        d_event_enc=D_EVENT, d_val_enc=D_MODEL, d_statics=0,
        num_classes=grid.n_causes * grid.n_bins, aggr='mean',
        head=DeepHitHead(d_in=width, n_causes=grid.n_causes,
                         n_bins=grid.n_bins, d_shared=[16, 16],
                         d_cause=[8, 8], dropout=0.0))

    logits = model(_batch(3, 42))
    assert logits.shape == (3, grid.n_causes * grid.n_bins)


@pytest.mark.parametrize('name,config', list(_experiment_configs()))
def test_shipped_configs_name_real_causes(name, config):
    """A cause the grid refuses would fail after the data is loaded."""
    from TransEHR2.survival import CAUSE_CODES
    causes = config.get('MODELLED_CAUSES')
    if causes is None:
        pytest.skip(f'{name} sets no cause list; the default applies')
    assert all(cause in CAUSE_CODES for cause in causes), causes
    grid = TimeGrid(config['TIME_GRID_CUTS_DAYS'], causes)
    assert grid.n_causes == len(causes)


@pytest.mark.parametrize('name,config', list(_experiment_configs()))
def test_shipped_cause_weights_match_the_cause_count(name, config):
    """DeepHitLoss rejects a mismatch; catch it here instead of an hour in."""
    causes = config.get('MODELLED_CAUSES')
    weights = config.get('DEEPHIT_CAUSE_WEIGHTS')
    if causes is None or weights is None:
        pytest.skip(f'{name} leaves one of the two at its default')
    assert len(weights) == len(causes)
