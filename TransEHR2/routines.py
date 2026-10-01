"""Training and evaluation on one GPU.

Why one GPU
-----------

The MIMIC fork ran these routines under Accelerate with FSDP or DDP, which
bought data parallelism at the cost of a collective operation in every hot
path: gathers around every metric, broadcasts around every filesystem check,
a barrier around every checkpoint, and a state dict that had to be
reassembled from shards before it could be read. None of that earns its
keep here. This cohort's arrays fit on one card, and the work that actually
wants parallelism -- five cross-validation folds, or a sweep over one
hyperparameter -- is embarrassingly parallel at the level of the *run*. Five
independent single-GPU processes, one per fold, use the same hardware with
none of the synchronization, and a failure takes down one fold instead of
the job.

So there is no rank here, no gather, no barrier, and a checkpoint is a
dictionary in a file.

What is predicted
-----------------

One task: the joint distribution over (cause, time bin), fitted with
DeepHit. Unplanned readmission and death compete, and a model that scored
them separately would be answering a question the cohort cannot pose --
a patient who dies cannot be readmitted. The MIMIC fork's three tasks
(in-hospital mortality, length of stay, phenotyping) are gone: the first
does not exist in this data, which follows patients after discharge rather
than within a stay, and the other two were never IBD questions.

``TransEHR2.survival`` holds the grid, the likelihood's geometry and the
metrics; this module holds the loops.
"""

import json
import numpy as np
import os
import shutil
import torch
import torch.utils.tensorboard  # noqa: F401  (the annotations below name it)

from torch import Tensor
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, TypeAlias

from TransEHR2.losses import (
    DeepHitLoss, MaskedDiscriminatorLoss, MaskedGeneratorLoss,
    TransformerHawkesLoss
)
from TransEHR2.models import MixedClassifier
from TransEHR2.survival import TimeGrid, survival_metrics
from TransEHR2.utils import Timer
from TransEHR2.utils import format_pretraining_performance_table
from TransEHR2.utils import generate_record_masks, move_batch_to_device
from TransEHR2.utils import print_peak_memory


MetadataDict: TypeAlias = Dict[str, Any]
StateDict: TypeAlias = Dict[str, Tensor]

# Relative improvement threshold for early stopping, matching ReduceLROnPlateau's default.
# Without one, "no improvement" means an exactly equal or worse loss, so movement in the fifth
# decimal place holds a run open and the epoch of the best model becomes a function of numeric
# noise rather than of convergence.
IMPROVEMENT_THRESHOLD = 1e-4

# Epochs without improvement before a run stops.
EARLY_STOPPING_PATIENCE = 30

# Which validation number selects the finetuned model. They disagree often
# enough to matter: the ranking term rewards ordering the cohort, the
# likelihood rewards calibrating it, and a run can improve at one while
# losing ground on the other.
#
# 'brier' is the one that answers the question this study asks. The other
# two score all causes at once -- a run that orders deaths well and
# readmissions poorly looks good on 'cindex' -- and neither says anything
# about a horizon. The integrated Brier score is readmission's alone, and
# it is integrated from discharge out to TimeGrid.brier_integration_days,
# one year by default. Lower is better.
SELECTION_METRICS = {
    'brier': ('Readmission_Integrated_Brier', False),
    'cindex': ('Mean_Cindex', True),
    'loss': ('Loss_DeepHit', False),
}

# The cause 'brier' scores. A run that does not model it has no such
# column to select on.
BRIER_SELECTION_CAUSE = 'readmission'


def resolve_decay_factor(lr_half_life: Optional[float]) -> float:
    """Per-epoch multiplicative decay factor for a half-life given in epochs.

    The scheduler steps once per epoch, so the factor and the schedule are the same statement:
    lr(e) = lr0 * 0.5 ** (e / H). Expressing it as a half-life keeps the number in epochs,
    which is the unit that can be held against how long a run actually lasts; a bare
    multiplicative factor only means something once the step cadence is also known.

    Args:
        lr_half_life: Epochs over which the learning rate halves. None or non-positive
            leaves the rate constant.

    Returns:
        float: The gamma to pass to ExponentialLR.
    """
    if lr_half_life is None or lr_half_life <= 0:
        return 1.0
    return 0.5 ** (1.0 / lr_half_life)


def is_improvement(
    current: float,
    best: float,
    threshold: float = IMPROVEMENT_THRESHOLD,
    higher_is_better: bool = False
) -> bool:
    """Whether a validation score improves on the incumbent by more than a relative threshold.

    Scaled by the magnitude of the incumbent so the test means the same thing for scores of
    any size. The scaling uses the absolute value rather than the plain ``best * (1 -
    threshold)`` form, which inverts for a negative value and would accept a worse one.

    Args:
        current: The current epoch's validation score.
        best: The best validation score so far, or an infinity before any epoch has run.
        threshold: Minimum relative improvement required.
        higher_is_better: True for a score such as concordance, False for a loss.

    Returns:
        bool: True if current improves on best by more than the threshold.
    """
    if not np.isfinite(current):
        return False
    if not np.isfinite(best):
        return True
    margin = abs(best) * threshold
    return current > best + margin if higher_is_better else current < best - margin


# ------------------------------------------------------------ checkpointing

def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy/torch scalars to JSON-writable Python ones."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    return obj


def save_checkpoint(
    checkpoint_path: str,
    metadata_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metadata: MetadataDict,
    best_state_dict: Optional[StateDict] = None,
    best_state_path: Optional[str] = None,
    timer: Optional[Timer] = None
) -> None:
    """Write the resumable state and the run's metadata.

    Two files rather than one: the tensors are large and rewritten every
    checkpoint, the metadata is small and wanted by anything inspecting a
    run in progress. The best state dict, when given, is a third -- it is
    the answer the run exists to produce, and it should not be reachable
    only through a file whose purpose is to be deleted on success.
    """
    torch.save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        },
        checkpoint_path,
    )

    payload = dict(_json_safe(metadata))
    payload['start_epoch'] = epoch + 1
    if timer is not None:
        payload.update(_json_safe(timer.get_timer_state_for_checkpoint()))
    with open(metadata_path, 'w') as f_out:
        json.dump(payload, f_out, indent=2)

    if best_state_dict is not None and best_state_path is not None:
        torch.save(best_state_dict, best_state_path)

    print(f"\nSaved checkpoint at epoch {epoch + 1}")


def load_checkpoint(
    checkpoint_path: Optional[str],
    metadata_path: Optional[str],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    best_state_path: Optional[str] = None,
    timer: Optional[Timer] = None
) -> Tuple[MetadataDict, Optional[StateDict]]:
    """Restore a run in progress, or report that there is none to restore.

    Both files have to be there. A checkpoint without its metadata would
    resume at epoch zero with a trained model and a reset early-stopping
    counter, which is worse than starting over because it looks like it
    worked.
    """
    have_both = (checkpoint_path is not None and metadata_path is not None
                 and os.path.exists(checkpoint_path)
                 and os.path.exists(metadata_path))
    if not have_both:
        print("\nOne of checkpoint or training metadata not found. "
              "Training from scratch.")
        return {}, None

    with open(metadata_path, 'r') as f_in:
        metadata = json.load(f_in)

    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])

    if timer is not None:
        timer.restore_from_checkpoint(metadata)

    best_state_dict = None
    if best_state_path is not None and os.path.exists(best_state_path):
        best_state_dict = torch.load(best_state_path, map_location='cpu',
                                     weights_only=False)
        print(f"Loaded best state dict from epoch "
              f"{metadata.get('best_epoch', -1)}")

    print(f"\nResuming from checkpoint at epoch "
          f"{metadata.get('start_epoch', 0)}")
    return metadata, best_state_dict


def _clear_checkpoint(*paths: Optional[str]) -> None:
    for path in paths:
        if path is None or not os.path.exists(path):
            continue
        shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)


def cpu_state_dict(model: torch.nn.Module) -> StateDict:
    """A detached CPU copy of the model's parameters.

    Held for the length of a run, so it must not keep GPU memory alive and
    must not alias tensors the optimizer is about to overwrite.
    """
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_encoder_weights(
    best_state_dict: Optional[StateDict],
    save_dir: str,
    expect_event_encoder: bool = True
) -> None:
    """Split the pretrained encoders out of the ELECTRA state dict.

    Finetuning rebuilds the two encoders and loads these files into them,
    so they are the whole product of pretraining as far as the downstream
    model is concerned.

    ``expect_event_encoder`` is False when the Hawkes process was switched
    off: there is then no ``hawkes.encoder.*`` in the state dict, because
    the module was never built, and finetuning trains its event encoder
    from scratch instead.

    Args:
        best_state_dict: Full CPU state dict of the best pretraining epoch.
        save_dir: Directory to write ``value_encoder.pt`` and, when there
            is one, ``event_encoder.pt``.
        expect_event_encoder: Whether a missing event encoder is a problem.
    """
    if best_state_dict is None:
        print("Warning: no best state dict to extract encoder weights from")
        return

    value_encoder_state = {}
    event_encoder_state = {}
    for key, value in best_state_dict.items():
        if key.startswith('discriminator.encoder.'):
            value_encoder_state[key[len('discriminator.encoder.'):]] = \
                value.cpu().clone()
        elif key.startswith('hawkes.encoder.'):
            event_encoder_state[key[len('hawkes.encoder.'):]] = \
                value.cpu().clone()

    if not value_encoder_state:
        print("Warning: no value encoder weights found in the state dict")
        return

    os.makedirs(save_dir, exist_ok=True)
    torch.save(value_encoder_state, os.path.join(save_dir, 'value_encoder.pt'))
    if event_encoder_state:
        torch.save(event_encoder_state,
                   os.path.join(save_dir, 'event_encoder.pt'))
    elif expect_event_encoder:
        print("Warning: no event encoder weights found in the state dict")

    print(f"\nSaved encoder weights to {save_dir}")


# --------------------------------------------------------------- pretraining

_EMPTY_PRETRAIN_LOSSES = {
    'Optimization_Loss': np.inf,
    'Generator_Loss': np.inf,
    'Discriminator_Loss': np.inf,
    'THP_Loss': np.inf,
    'THP_NLL_Loss': np.inf,
    'THP_Type_Loss': np.inf,
    'THP_Time_Loss': np.inf,
}


def _mean_losses(records: Dict[str, List[float]]) -> Dict[str, float]:
    """Epoch means, with the generator term trimmed.

    The generator term is the only unbounded one in the objective -- numeric features are
    scored by squared error, everything else by cross-entropy, BCE or a cosine distance in
    [0, 1]. A single step with a large residual therefore moves the epoch mean a long way,
    and the median is the summary that survives it.
    """
    out = {}
    for name, values in records.items():
        if not values:
            out[name] = float('nan')
        elif name == 'Generator_Loss':
            out[name] = float(np.median(values))
        else:
            out[name] = float(np.mean(values))
    return out


def _run_pretrain_batches(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    gen_loss_fn: MaskedGeneratorLoss,
    disc_loss_fn: MaskedDiscriminatorLoss,
    thp_loss_fn: Optional[TransformerHawkesLoss],
    thp_loss_mc_samples: int,
    record_mask_ratio: float,
    obs_unobs_sample_ratio: float,
    cmpnt_mask_ratio: float,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    desc: str,
    mem_test_mode: bool
) -> Dict[str, float]:
    """One pass over ``loader``; optimizes when given an optimizer.

    Training and validation differ only in whether a step is taken and
    whether gradients are tracked, so they share a body. The masking is
    resampled either way: a fixed validation mask would measure the model
    against one draw of a stochastic objective.
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    records: Dict[str, List[float]] = {k: [] for k in _EMPTY_PRETRAIN_LOSSES}

    with torch.set_grad_enabled(training):
        for i, batch in tqdm(enumerate(loader), desc=desc, leave=False):
            batch = move_batch_to_device(batch, device=device)
            value_masks, _ = generate_record_masks(
                batch,
                feature_sample_rate=record_mask_ratio,
                obs_unobs_ratio=obs_unobs_sample_ratio,
                subsample_rate=cmpnt_mask_ratio
            )

            output = model(
                batch,
                value_masks,
                device=device,
                trace_grads=False,
                compute_intensities=thp_loss_fn is not None,
                thp_loss_mc_samples=thp_loss_mc_samples
            )

            thp_type_preds, thp_time_preds = output.get(
                'hawkes_predictions', (None, None))

            gen_loss = gen_loss_fn(output['generator'],
                                   output['masked_targets'], value_masks)
            disc_loss = disc_loss_fn(output['discriminator'], value_masks)
            loss = gen_loss + disc_loss

            if thp_loss_fn is not None and 'thp_intensities' in output:
                intensities = output['thp_intensities']
                thp_loss, (nll, type_loss, time_loss) = thp_loss_fn(
                    intensities['obs_initial'],
                    intensities['obs_conditional'],
                    intensities['sampled'],
                    batch['event_data'],
                    thp_type_preds,
                    thp_time_preds
                )
                loss = loss + thp_loss
                records['THP_Loss'].append(thp_loss.item())
                records['THP_NLL_Loss'].append(nll.item())
                records['THP_Type_Loss'].append(type_loss.item())
                records['THP_Time_Loss'].append(time_loss.item())
            else:
                for key in ('THP_Loss', 'THP_NLL_Loss', 'THP_Type_Loss',
                            'THP_Time_Loss'):
                    records[key].append(0.0)

            records['Optimization_Loss'].append(loss.item())
            records['Generator_Loss'].append(gen_loss.item())
            records['Discriminator_Loss'].append(disc_loss.item())

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=1.0)
                optimizer.step()

            if mem_test_mode and i == 1:
                print(f"Memory usage during {desc.lower()}:", flush=True)
                print_peak_memory(device)
                break

            del output, batch

    return _mean_losses(records)


def pretrain_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    gen_loss_fn: MaskedGeneratorLoss,
    disc_loss_fn: MaskedDiscriminatorLoss,
    thp_loss_fn: Optional[TransformerHawkesLoss],
    thp_loss_mc_samples: int,
    record_mask_ratio: float,
    obs_unobs_sample_ratio: float,
    cmpnt_mask_ratio: float,
    device: torch.device,
    desc: str = "Training",
    mem_test_mode: bool = False,
) -> Dict[str, float]:
    """Execute one epoch of pretraining. Returns the epoch's mean losses."""
    return _run_pretrain_batches(
        model, loader, gen_loss_fn, disc_loss_fn, thp_loss_fn,
        thp_loss_mc_samples, record_mask_ratio, obs_unobs_sample_ratio,
        cmpnt_mask_ratio, device, optimizer, desc, mem_test_mode)


def pretrain_validate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    gen_loss_fn: MaskedGeneratorLoss,
    disc_loss_fn: MaskedDiscriminatorLoss,
    thp_loss_fn: Optional[TransformerHawkesLoss],
    thp_loss_mc_samples: int,
    record_mask_ratio: float,
    obs_unobs_sample_ratio: float,
    cmpnt_mask_ratio: float,
    device: torch.device,
    desc: str = "Validation",
    mem_test_mode: bool = False,
) -> Dict[str, float]:
    """Evaluate the pretraining objective. Returns the epoch's mean losses."""
    return _run_pretrain_batches(
        model, loader, gen_loss_fn, disc_loss_fn, thp_loss_fn,
        thp_loss_mc_samples, record_mask_ratio, obs_unobs_sample_ratio,
        cmpnt_mask_ratio, device, None, desc, mem_test_mode)


def _build_pretrain_losses(
    device: torch.device,
    disc_loss_weight: float,
    use_thp: bool,
    thp_loss_nll_weight: float,
    use_thp_pred_loss: bool,
    thp_pred_loss_type_wt: float,
    thp_pred_loss_time_wt: float,
    ordinal_features: Optional[List[int]]
) -> Tuple[MaskedGeneratorLoss, MaskedDiscriminatorLoss,
           Optional[TransformerHawkesLoss]]:
    """The three pretraining objectives, on the training device.

    Moving them matters for exactly one and is free for the other two.
    MaskedGeneratorLoss holds a BetaLoss per ordinal feature, and BetaLoss
    registers its class-probability table as a buffer. Left on the CPU,
    that buffer is indexed by target classes that live on the GPU, and
    every ordinal feature raises "indices should be either on cpu or on the
    same device as the indexed tensor".
    """
    gen_loss_fn = MaskedGeneratorLoss(
        ordinal_features=ordinal_features).to(device)
    disc_loss_fn = MaskedDiscriminatorLoss(weight=disc_loss_weight).to(device)
    thp_loss_fn = None
    if use_thp:
        thp_loss_fn = TransformerHawkesLoss(
            add_prediction_loss=use_thp_pred_loss,
            nll_weight=thp_loss_nll_weight,
            type_weight=thp_pred_loss_type_wt,
            time_weight=thp_pred_loss_time_wt
        ).to(device)
    return gen_loss_fn, disc_loss_fn, thp_loss_fn


def pretrain_model(
    model: torch.nn.Module,
    save_path: str,
    loaders: Tuple[torch.utils.data.DataLoader, ...],
    writer: Optional[torch.utils.tensorboard.SummaryWriter],
    learning_rate: float,
    device: torch.device,
    lr_half_life: Optional[float] = None,
    total_epoch: int = 100,
    disc_loss_weight: float = 0.5,
    use_thp: bool = True,
    thp_loss_nll_weight: float = 1e-3,
    thp_loss_mc_samples: int = 100,
    use_thp_pred_loss: bool = True,
    thp_pred_loss_type_wt: float = 1.0,
    thp_pred_loss_time_wt: float = 0.01,
    record_mask_ratio: float = 0.25,
    obs_unobs_sample_ratio: float = 4.0,
    cmpnt_mask_ratio: float = 0.5,
    checkpoint_dir: Optional[str] = None,
    resume_from_checkpoint: bool = True,
    timer: Optional[Timer] = None,
    mem_test_mode: bool = False,
    ordinal_features: Optional[List[int]] = None
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Pre-train the ELECTRA-style model and write its encoders out.

    Args:
        model: The ELECTRA model to pre-train.
        save_path: Path to save the best model state dict. Its directory
            also receives the two encoder files finetuning reads.
        loaders: (train, val) or (train, val, test) DataLoaders.
        writer: TensorBoard writer, or None.
        learning_rate: Learning rate for the optimizer.
        device: Where the model and every batch live.
        lr_half_life: Epochs over which the learning rate halves. The scheduler steps every
            epoch, so this is the schedule in full. None leaves the rate constant.
        total_epoch: Maximum number of epochs.
        disc_loss_weight: Weight for the discriminator loss.
        use_thp: Whether the Transformer Hawkes process contributes at all.
            False drops its term from the objective and leaves the event
            encoder untrained, which is the point of the switch: the THP
            log-likelihood integrates an intensity over inter-event gaps,
            so a broad dynamic range of timestamps makes it explode, and
            no weighting fixes that because it is the form of the loss.
            The event stream still reaches the classifier downstream; its
            encoder simply learns during finetuning instead.
        thp_loss_nll_weight: Weight for the THP negative log-likelihood.
        thp_loss_mc_samples: Monte Carlo samples for the THP integral.
        use_thp_pred_loss: Whether to include the THP prediction losses.
        thp_pred_loss_type_wt: Weight for the THP type prediction loss.
        thp_pred_loss_time_wt: Weight for the THP time prediction loss.
        record_mask_ratio: Fraction of timesteps to mask.
        obs_unobs_sample_ratio: Ratio of observed to unobserved records.
        cmpnt_mask_ratio: Fraction of components to mask in vector features.
        checkpoint_dir: Directory for resumable checkpoints, or None.
        resume_from_checkpoint: Whether to resume if a checkpoint is there.
        timer: Timer for tracking training time.
        mem_test_mode: If True, runs two batches per phase and reports peak
            memory. Useful for finding the batch size a card will take.
        ordinal_features: Indices of ordinal features, for the generator loss.

    Returns:
        Tuple of (best_train_losses, best_val_losses) dictionaries.
    """
    report_freq = 10
    chkpt_freq = 10

    train_loader, val_loader = loaders[0], loaders[1]

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=resolve_decay_factor(lr_half_life))

    gen_loss_fn, disc_loss_fn, thp_loss_fn = _build_pretrain_losses(
        device, disc_loss_weight, use_thp, thp_loss_nll_weight,
        use_thp_pred_loss, thp_pred_loss_type_wt, thp_pred_loss_time_wt,
        ordinal_features)

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, 'pretrain_checkpoint.pt')
        metadata_path = os.path.join(checkpoint_dir, 'training_metadata.json')
        best_state_path = os.path.join(checkpoint_dir, 'best_model_state.pt')
    else:
        checkpoint_path = metadata_path = best_state_path = None

    metadata: MetadataDict = {}
    best_state_dict = None
    if resume_from_checkpoint:
        metadata, best_state_dict = load_checkpoint(
            checkpoint_path, metadata_path, model, optimizer, scheduler,
            best_state_path, timer)

    start_epoch = metadata.get('start_epoch', 0)
    best_epoch = metadata.get('best_epoch', -1)
    best_train_losses = metadata.get('best_epoch_train_losses',
                                     dict(_EMPTY_PRETRAIN_LOSSES))
    best_val_losses = metadata.get('best_epoch_val_losses',
                                   dict(_EMPTY_PRETRAIN_LOSSES))
    best_val_loss = best_val_losses['Optimization_Loss']
    early_stopping_counter = metadata.get('early_stopping_counter', 0)

    for epoch in tqdm(range(start_epoch, total_epoch)):
        train_losses = pretrain_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            gen_loss_fn=gen_loss_fn, disc_loss_fn=disc_loss_fn,
            thp_loss_fn=thp_loss_fn,
            thp_loss_mc_samples=thp_loss_mc_samples,
            record_mask_ratio=record_mask_ratio,
            obs_unobs_sample_ratio=obs_unobs_sample_ratio,
            cmpnt_mask_ratio=cmpnt_mask_ratio, device=device,
            desc=f"Epoch {epoch + 1} Training", mem_test_mode=mem_test_mode)

        val_losses = pretrain_validate(
            model=model, loader=val_loader,
            gen_loss_fn=gen_loss_fn, disc_loss_fn=disc_loss_fn,
            thp_loss_fn=thp_loss_fn,
            thp_loss_mc_samples=thp_loss_mc_samples,
            record_mask_ratio=record_mask_ratio,
            obs_unobs_sample_ratio=obs_unobs_sample_ratio,
            cmpnt_mask_ratio=cmpnt_mask_ratio, device=device,
            desc=f"Epoch {epoch + 1} Validation", mem_test_mode=mem_test_mode)

        if writer is not None:
            for name, value in train_losses.items():
                writer.add_scalar(f'{name}/train', value, epoch)
            for name, value in val_losses.items():
                writer.add_scalar(f'{name}/val', value, epoch)

        if is_improvement(val_losses['Optimization_Loss'], best_val_loss):
            best_epoch = epoch
            best_val_loss = val_losses['Optimization_Loss']
            best_train_losses = train_losses
            best_val_losses = val_losses
            best_state_dict = cpu_state_dict(model)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if checkpoint_dir is not None and (epoch + 1) % chkpt_freq == 0:
            save_checkpoint(
                checkpoint_path=checkpoint_path, metadata_path=metadata_path,
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch,
                metadata={
                    'best_epoch': best_epoch,
                    'early_stopping_counter': early_stopping_counter,
                    'best_epoch_train_losses': best_train_losses,
                    'best_epoch_val_losses': best_val_losses,
                },
                best_state_dict=best_state_dict,
                best_state_path=best_state_path, timer=timer)

        if epoch == 0 or (epoch + 1) % report_freq == 0:
            print("\n" + format_pretraining_performance_table(
                epoch=epoch + 1,
                current_train_losses=train_losses,
                current_val_losses=val_losses,
                best_train_losses=best_train_losses,
                best_val_losses=best_val_losses,
                use_thp_pred_loss=use_thp and use_thp_pred_loss
            ) + "\n")

        if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nNo improvement observed within {early_stopping_counter} "
                  f"epochs. Stopping early.\n")
            break

        scheduler.step()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_state_dict, save_path)
    print(f"Saved pretrained model weights to {save_path}")

    save_encoder_weights(best_state_dict, os.path.dirname(save_path),
                         expect_event_encoder=use_thp)
    _clear_checkpoint(checkpoint_path, metadata_path, best_state_path)
    print_peak_memory(device)

    return best_train_losses, best_val_losses


# ---------------------------------------------------------------- finetuning

def _survival_targets(
    batch: Dict[str, Any], grid: TimeGrid
) -> Tuple[Tensor, Tensor, Tensor]:
    """The three label tensors DeepHit needs, from a collated batch."""
    targets = batch['targets']
    return grid.discretize(targets['time_to_event'], targets['event_type'])


def _forward_survival(
    model: MixedClassifier,
    batch: Dict[str, Any],
    grid: TimeGrid,
    loss_fn: DeepHitLoss
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """One forward pass: ``(loss, logits, time_to_event, event_type, nll)``.

    The raw labels come back alongside the discretized ones because the
    metrics re-derive the grid themselves -- they are reported at several
    horizons, and a caller that had only the bin index could not ask for a
    different one.
    """
    logits = model(batch)
    bin_index, cause_index, is_event = _survival_targets(batch, grid)
    loss, (nll, _) = loss_fn(logits, bin_index, cause_index, is_event)
    targets = batch['targets']
    return (loss, logits.detach(),
            targets['time_to_event'].detach().reshape(-1),
            targets['event_type'].detach().reshape(-1), nll.detach())


def _score(
    logits: List[Tensor],
    times: List[Tensor],
    types: List[Tensor],
    losses: List[float],
    grid: TimeGrid,
    prefix: str,
    global_step: int,
    writer: Optional[torch.utils.tensorboard.SummaryWriter]
) -> Dict[str, float]:
    """Turn accumulated batch output into the reported scores.

    The metrics are computed over the whole split at once rather than
    averaged over batches. Concordance is a property of the pair set, and
    the pairs that cross a batch boundary are most of them.
    """
    if not logits:
        return {}

    scores = survival_metrics(
        torch.cat(logits, dim=0).float().cpu().numpy(),
        torch.cat(times, dim=0).cpu().numpy(),
        torch.cat(types, dim=0).cpu().numpy(),
        grid,
    )
    scores['Loss_DeepHit'] = float(np.mean(losses)) if losses else float('nan')

    if writer is not None:
        for name, value in scores.items():
            if np.isfinite(value):
                writer.add_scalars(name, {prefix: value},
                                   global_step=global_step)
    return scores


def evaluate_finetuned_model(
    model: MixedClassifier,
    loader: torch.utils.data.DataLoader,
    grid: TimeGrid,
    loss_fn: DeepHitLoss,
    device: torch.device,
    prefix: str = '',
    global_step: int = 0,
    writer: Optional[torch.utils.tensorboard.SummaryWriter] = None,
    mem_test_mode: bool = False
) -> Dict[str, float]:
    """Score a finetuned competing-risks model on one split.

    Args:
        model: The finetuned model.
        loader: DataLoader for the split.
        grid: The discrete time grid the head predicts on.
        loss_fn: The DeepHit objective, for the reported loss.
        device: Where the model and batches live.
        prefix: TensorBoard series name, e.g. 'val'.
        global_step: Epoch number, for TensorBoard.
        writer: TensorBoard writer, or None.
        mem_test_mode: If True, stops after two batches.

    Returns:
        The dict ``survival.survival_metrics`` returns, plus
        ``Loss_DeepHit``.
    """
    model.eval()
    logits, times, types, losses = [], [], [], []

    with torch.no_grad():
        desc = 'Evaluating competing-risks model performance'
        for i, batch in tqdm(enumerate(loader), desc=desc, leave=False):
            batch = move_batch_to_device(batch, device=device)
            loss, batch_logits, tte, etype, _ = _forward_survival(
                model, batch, grid, loss_fn)

            losses.append(loss.item())
            logits.append(batch_logits)
            times.append(tte)
            types.append(etype)

            if mem_test_mode and i == 1:
                print("Memory usage during finetuned model evaluation:",
                      flush=True)
                print_peak_memory(device)
                break

            del batch

    scores = _score(logits, times, types, losses, grid, prefix, global_step,
                    writer)
    model.train()
    return scores


def finetune_model(
    model: MixedClassifier,
    save_path: str,
    loaders: Tuple[torch.utils.data.DataLoader, ...],
    grid: TimeGrid,
    writer: Optional[torch.utils.tensorboard.SummaryWriter],
    learning_rate: float,
    device: torch.device,
    rank_weight: float = 1.0,
    sigma: float = 0.1,
    cause_weights: Optional[List[float]] = None,
    selection_metric: str = 'brier',
    lr_half_life: Optional[float] = None,
    total_epoch: int = 100,
    checkpoint_dir: Optional[str] = None,
    resume_from_checkpoint: bool = True,
    timer: Optional[Timer] = None,
    mem_test_mode: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Fine-tune the pretrained encoders on the competing-risks outcome.

    Args:
        model: A MixedClassifier carrying a DeepHitHead, with the
            pretrained encoder weights already loaded.
        save_path: Where to write the best epoch's state dict.
        loaders: (train, val) or (train, val, test) DataLoaders.
        grid: The discrete time grid the head predicts on.
        writer: TensorBoard writer, or None.
        learning_rate: Learning rate for the optimizer.
        device: Where the model and every batch live.
        rank_weight: Weight on DeepHit's ranking term.
        sigma: Scale of the ranking term's exponential.
        cause_weights: Per-cause weight inside the ranking term.
        selection_metric: 'brier', 'cindex' or 'loss'; which validation
            number picks the best epoch. See SELECTION_METRICS.
        lr_half_life: Epochs over which the learning rate halves. The scheduler steps every
            epoch, so this is the schedule in full. None leaves the rate constant.
        total_epoch: Maximum number of epochs.
        checkpoint_dir: Directory for resumable checkpoints, or None.
        resume_from_checkpoint: Whether to resume if a checkpoint is there.
        timer: Timer for tracking training time.
        mem_test_mode: If True, runs two batches per phase and reports peak
            memory.

    Returns:
        Tuple of (best_train_scores, best_val_scores) dictionaries.
    """
    chkpt_freq = 10

    if selection_metric not in SELECTION_METRICS:
        raise ValueError(f'selection_metric: expected one of '
                         f'{sorted(SELECTION_METRICS)}, got {selection_metric}')
    metric_key, higher_is_better = SELECTION_METRICS[selection_metric]

    # Caught here rather than at the first epoch, where a missing key would
    # read as a NaN score, never improve on the incumbent, and save the
    # randomly initialized weights after the patience ran out.
    if (selection_metric == 'brier'
            and BRIER_SELECTION_CAUSE not in grid.cause_names):
        raise ValueError(
            f"selection_metric 'brier' scores {BRIER_SELECTION_CAUSE}, which "
            f"MODELLED_CAUSES does not include: {list(grid.cause_names)}. "
            f"Add it, or select on 'cindex' or 'loss'.")

    train_loader, val_loader = loaders[0], loaders[1]

    model = model.to(device)
    loss_fn = DeepHitLoss(
        n_causes=grid.n_causes, n_bins=grid.n_bins, sigma=sigma,
        rank_weight=rank_weight, cause_weights=cause_weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=resolve_decay_factor(lr_half_life))

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, 'finetune_checkpoint.pt')
        metadata_path = os.path.join(checkpoint_dir,
                                     'finetune_metadata.json')
        best_state_path = os.path.join(checkpoint_dir,
                                       'best_finetuned_state.pt')
    else:
        checkpoint_path = metadata_path = best_state_path = None

    metadata: MetadataDict = {}
    best_state_dict = None
    if resume_from_checkpoint:
        metadata, best_state_dict = load_checkpoint(
            checkpoint_path, metadata_path, model, optimizer, scheduler,
            best_state_path, timer)

    # A checkpoint carries the incumbent best *score*, which only means
    # anything under the metric that produced it. Resuming with a different
    # one would compare a concordance against a likelihood and keep whichever
    # happened to be larger.
    resumed_metric = metadata.get('selection_metric')
    if resumed_metric is not None and resumed_metric != selection_metric:
        raise ValueError(
            f"The checkpoint in {checkpoint_dir} selected on "
            f"'{resumed_metric}', but this run selects on "
            f"'{selection_metric}'. Delete the checkpoint to start over, or "
            f"set FINETUNE_SELECTION_METRIC back to '{resumed_metric}'."
        )

    start_epoch = metadata.get('start_epoch', 0)
    best_epoch = metadata.get('best_epoch', -1)
    best_train_scores = metadata.get('best_epoch_train_scores', {})
    best_val_scores = metadata.get('best_epoch_val_scores', {})
    best_metric = metadata.get('best_epoch_val_metric',
                               -np.inf if higher_is_better else np.inf)
    early_stopping_counter = metadata.get('early_stopping_counter', 0)

    for epoch in range(start_epoch, total_epoch):
        model.train()
        logits, times, types, losses = [], [], [], []

        desc = f"Epoch {epoch + 1}, Training"
        for i, batch in tqdm(enumerate(train_loader), desc=desc, leave=False):
            batch = move_batch_to_device(batch, device=device)
            loss, batch_logits, tte, etype, _ = _forward_survival(
                model, batch, grid, loss_fn)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(loss.item())
            logits.append(batch_logits)
            times.append(tte)
            types.append(etype)

            if mem_test_mode:
                print("Memory usage during finetuning training:", flush=True)
                print_peak_memory(device)
                if i == 1:
                    break

        train_scores = _score(logits, times, types, losses, grid, 'train',
                              epoch + 1, writer)
        del logits, times, types, losses

        val_scores = evaluate_finetuned_model(
            model=model, loader=val_loader, grid=grid, loss_fn=loss_fn,
            device=device, prefix='val', global_step=epoch + 1, writer=writer,
            mem_test_mode=mem_test_mode)

        current = val_scores.get(metric_key, np.nan)
        if is_improvement(current, best_metric,
                          higher_is_better=higher_is_better):
            best_epoch = epoch
            best_metric = current
            best_train_scores = train_scores
            best_val_scores = val_scores
            best_state_dict = cpu_state_dict(model)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if checkpoint_dir is not None and (epoch + 1) % chkpt_freq == 0:
            save_checkpoint(
                checkpoint_path=checkpoint_path, metadata_path=metadata_path,
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch,
                metadata={
                    'best_epoch': best_epoch,
                    'early_stopping_counter': early_stopping_counter,
                    'best_epoch_val_metric': best_metric,
                    'best_epoch_train_scores': best_train_scores,
                    'best_epoch_val_scores': best_val_scores,
                    'selection_metric': selection_metric,
                },
                best_state_dict=best_state_dict,
                best_state_path=best_state_path, timer=timer)

        if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nNo improvement observed within {early_stopping_counter} "
                  f"epochs. Stopping early.\n")
            break

        scheduler.step()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_state_dict, save_path)
    print(f"Saved finetuned model weights from epoch {best_epoch + 1} to "
          f"{save_path}\n")

    _clear_checkpoint(checkpoint_path, metadata_path, best_state_path)
    print_peak_memory(device)

    return best_train_scores, best_val_scores


# ------------------------------------------------------- hyperparameter runs

def pretrain_with_hyperparameter(
    hp_name: str,
    hp_value: Any,
    model: torch.nn.Module,
    loaders: Tuple[torch.utils.data.DataLoader, ...],
    writer: Optional[torch.utils.tensorboard.SummaryWriter],
    learning_rate: float,
    device: torch.device,
    lr_half_life: Optional[float] = None,
    total_epoch: int = 100,
    disc_loss_weight: float = 0.5,
    use_thp: bool = True,
    thp_loss_nll_weight: float = 1e-3,
    thp_loss_mc_samples: int = 100,
    use_thp_pred_loss: bool = True,
    thp_pred_loss_type_wt: float = 1.0,
    thp_pred_loss_time_wt: float = 0.01,
    record_mask_ratio: float = 0.25,
    obs_unobs_sample_ratio: float = 4.0,
    cmpnt_mask_ratio: float = 0.5,
    checkpoint_dir: Optional[str] = None,
    resume_from_checkpoint: bool = True,
    timer: Optional[Timer] = None,
    ordinal_features: Optional[List[int]] = None
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Pretrain under one hyperparameter setting, selecting on the test split.

    The difference from :func:`pretrain_model` is which split judges the
    run and what is kept afterwards. A sweep is choosing between settings,
    not producing the encoders a downstream model will load, so nothing is
    written but the scores; the validation split is left untouched so that
    the run chosen here still has an unused split to select its own epoch
    on later.

    Args:
        hp_name: Name of the hyperparameter being tuned, for file naming.
        hp_value: The value under test, for file naming.
        model: The ELECTRA model to pre-train.
        loaders: (train, test) or (train, val, test) DataLoaders.
        writer: TensorBoard writer, or None.
        learning_rate: Learning rate for the optimizer.
        device: Where the model and every batch live.
        lr_half_life: Epochs over which the learning rate halves.
        total_epoch: Maximum number of epochs.
        disc_loss_weight: Weight for the discriminator loss.
        use_thp: Whether the Transformer Hawkes process contributes at all.
        thp_loss_nll_weight: Weight for the THP negative log-likelihood.
        thp_loss_mc_samples: Monte Carlo samples for the THP integral.
        use_thp_pred_loss: Whether to include the THP prediction losses.
        thp_pred_loss_type_wt: Weight for the THP type prediction loss.
        thp_pred_loss_time_wt: Weight for the THP time prediction loss.
        record_mask_ratio: Fraction of timesteps to mask.
        obs_unobs_sample_ratio: Ratio of observed to unobserved records.
        cmpnt_mask_ratio: Fraction of components to mask in vector features.
        checkpoint_dir: Directory for resumable checkpoints, or None.
        resume_from_checkpoint: Whether to resume if a checkpoint is there.
        timer: Timer for tracking training time.
        ordinal_features: Indices of ordinal features, for the generator loss.

    Returns:
        Tuple of (best_train_losses, best_test_losses) dictionaries.
    """
    report_freq = 1
    chkpt_freq = 10

    train_loader = loaders[0]
    test_loader = loaders[-1]

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=resolve_decay_factor(lr_half_life))

    gen_loss_fn, disc_loss_fn, thp_loss_fn = _build_pretrain_losses(
        device, disc_loss_weight, use_thp, thp_loss_nll_weight,
        use_thp_pred_loss, thp_pred_loss_type_wt, thp_pred_loss_time_wt,
        ordinal_features)

    tag = f'{hp_name}_{hp_value}'
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir,
                                       f'pretrain_checkpoint_{tag}.pt')
        metadata_path = os.path.join(checkpoint_dir,
                                     f'training_metadata_{tag}.json')
    else:
        checkpoint_path = metadata_path = None

    metadata: MetadataDict = {}
    if resume_from_checkpoint:
        metadata, _ = load_checkpoint(checkpoint_path, metadata_path, model,
                                      optimizer, scheduler, None, timer)

    start_epoch = metadata.get('start_epoch', 0)
    best_epoch = metadata.get('best_epoch', -1)
    best_train_losses = metadata.get('best_epoch_train_losses',
                                     dict(_EMPTY_PRETRAIN_LOSSES))
    best_test_losses = metadata.get('best_epoch_val_losses',
                                    dict(_EMPTY_PRETRAIN_LOSSES))
    best_test_loss = best_test_losses['Optimization_Loss']
    early_stopping_counter = metadata.get('early_stopping_counter', 0)

    for epoch in tqdm(range(start_epoch, total_epoch)):
        train_losses = pretrain_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            gen_loss_fn=gen_loss_fn, disc_loss_fn=disc_loss_fn,
            thp_loss_fn=thp_loss_fn,
            thp_loss_mc_samples=thp_loss_mc_samples,
            record_mask_ratio=record_mask_ratio,
            obs_unobs_sample_ratio=obs_unobs_sample_ratio,
            cmpnt_mask_ratio=cmpnt_mask_ratio, device=device,
            desc=f"{hp_name}={hp_value} epoch {epoch + 1} Training")

        test_losses = pretrain_validate(
            model=model, loader=test_loader,
            gen_loss_fn=gen_loss_fn, disc_loss_fn=disc_loss_fn,
            thp_loss_fn=thp_loss_fn,
            thp_loss_mc_samples=thp_loss_mc_samples,
            record_mask_ratio=record_mask_ratio,
            obs_unobs_sample_ratio=obs_unobs_sample_ratio,
            cmpnt_mask_ratio=cmpnt_mask_ratio, device=device,
            desc=f"{hp_name}={hp_value} epoch {epoch + 1} Test")

        if writer is not None:
            for name, value in train_losses.items():
                writer.add_scalars(name, {f'train_{tag}': value}, epoch)
            for name, value in test_losses.items():
                writer.add_scalars(name, {f'test_{tag}': value}, epoch)

        if is_improvement(test_losses['Optimization_Loss'], best_test_loss):
            best_epoch = epoch
            best_test_loss = test_losses['Optimization_Loss']
            best_train_losses = train_losses
            best_test_losses = test_losses
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if checkpoint_dir is not None and (epoch + 1) % chkpt_freq == 0:
            save_checkpoint(
                checkpoint_path=checkpoint_path, metadata_path=metadata_path,
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch,
                metadata={
                    'best_epoch': best_epoch,
                    'early_stopping_counter': early_stopping_counter,
                    'best_epoch_train_losses': best_train_losses,
                    'best_epoch_val_losses': best_test_losses,
                    hp_name: hp_value,
                },
                timer=timer)

        if epoch == 0 or (epoch + 1) % report_freq == 0:
            print("\n" + format_pretraining_performance_table(
                epoch=epoch + 1,
                current_train_losses=train_losses,
                current_val_losses=test_losses,
                best_train_losses=best_train_losses,
                best_val_losses=best_test_losses,
                use_thp_pred_loss=use_thp and use_thp_pred_loss
            ) + "\n")

        if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nNo improvement observed within {early_stopping_counter} "
                  f"epochs. Stopping early.\n")
            break

        scheduler.step()

    _clear_checkpoint(checkpoint_path, metadata_path)
    print_peak_memory(device)

    return best_train_losses, best_test_losses
