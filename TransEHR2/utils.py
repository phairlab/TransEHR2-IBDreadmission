import numpy as np
import os
import time
import torch
import torch.distributed as dist
import yaml

from accelerate import Accelerator
from datetime import timedelta
from torch import Tensor
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, Union

from TransEHR2.data.datasets import MixedDataset
from TransEHR2.data.custom_types import MixedTensorDataset


class DistributedTimer:
    """Simplified timer for tracking pretraining times with checkpoint coordination."""
    
    def __init__(self, results_path: str = None):
        self.results_path = results_path
        self.world_size = 1
        self.rank = 0
        
        # Initialize distributed info if available
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            self.is_main_process = (self.rank == 0)
        
        self.times = {
            'total_start_time': None,
            'pretrain_total_time': 0.0,
            'finetune_total_time': 0.0,
            'most_recent_pretrain_time': 0.0,
            'most_recent_finetune_time': 0.0,
            'fold_times': {},
            'current_fold': None,
            'current_fold_start_time': None,
            'current_phase_start_time': None,
            'current_phase_elapsed': 0.0,
            'world_size': self.world_size
        }
    
    def start_total_timing(self):
        """Start timing the entire experiment."""
        if self.times['total_start_time'] is None:
            self.times['total_start_time'] = time.time()
    
    def start_fold(self, fold_name: str):
        """Start timing a specific fold."""
        self.times['current_fold'] = fold_name
        if fold_name not in self.times['fold_times']:
            self.times['fold_times'][fold_name] = {
                'pretrain_time': 0.0,
                'finetune_time': 0.0,
                'total_time': 0.0,
                'start_time': time.time()
            }
        self.times['current_fold_start_time'] = time.time()
    
    def start_phase(self, phase: str, is_main_process: bool):
        """Start timing a phase (pretrain/finetune)."""
        if self.world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.barrier()
        
        # If resuming from checkpoint, account for already elapsed time
        if self.times['current_phase_elapsed'] > 0:
            self.times['current_phase_start_time'] = time.time() - self.times['current_phase_elapsed']
            if is_main_process:
                print(f"Resuming {phase} phase with {self.times['current_phase_elapsed']:.1f}s already elapsed")
        else:
            self.times['current_phase_start_time'] = time.time()
    
    def end_phase(self, phase: str, is_main_process: bool):
        """End timing a phase and update totals."""
        if self.times['current_phase_start_time'] is None:
            return 0.0
        
        if self.world_size > 1 and dist.is_available() and dist.is_initialized():
            dist.barrier()
        
        elapsed = time.time() - self.times['current_phase_start_time']
        
        if phase == 'pretrain':
            self.times['pretrain_total_time'] += elapsed
            self.times['most_recent_pretrain_time'] = elapsed  # Overwrite with most recent
            if self.times['current_fold']:
                self.times['fold_times'][self.times['current_fold']]['pretrain_time'] += elapsed
        elif phase == 'finetune':
            self.times['finetune_total_time'] += elapsed
            self.times['most_recent_finetune_time'] = elapsed  # Overwrite with most recent
            if self.times['current_fold']:
                self.times['fold_times'][self.times['current_fold']]['finetune_time'] += elapsed
        
        # Reset phase tracking
        self.times['current_phase_start_time'] = None
        self.times['current_phase_elapsed'] = 0.0
        
        # Save results immediately
        if is_main_process:
            self.save_results(is_main_process)
        
        return elapsed
    
    def end_fold(self, is_main_process: bool):
        """End timing current fold."""
        if self.times['current_fold'] and self.times['current_fold_start_time']:
            fold_name = self.times['current_fold']
            total_fold_time = time.time() - self.times['current_fold_start_time']
            self.times['fold_times'][fold_name]['total_time'] = total_fold_time
            
            if is_main_process:
                print(f"\n{'='*60}")
                print(f"FOLD {fold_name.upper()} COMPLETED")
                print(f"{'='*60}")
                print(f"Most recent pretraining time: {self._format_time(self.times['most_recent_pretrain_time'])}")
                print(f"Most recent finetuning time: {self._format_time(self.times['most_recent_finetune_time'])}")
                print(f"Total fold time: {self._format_time(total_fold_time)}")
                if self.world_size > 1:
                    print(f"World size: {self.world_size} GPUs")
                print(f"{'='*60}\n")
        
        self.times['current_fold'] = None
        self.times['current_fold_start_time'] = None
        
        if is_main_process:
            self.save_results(is_main_process)
    
    def get_timer_state_for_checkpoint(self) -> dict:
        """Get timer state to include in model checkpoints."""
        if self.times['current_phase_start_time'] is not None:
            current_elapsed = time.time() - self.times['current_phase_start_time']
        else:
            current_elapsed = 0.0
            
        return {
            'timer_state': self.times.copy(),
            'current_phase_elapsed': current_elapsed
        }
    
    def restore_from_checkpoint(self, checkpoint_data: dict, is_main_process: bool):
        """Restore timer state from model checkpoint."""
        if 'timer_state' in checkpoint_data:
            timer_state = checkpoint_data['timer_state']
            current_elapsed = checkpoint_data.get('current_phase_elapsed', 0.0)
            
            self.times.update(timer_state)
            self.times['current_phase_elapsed'] = current_elapsed
            
            if is_main_process:
                print(f"Restored timer state from checkpoint")
                if current_elapsed > 0:
                    print(f"Will resume with {current_elapsed:.1f}s already elapsed")
    
    def get_total_time(self):
        """Get total experiment time."""
        if self.times['total_start_time']:
            return time.time() - self.times['total_start_time']
        return 0.0
    
    def print_final_summary(self, is_main_process: bool):
        """Print final timing summary."""
        if not is_main_process:
            return
        
        total_time = self.get_total_time()
        
        print(f"\n{'='*80}")
        print(f"EXPERIMENT TIMING SUMMARY")
        if self.world_size > 1:
            print(f"Multi-GPU Training: {self.world_size} GPUs")
        print(f"{'='*80}")
        print(f"Total experiment time: {self._format_time(total_time)}")
        print(f"Total cumulative pretraining time: {self._format_time(self.times['pretrain_total_time'])}")
        print(f"Total cumulative finetuning time: {self._format_time(self.times['finetune_total_time'])}")
        print()
        print(f"Most recent model pretraining time: {self._format_time(self.times['most_recent_pretrain_time'])}")
        print(f"Most recent model finetuning time: {self._format_time(self.times['most_recent_finetune_time'])}")
        
        if self.world_size > 1:
            recent_pretrain_gpu_hours = self.times['most_recent_pretrain_time'] * self.world_size
            recent_finetune_gpu_hours = self.times['most_recent_finetune_time'] * self.world_size
            total_pretrain_gpu_hours = self.times['pretrain_total_time'] * self.world_size
            total_finetune_gpu_hours = self.times['finetune_total_time'] * self.world_size
            
            print(f"\nEffective compute time (GPU-hours):")
            print(f"  Most recent model pretraining: {self._format_time(recent_pretrain_gpu_hours)}")
            print(f"  Most recent model finetuning: {self._format_time(recent_finetune_gpu_hours)}")
            print(f"  Total cumulative pretraining: {self._format_time(total_pretrain_gpu_hours)}")
            print(f"  Total cumulative finetuning: {self._format_time(total_finetune_gpu_hours)}")
        
        if self.times['fold_times']:
            print(f"\nPer-fold breakdown:")
            print(f"{'-'*80}")
            for fold_name, fold_times in self.times['fold_times'].items():
                print(f"{fold_name:10} | "
                      f"Pretrain: {self._format_time(fold_times['pretrain_time']):>12} | "
                      f"Finetune: {self._format_time(fold_times['finetune_time']):>12} | "
                      f"Total: {self._format_time(fold_times['total_time']):>12}")
        
        print(f"{'='*80}\n")
        
        # Final save
        if is_main_process:
            self.save_results(is_main_process)
    
    def save_results(self, is_main_process: bool):
        """Save timing results to disk."""
        if not is_main_process or not self.results_path:
            return
            
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        
        results_data = self.times.copy()
        results_data['total_experiment_time'] = self.get_total_time()
        results_data['updated_at'] = time.time()
        results_data['updated_at_readable'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        
        with open(self.results_path, 'w') as f:
            yaml.dump(results_data, f, default_flow_style=False, indent=2)
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        return str(timedelta(seconds=int(seconds)))

# Keep backward compatibility and add convenience function
Timer = DistributedTimer


def create_timer(results_dir: str = None, experiment_name: str = "experiment") -> DistributedTimer:
    """Create a timer with simplified settings."""
    
    results_path = None
    if results_dir:
        results_path = os.path.join(results_dir, f"{experiment_name}_timing_results.yaml")
    
    return DistributedTimer(results_path=results_path)


def _densify_lookup_entry(lookup: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild one batch's lookup tensors in place, and return the entry.

    Idempotent: an entry that already carries ``slot_values`` has no
    ``sparse`` key left to consume and is returned untouched, which is
    what lets every reader of the family call this defensively.
    """

    if 'sparse' not in lookup:
        return lookup

    blocks = lookup.pop('sparse')
    batch_size, max_ts_len = lookup['indicators'].shape[:2]
    for key in ('slot_values', 'doses', 'masks'):
        lookup[key] = []

    for block in blocks:
        episodes, timesteps = block['episode_index'], block['timestep_index']
        for key, source in (('slot_values', 'values'),
                            ('doses', 'doses'),
                            ('masks', 'masks')):
            entries = block[source]
            if entries is None:
                # A single-slot feature's weight is 1 by definition, so
                # it has no dose or mask tensor to build (section 4.3).
                lookup[key].append(None)
                continue
            # The trailing dimensions come from the entries themselves --
            # (D,) for a single-slot feature, (S, D) otherwise -- so a
            # feature no episode in the batch filled still lands at its
            # own width rather than needing the shape carried beside it.
            dense = torch.zeros(
                (batch_size, max_ts_len, *entries.shape[1:]),
                dtype=entries.dtype, device=entries.device
            )
            if episodes.numel():
                dense[episodes, timesteps] = entries
            lookup[key].append(dense)

    return lookup


def densify_lookup_slots(batch: MixedTensorDataset) -> MixedTensorDataset:
    """Rebuild the lookup family's dense tensors from the collated blocks.

    ``collate_tensorized`` ships one flat block per lookup feature --
    episode index, timestep index, and the entries -- rather than the
    dense ``(batch, ts_len, ...)`` tensors the model consumes. Records in
    the family are rare against the timestep axis, so the dense form is
    almost entirely zeros: about 4.8 GB per batch at section 4.5's
    ``T = 500`` and a batch of 200, carrying about 7 MB of content.

    This runs inside ``move_batch_to_device``, so it builds the dense
    tensors wherever the batch has just landed. VRAM is unchanged --
    those tensors were already built there -- and what it saves is the
    worker boundary and the host-side copy in front of it.

    Args:
        batch: A batch whose ``val_data['lookup']`` entry holds
            ``sparse``. A batch with no lookup feature, or one already
            densified, is returned untouched.

    Returns:
        The same batch, with ``slot_values``, ``doses`` and ``masks`` in
        place of the sparse blocks.
    """

    lookup = (batch.get('val_data', {}).get('lookup')
              if isinstance(batch, dict) else None)
    if isinstance(lookup, dict):
        _densify_lookup_entry(lookup)
    return batch


def move_batch_to_device(batch: MixedTensorDataset, device: torch.device) -> MixedTensorDataset:
    """Recursively move all tensors in a batch to the specified device.
    
    This is needed when using custom samplers that bypass accelerator.prepare_data_loader(),
    which would otherwise handle automatic device transfer.
    
    Args:
        batch: The batch dictionary from the dataloader
        device: Target device (e.g., accelerator.device)
        
    Returns:
        The batch with all tensors moved to the specified device
    """
    def _move_to_device(obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(device, non_blocking=True)
        elif isinstance(obj, dict):
            return {k: _move_to_device(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_move_to_device(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(_move_to_device(item) for item in obj)
        else:
            return obj

    # Densified after the move, so the dense tensors are allocated on the
    # target device and only the family's records cross the boundary.
    return densify_lookup_slots(_move_to_device(batch))


def ensure_float32(data: MixedDataset) -> MixedDataset:
    """Converts float64-valued tensors in `data` to float32.

    MPS doesn't support float64 tensors, so this function is used to ensure that all tensors in the dataset are float32.

    Args:
        data (MixedDataset): The dataset to be converted.
    
    Returns:
        MixedDataset: The converted dataset.
    """

    for i, tnsr in enumerate(data):
        if tnsr.dtype == torch.float64:
            data[i] = tnsr.float()
    return data


def get_non_pad_mask(seq: Tensor, padding_value: int = 0) -> Tensor:
    """Get the mask for non-padding items in the input sequence
    
    Given a predetermined maximum sequence length, this function returns a mask tensor that indicates which tokens
    in the sequence are *not* padding tokens.

    Args:
        seq (Tensor): A `Tensor` of shape [batch size, sequence length] containing the input sequence.
        padding_value (int, optional): The value of the padding tokens in `seq`. Defaults to 0.
    
    Returns:
        Tensor: A `Tensor` of shape [batch size, sequence length, 1] containing the mask for non-padding tokens.
    """

    assert seq.dim() == 2
    non_padding_mask = seq.ne(padding_value).type(torch.float32).unsqueeze(-1)  # ne: not equal 
    return non_padding_mask


def pool_lookup_slots(
    slot_values: Tensor,
    doses: Optional[Tensor],
    masks: Optional[Tensor]
) -> Tensor:
    """Reduce one lookup feature's slots to a single vector per timestep.

    The one drug-specific step in the family, and it sits *before* the family rather than
    inside it (section 5.1): dose-scale, zero the unused slots, sum, and divide by the mask
    sum. Text passes through as a no-op -- one slot, weight 1 -- so this is a single
    parameterized step and not a fork.

    Doing it here, in the forward pass, rather than in ``__getitem__`` is what leaves a
    gradient on an individual slot and therefore on an individual DIN (section 4.3). Pooling
    earlier destroys that attribution irrecoverably.

    Args:
        slot_values: Tensor of shape (batch_size, max_timeseries_length, n_slots, D_f), or
            (batch_size, max_timeseries_length, D_f) for a single-slot feature.
        doses: Tensor of shape (batch_size, max_timeseries_length, n_slots) of relative daily
            quantities, or None for a single-slot feature.
        masks: Tensor of shape (batch_size, max_timeseries_length, n_slots), 1 on a slot a
            record actually filled, or None for a single-slot feature.

    Returns:
        Tensor: The pooled values, of shape (batch_size, max_timeseries_length, D_f).
    """

    if doses is None:
        # One slot, weight 1: the values are already what the encoder consumes. Returned
        # rather than copied, so a single-slot feature's pooled tensor *is* its slot tensor;
        # the ELECTRA forward's in-place substitution of generated values therefore reaches
        # both, which is the behaviour the stacked text tensor had before section 5.1 and
        # nothing in a forward pass reads the slots again after it.
        return slot_values
    weights = (doses * masks).unsqueeze(-1)
    pooled = (slot_values * weights).sum(dim=-2)
    # A timestep reaching the pool with no unmasked slot would otherwise give 0/0.
    return pooled / masks.sum(dim=-1).clamp(min=1).unsqueeze(-1)


def resolve_lookup_embeddings(lookup: Dict[str, Any]) -> List[Tensor]:
    """Pool every lookup feature in a batch's ``lookup`` entry, once.

    The result is memoized under ``lookup['embedded_values']`` because the ELECTRA forward pass
    reads it three times over -- to extract the generator's targets, to feed the generator, and,
    after the generator's predictions have replaced the masked positions, to feed the
    discriminator -- and the substitution has to survive into the discriminator's input rather
    than being pooled away again.

    Args:
        lookup: A batch's ``val_data['lookup']``, carrying ``slot_values``, ``doses`` and
            ``masks`` as per-feature lists.

    Returns:
        List[Tensor]: One (batch_size, max_timeseries_length, D_f) tensor per lookup feature.
    """

    # Densified if it has not been already. `move_batch_to_device` normally does this; the call
    # is idempotent and free once the tensors exist.
    _densify_lookup_entry(lookup)
    if 'embedded_values' not in lookup:
        lookup['embedded_values'] = [
            pool_lookup_slots(values, doses, masks) for values, doses, masks
            in zip(lookup['slot_values'], lookup['doses'], lookup['masks'])
        ]
    return lookup['embedded_values']


def combine_value_and_lookup_data(
        value_assoc_indicators: Tensor,
        value_assoc_values: Tensor,
        lookup_assoc_indicators: Tensor,
        lookup_embeddings: List[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """Concatenate the lookup family's embeddings with the value-associated data.

        Text and drugs are one family (section 5.1): a drug feature is a lookup feature with
        dose-weighted slots, a text feature is the same thing with one unweighted slot, and both
        arrive here already reduced to one `(batch_size, max_timeseries_length, D_f)` tensor per
        feature. The per-feature list is what lets `D_f` differ between features -- text is 4096 or
        8192 beside ClinVec's 128 -- which a single stacked tensor cannot represent.

        Note:
            The `value_assoc_indicators` and `lookup_assoc_indicators` tensors must be aligned along
            the last dimension, meaning the feature order and batch/timestep alignment must be
            consistent for correct concatenation. `lookup_embeddings` must be in the same feature
            order as the columns of `lookup_assoc_indicators`.

        Args:
            value_assoc_indicators: Tensor of shape (batch_size, max_timeseries_length, n_features) with indicators for 
                value-associated data.
            value_assoc_values: Tensor of shape (batch_size, max_timeseries_length, *) with value-associated 
                data, where * is the total number of numeric and categorical feature dimensions.
            lookup_assoc_indicators: Tensor of shape (batch_size, max_timeseries_length, n_lookup_features) for
                lookup-associated indicators.
            lookup_embeddings: List of one Tensor of shape (batch_size, max_timeseries_length, D_f) per lookup
                feature, `D_f` being that feature's embedding width.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - Concatenated indicators of shape (batch_size, max_timeseries_length, total_num_features), where
                  `total_num_features` is the sum of the number of numeric, categorical, and lookup features.
                - Concatenated values of shape (batch_size, max_timeseries_length, total_feature_dim), where
                  `total_feature_dim` is the sum of numeric, categorical, and lookup feature dimensions.
        """

        # Concatenate value- and lookup-associated indicators
        indicators = torch.cat([value_assoc_indicators, lookup_assoc_indicators], dim=-1)
        # One concatenation, not two: `cat(embeds)` followed by `cat([values, embeds_cat])` would
        # materialize an intermediate the size of the whole embedded block (section 5.1).
        values = torch.cat([value_assoc_values, *lookup_embeddings], dim=-1)

        return indicators, values


def generate_record_masks(
    data: MixedTensorDataset,
    feature_sample_rate: float = 0.15,
    obs_unobs_ratio: float = 4.0,
    subsample_rate: float = 0.5
) -> Tuple[Dict[str, Dict[str, Union[Tensor, List[Tensor]]]], Tensor]:
    """Generate masks for observed and unobserved records in the dataset.

    This function randomly selects records to mask in the dataset. It samples a fixed percentage of the observed records to mask and tries to sample a number of unobserved records such that the ratio of masked observed records to masked unobserved records satisfies a specified ratio (`obs_unobs_ratio`). For numeric features, the components of vector-valued features that were selected for masking are subsampled so that only a specified portion of the components are masked; the number of components to subsample is max(1, floor(subsample_rate * dim)), where `dim` is the number of components in the vector-valued feature. For categorical and ordinal features, which are stored as one-hot vectors, every component of the selected feature position is masked so that the generator predicts the full class distribution. Masked components are represented as ones; unmasked components are represented as zeros.

    Args:
        data: Batched MixedTensorDataset from DataLoader containing value-associated and event-associated data.
        feature_sample_rate (float): The rate at which features are sampled for masking.
        obs_unobs_ratio (float): The ratio of observed to unobserved records that sampling will try to achieve. If
            None, only observed records will be masked. This does not apply to lookup features, which only mask observed records because embeddings of unobserved text or drugs are zero vectors and the cosine similarity loss is not defined when one of the target or prediction vectors is zero.
        subsample_rate (float): The rate at which components of vector-valued features are subsampled for masking.

    Returns:
        Tuple[Dict[str, Dict[str, Union[Tensor, List[Tensor]]]], Tensor]: A tuple of masks for value-associated and event-associated data, respectively.
        Value-associated masks have the following structure:
        ```
        {
            'numeric': {
                'indicators': tensor(batch_size, max_ts_len, n_numeric_feats)  # Mask indicators for features
                'values': [  # List of features
                    tensor(batch_size, max_ts_len, feature_dim),  # Mask indicators for Feature 1 components
                    tensor(batch_size, max_ts_len, feature_dim),  # Mask indicators for Feature 2 components
                    ...  # More features
                ]
            },
            'categorical': {
                'indicators': tensor(batch_size, max_ts_len, n_cat_feats)  # Mask indicators for categorical features
                'values': [  # List of features
                    tensor(batch_size, max_ts_len, feature_dim),  # Mask indicators for Feature 1 components
                    tensor(batch_size, max_ts_len, feature_dim),  # Mask indicators for Feature 2 components
                    ...  # More features
                ]
            },
            'lookup': {
                'indicators': tensor(batch_size, max_ts_len, n_lookup_feats)  # Mask indicators for lookup features
                'embedded_values': [  # List of lookup features; widths are per-feature (section 5.1)
                    tensor(batch_size, max_ts_len, D_1),  # Mask indicators for Feature 1 components
                    tensor(batch_size, max_ts_len, D_2),  # Mask indicators for Feature 2 components
                    ...  # More features
                ]
            } 
        ```

        Event-associated masks have the following structure:
        ```
        tensor(batch_size, max_ts_len, n_event_feats)  # Mask indicators for features
        ```
    """

    # The lookup family reaches the masks dense, whether or not the caller came through
    # `move_batch_to_device`: both the mask widths below and `_gen_val_assoc_feat_mask` read
    # `slot_values`, and a caller reading a collated batch directly would otherwise fail on a
    # missing key. The call is idempotent and free once the tensors exist.
    densify_lookup_slots(data)

    # Get batch dimensions from the collated data structure
    batch_size, max_ts_len = data['val_data']['times'].shape
    batch_device = data['val_data']['times'].device

    val_masks = {}
    event_masks = None

    # Initialize value-associated data masks
    for feature_type in ['numeric', 'categorical', 'ordinal', 'multilabel', 'lookup']:
        if feature_type in data['val_data']:
            feature_data = data['val_data'][feature_type]
            # Get number of features from first episode's first timestep
            n_features = feature_data['indicators'].shape[-1]  # features per timestep
            # Initialize indicator mask tensor
            indicator_mask = torch.zeros_like(feature_data['indicators'], device=batch_device)
            if feature_type == 'lookup':
                # Lookup features carry pre-computed embeddings whose width is per-feature
                # (section 5.1), so each mask is sized from its own feature's tensor. The masks
                # are drawn before the model pools, so the width comes from the slot values,
                # whose last dimension is D_f whether or not there is a slot axis in front of it.
                val_masks[feature_type] = {
                    'indicators': indicator_mask,
                    'embedded_values': [
                        torch.zeros(
                            (batch_size, max_ts_len, values.shape[-1]), device=batch_device
                        )
                        for values in feature_data['slot_values']
                    ]
                }
            else:
                # For numeric/categorical, we need to determine feature dimensions
                val_masks[feature_type] = {
                    'indicators': indicator_mask,
                    'values': [torch.zeros_like(tnsr, device=batch_device) for tnsr in feature_data['values']]
                }

    # Initialize event-associated masks
    if 'event_data' in data:
        event_masks = torch.zeros_like(data['event_data']['indicators'], device=batch_device)

    # Generate the value-associated masks
    for feature_type in ['numeric', 'categorical', 'ordinal', 'multilabel', 'lookup']:
        if feature_type in val_masks:
            _gen_val_assoc_feat_mask(
                data, feature_type, val_masks, feature_sample_rate, obs_unobs_ratio, subsample_rate
            )
    
    # Generate the event-associated data masks
    if event_masks is not None:
        _gen_event_assoc_feat_mask(
            data, event_masks, feature_sample_rate, obs_unobs_ratio
        )

    return val_masks, event_masks


def _gen_val_assoc_feat_mask(
    data: Dict, feature_type: str, val_masks: Dict, 
    feature_sample_rate: float, obs_unobs_ratio: float, subsample_rate: float
):
    """Generate value-associated feature masks using vectorized operations.
    
    Feature masking is done over all batch instances combined.
    """

    indicators_data = data['val_data'][feature_type]['indicators']
    if feature_type != 'lookup':
        values_key = 'values'
        values_data = data['val_data'][feature_type][values_key]
    else:
        # The family's masks are over the pooled vector, but the pool happens in the forward
        # pass, so the width is read from the unpooled slot values -- whose last dimension is
        # D_f whether or not there is a slot axis in front of it.
        values_key = 'embedded_values'
        values_data = data['val_data'][feature_type]['slot_values']
    padding_mask = data['val_data']['masks'].unsqueeze(-1)  # (batch_size, max_ts_len, 1)

    batch_size, max_ts_len, n_features = indicators_data.shape
    device = indicators_data.device

    # Identify observed and unobserved positions
    obs_feats = (indicators_data == 1) & padding_mask.bool()
    unobs_feats = (indicators_data == 0) & padding_mask.bool()

    obs_positions = torch.nonzero(obs_feats, as_tuple=False)
    unobs_positions = torch.nonzero(unobs_feats, as_tuple=False)

    obs_count = obs_positions.size(0)
    unobs_count = unobs_positions.size(0)

    n_obs_masked = int(feature_sample_rate * obs_count)
    if obs_unobs_ratio is None or feature_type == 'lookup':
        n_unobs_masked = 0
    else:
        # Attempt to maintain the specified observed-to-unobserved ratio. If there are not enough unobserved positions to satisfy the ratio, mask all the unobserved positions. If the number of unobserved positions to sample is calculated to be less than 1, mask one unobserved position (if any exist).
        n_unobs_masked = min(unobs_count, max(1, int(n_obs_masked / obs_unobs_ratio))) if unobs_count > 0 else 0

    # Sample positions globally first
    selected_obs = None
    selected_unobs = None
    
    if n_obs_masked > 0:
        perm = torch.randperm(obs_count, device=device)[:n_obs_masked]
        selected_obs = obs_positions[perm]
        # Set indicator masks for observed positions
        val_masks[feature_type]['indicators'][selected_obs[:, 0], selected_obs[:, 1], selected_obs[:, 2]] = 1.0
    
    if n_unobs_masked > 0:
        perm = torch.randperm(unobs_count, device=device)[:n_unobs_masked]
        selected_unobs = unobs_positions[perm]
        # Set indicator masks for unobserved positions
        val_masks[feature_type]['indicators'][selected_unobs[:, 0], selected_unobs[:, 1], selected_unobs[:, 2]] = 1.0

    # Combine all selected positions
    if selected_obs is not None and selected_unobs is not None:
        all_selected = torch.cat([selected_obs, selected_unobs], dim=0)
    elif selected_obs is not None:
        all_selected = selected_obs
    elif selected_unobs is not None:
        all_selected = selected_unobs
    else:
        return  # Nothing to mask

    # Process value masks per feature (vectorized within each feature)
    for f in range(n_features):
        feat_dim = values_data[f].shape[-1]

        # Filter to positions for this feature
        feat_mask = all_selected[:, 2] == f
        feat_positions = all_selected[feat_mask]

        n_pos = feat_positions.size(0)
        if n_pos == 0:
            continue

        if feature_type in ('categorical', 'ordinal'):
            # Categorical and ordinal features are stored as one-hot vectors; the generator
            # predicts class logits / CLM probabilities for the whole vector. Mask every
            # component of the one-hot row at a selected (batch, timestep) position.
            val_masks[feature_type][values_key][f][
                feat_positions[:, 0], feat_positions[:, 1], :
            ] = 1.0
        else:
            n_components_to_mask = max(1, int(subsample_rate * feat_dim))

            # Generate component masks for all positions at once using random sorting
            rand_vals = torch.rand(n_pos, feat_dim, device=device)
            _, component_order = rand_vals.sort(dim=1)
            selected_components = component_order[:, :n_components_to_mask]  # (n_pos, n_components)

            # Build expanded index tensors
            batch_idx = feat_positions[:, 0].unsqueeze(1).expand(-1, n_components_to_mask)
            time_idx = feat_positions[:, 1].unsqueeze(1).expand(-1, n_components_to_mask)

            # Flatten and assign
            val_masks[feature_type][values_key][f][
                batch_idx.reshape(-1),
                time_idx.reshape(-1),
                selected_components.reshape(-1)
            ] = 1.0


def _gen_event_assoc_feat_mask(
    data: Dict, event_masks: Tensor, 
    feature_sample_rate: float, obs_unobs_ratio: float
):
    """Generate event-associated feature masks using efficient tensor operations."""

    indicators_data = data['event_data']['indicators']
    padding_mask = data['event_data']['masks'].unsqueeze(-1)

    # Create boolean masks for observed and unobserved features
    obs_feats = (indicators_data == 1) & padding_mask.bool()
    unobs_feats = (indicators_data == 0) & padding_mask.bool()

    # Get tensor positions (avoid .tolist() conversion)
    obs_positions = torch.nonzero(obs_feats, as_tuple=False)
    unobs_positions = torch.nonzero(unobs_feats, as_tuple=False)

    # Calculate the number of observed and unobserved features to mask
    n_obs_masked = int(feature_sample_rate * obs_positions.size(0))
    if obs_unobs_ratio is None:
        n_unobs_masked = 0
    else:
        # Attempt to maintain the specified observed-to-unobserved ratio. If there are not enough unobserved positions to satisfy the ratio, mask all the unobserved positions. If the number of unobserved positions to sample is calculated to be less than 1, mask one unobserved position (if any exist).
        n_unobs_masked = min(
            unobs_positions.size(0), max(1, int(n_obs_masked / obs_unobs_ratio))
        ) if unobs_positions.size(0) > 0 else 0

    # Process both observed and unobserved positions efficiently
    for positions, n_masked in [(obs_positions, n_obs_masked), (unobs_positions, n_unobs_masked)]:
        if n_masked > 0:
            perm = torch.randperm(positions.size(0))[:n_masked]
            selected = positions[perm]
            # Vectorized assignment
            event_masks[selected[:,0], selected[:,1], selected[:,2]] = 1.0
    
    return event_masks


def format_pretraining_performance_table(
    epoch: int,
    current_train_losses: dict,
    current_val_losses: dict,
    best_train_losses: dict,
    best_val_losses: dict,
    use_thp_pred_loss: bool
) -> str:
    """Format an ASCII table for pretraining metrics."""
    
    # Fixed width calculation:
    # 37 (max description width) + 4 (spaces) + 17 (max value width) = 58
    # Total table width with borders: 58 + 2 = 60
    TABLE_WIDTH = 60
    CONTENT_WIDTH = 58
    DESC_WIDTH = 37  # Maximum description width
    VALUE_WIDTH = 17  # Maximum value width (12 digits + 1 decimal + 4 trailing)
    
    def format_value(value, is_string=False):
        """Format a value to occupy exactly 17 characters if ≤12 leading digits."""
        if is_string:
            return f"{value:>{VALUE_WIDTH}}"
        
        # Format the number with 4 decimal places
        formatted = f"{value:.4f}"
        
        # Find the decimal point position to count leading digits
        decimal_pos = formatted.find('.')
        leading_digits = decimal_pos
        
        # If 12 or fewer leading digits, pad to exactly 17 characters
        if leading_digits <= 12:
            padding_needed = max(0, 12 - leading_digits)
            padded_value = " " * padding_needed + formatted
            return f"{padded_value:>{VALUE_WIDTH}}"[:VALUE_WIDTH]
        else:
            # More than 12 leading digits - no padding, will misalign
            return formatted
    
    def format_loss_section(losses: dict, section_title: str) -> list:
        lines = []
        lines.append(f"│ {section_title:<{CONTENT_WIDTH}} │")
        lines.append("├" + "─" * TABLE_WIDTH + "┤")
        
        # Each line: description (37 chars) + 4 spaces + value (17 chars)
        lines.append(f"│ {'Mean total loss:':<{DESC_WIDTH}}    {format_value(losses['Optimization_Loss'])} │")
        lines.append(f"│ {'Generator:':<{DESC_WIDTH}}    {format_value(losses['Generator_Loss'])} │")
        lines.append(f"│ {'Discriminator:':<{DESC_WIDTH}}    {format_value(losses['Discriminator_Loss'])} │")
        lines.append(f"│ {'THP (overall, weighted):':<{DESC_WIDTH}}    {format_value(losses['THP_Loss'])} │")
        lines.append(f"│ {'THP neg. log likelihood (unweighted):':<{DESC_WIDTH}}    {format_value(losses['THP_NLL_Loss'])} │")
        
        # Handle THP prediction losses
        if use_thp_pred_loss:
            lines.append(f"│ {'THP event type loss (unweighted):':<{DESC_WIDTH}}    {format_value(losses['THP_Type_Loss'])} │")
            lines.append(f"│ {'THP time loss (unweighted):':<{DESC_WIDTH}}    {format_value(losses['THP_Time_Loss'])} │")
        else:
            lines.append(f"│ {'THP event type loss (unweighted):':<{DESC_WIDTH}}    {format_value('Not used', is_string=True)} │")
            lines.append(f"│ {'THP time loss (unweighted):':<{DESC_WIDTH}}    {format_value('Not used', is_string=True)} │")
        
        return lines
    
    # Build the complete table
    table_lines = []
    
    # Title
    title = f"Pretraining Performance, Epoch {epoch}"
    table_lines.append("┌" + "─" * TABLE_WIDTH + "┐")
    table_lines.append(f"│ {title:^{CONTENT_WIDTH}} │")
    table_lines.append("├" + "─" * TABLE_WIDTH + "┤")
    
    # Current epoch validation losses
    table_lines.extend(format_loss_section(current_val_losses, "Current Epoch Mean Losses (Validation)"))
    table_lines.append("├" + "─" * TABLE_WIDTH + "┤")
    
    # Current epoch training losses
    table_lines.extend(format_loss_section(current_train_losses, "Current Epoch Mean Losses (Training)"))
    table_lines.append("├" + "─" * TABLE_WIDTH + "┤")
    
    # Best epoch validation losses
    table_lines.extend(format_loss_section(best_val_losses, "Best Epoch Mean Losses (Validation)"))
    table_lines.append("├" + "─" * TABLE_WIDTH + "┤")
    
    # Best epoch training losses  
    table_lines.extend(format_loss_section(best_train_losses, "Best Epoch Mean Losses (Training)"))
    
    # Close table
    table_lines.append("└" + "─" * TABLE_WIDTH + "┘")
    
    return "\n".join(table_lines)


def format_finetuning_performance_table(
    task: str,
    train_scores: dict,
    val_scores: dict,
    test_scores: dict
) -> str:
    """Format an ASCII table for finetuning metrics."""
    
    # Fixed width calculation: same as pretraining table
    TABLE_WIDTH = 60
    CONTENT_WIDTH = 58
    DESC_WIDTH = 37  # Maximum description width
    VALUE_WIDTH = 17  # Maximum value width (12 digits + 1 decimal + 4 trailing)
    
    def format_value(value, is_string=False):
        """Format a value to occupy exactly 17 characters if ≤12 leading digits."""
        if is_string:
            return f"{value:>{VALUE_WIDTH}}"
        
        # Format the number with 4 decimal places
        formatted = f"{value:.4f}"
        
        # Find the decimal point position to count leading digits
        decimal_pos = formatted.find('.')
        leading_digits = decimal_pos
        
        # If 12 or fewer leading digits, pad to exactly 17 characters
        if leading_digits <= 12:
            padding_needed = max(0, 12 - leading_digits)
            padded_value = " " * padding_needed + formatted
            return f"{padded_value:>{VALUE_WIDTH}}"[:VALUE_WIDTH]
        else:
            # More than 12 leading digits - no padding, will misalign
            return formatted
    
    def format_score_section(scores: dict, section_title: str) -> list:
        lines = []
        lines.append(f"│ {section_title:<{CONTENT_WIDTH}} │")
        lines.append("├" + "─" * TABLE_WIDTH + "┤")
        
        if task == 'mortality':
            lines.append(f"│ {'Mean cross-entropy loss:':<{DESC_WIDTH}}    {format_value(scores['Loss_Cross_Entropy'])} │")
            lines.append(f"│ {'Accuracy:':<{DESC_WIDTH}}    {format_value(scores['Accuracy'])} │")
            lines.append(f"│ {'AUROC:':<{DESC_WIDTH}}    {format_value(scores['AUROC'])} │")
            lines.append(f"│ {'AUPRC:':<{DESC_WIDTH}}    {format_value(scores['AUPRC'])} │")
            lines.append(f"│ {'F1:':<{DESC_WIDTH}}    {format_value(scores['F1_Score'])} │")

        elif task == 'length_of_stay':
            # CORRECTED: Use the actual keys from calculate_finetuning_eval_metrics
            lines.append(f"│ {'Mean squared error loss:':<{DESC_WIDTH}}    {format_value(scores['Loss_Mean_Squared_Error'])} │")
            lines.append(f"│ {'Mean absolute difference:':<{DESC_WIDTH}}    {format_value(scores['Mean_Absolute_Error'])} │")

        else:  # phenotype
            lines.append(f"│ {'Mean cross-entropy loss:':<{DESC_WIDTH}}    {format_value(scores['Loss_Cross_Entropy'])} │")
            lines.append(f"│ {'Microaveraged AUROC:':<{DESC_WIDTH}}    {format_value(scores['Micro_averaged_AUROC'])} │")
            lines.append(f"│ {'Macroaveraged AUROC:':<{DESC_WIDTH}}    {format_value(scores['Macro_averaged_AUROC'])} │")
        
        return lines
    
    # Build the complete table
    table_lines = []
    
    # Title
    title = f"Finetuned {task} Model Performance"
    table_lines.append("┌" + "─" * TABLE_WIDTH + "┐")
    table_lines.append(f"│ {title:^{CONTENT_WIDTH}} │")
    table_lines.append("├" + "─" * TABLE_WIDTH + "┤")
    
    # Training set scores
    table_lines.extend(format_score_section(train_scores, "Training set"))
    table_lines.append("├" + "─" * TABLE_WIDTH + "┤")
    
    # Validation set scores
    table_lines.extend(format_score_section(val_scores, "Validation set"))
    table_lines.append("├" + "─" * TABLE_WIDTH + "┤")
    
    # Test set scores
    table_lines.extend(format_score_section(test_scores, "Test set"))
    
    # Close table
    table_lines.append("└" + "─" * TABLE_WIDTH + "┘")
    
    return "\n".join(table_lines)


def convert_to_python_types(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert tensor and numpy types in a dictionary to native Python types."""

    converted = {}
    
    loss_key_order = [
        'Optimization_Loss',
        'Generator_Loss', 
        'Discriminator_Loss',
        'THP_Loss',
        'THP_NLL_Loss',
        'THP_Type_Loss',
        'THP_Time_Loss'
    ]
    
    # Add keys in the specified order
    for key in loss_key_order:
        if key in data:
            val = data[key]
            if hasattr(val, 'item'):  # It's a tensor
                converted[key] = float(val.item())
            elif isinstance(val, (torch.Tensor, np.ndarray)):
                converted[key] = float(val)
            else:
                converted[key] = val
    
    # Add any remaining keys not in the specified order
    for key, val in data.items():
        if key not in converted:
            if hasattr(val, 'item'):  # It's a tensor
                converted[key] = float(val.item())
            elif isinstance(val, (torch.Tensor, np.ndarray)):
                converted[key] = float(val)
            else:
                converted[key] = val
                
    return converted


def calc_time_diff(event_times: Tensor, non_pad_mask: Tensor, device: str) -> Tensor:
        """Calculate time differences between consecutive events.

        Temporal differences are calculated between the current
        timestamp and the previous one. A delta is valid only when
        *both* the current and previous timesteps are non-padding;
        otherwise the delta is zeroed out. This prevents spurious
        deltas at the boundary between left-padded zeros and the
        first real historical timestep.

        Args:
            event_times (Tensor): Event timestamps of shape
                (batch_size, max_ts_len)
            non_pad_mask (Tensor): Non-padding mask of shape
                (batch_size, max_ts_len)

        Returns:
            Tensor: Time differences of shape
                (batch_size, max_ts_len). The first timestep has
                a time difference of zero.

        """

        time_diff = torch.zeros_like(event_times, device=device)
        # A delta at position i is valid only when both position i
        # and position i-1 are non-padding.
        both_valid = non_pad_mask[:, 1:] * non_pad_mask[:, :-1]
        time_diff[:, 1:] = (
            (event_times[:, 1:] - event_times[:, :-1]) * both_valid
        )

        return time_diff


def sample_non_event_time_diff(time_diff_seq: Tensor, n: int, device: str) -> Tensor:
    """Sample non-event time differences using uniform distribution.

    Given a sequences of time differences between the time of the each observed event and the one preceding it, this function uniformly samples time differences in [0, time_diff). In other words, it uniformly samples timestamps between consecutive events and returns the difference between the sampled timestamp and the previous observed timestamp.

    Note that Zuo et al. used random sampling on the interval [0, time_diff), but this implementation uses evenly spaced sampling on the interval [0, time_diff], inclusive of the bounds. While intensities are technically undefined at the left side of the interval, it's okay to include the bounds of the interval because the sampled times are being used to estimate the integral of the intensity function over the interval, and including the bounds should improve the accuracy of the integral estimate without invalidating it.

    Args:
        n (int): Number of samples to draw
        time_diff_seq (Tensor): Time differences of shape (batch_size, max_ts_len)

    Returns:
        Tensor: Sampled time differences of shape (batch_size, max_ts_len, n_samples)
    """

    sampled_ratios = torch.linspace(0., 1., n, device=device)[None, None, :]  # (1, 1, n_samples)
    sampled_time_diffs = time_diff_seq[:, :, None] * sampled_ratios  # (batch_size, max_ts_len, n_samples)

    return sampled_time_diffs


def get_param_shapes(model: torch.nn.Module) -> OrderedDict[str, Tuple[int]]:
    """Get the expected shapes of all parameters and buffers in a model.
    
    This should be called BEFORE wrapping the model with FSDP, as FSDP will
    flatten parameters and make their shapes unreliable. The returned dictionary
    can then be used to reshape flattened parameters from FSDP state dicts.
    
    Args:
        model: An unwrapped model (ELECTRA or MixedClassifier)
    
    Returns:
        OrderedDict mapping parameter names to their shapes as tuples
    """
    param_shapes = OrderedDict()
    
    # Get all parameters
    for name, param in model.named_parameters():
        param_shapes[name] = tuple(param.shape)
    
    # Get all buffers (like running stats in BatchNorm)
    for name, buffer in model.named_buffers():
        param_shapes[name] = tuple(buffer.shape)
    
    return param_shapes


def print_peak_memory(accelerator: Accelerator):
    """Print peak memory usage across all ranks."""
    
    if not torch.cuda.is_available():
        return
    
    peak_gb = torch.cuda.max_memory_allocated(accelerator.device) / (1024**3)
    
    rank_data = {
        'rank': accelerator.process_index,
        'peak_gb': peak_gb
    }
    
    all_ranks = accelerator.gather_for_metrics([rank_data])
    
    if accelerator.is_main_process:
        print("\n" + "="*60)
        print("PEAK MEMORY USAGE")
        print("="*60)
        for data in all_ranks:
            print(f"Rank {data['rank']}: {data['peak_gb']:.3f} GB")
        print("="*60 + "\n")
    
    accelerator.wait_for_everyone()
    torch.cuda.reset_peak_memory_stats(accelerator.device)


def convert_model_to_dtype(model: torch.nn.Module, dtype: torch.dtype = torch.bfloat16) -> torch.nn.Module:
    """Convert all parameters and buffers in a model to the specified dtype.
    
    This should be called BEFORE accelerator.prepare() to ensure FSDP sees uniform dtypes.
    
    Args:
        model: The model to convert
        dtype: Target dtype (default: torch.bfloat16)
    
    Returns:
        The model with converted parameters/buffers
    """
    for param in model.parameters():
        param.data = param.data.to(dtype)
    for buffer_name, buffer in model.named_buffers():
        # Skip buffers that should remain as integers (like position indices)
        if buffer.dtype in (torch.int64, torch.int32, torch.long, torch.bool):
            continue
        buffer.data = buffer.data.to(dtype)
    return model
