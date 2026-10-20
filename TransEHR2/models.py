import torch

from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union

from TransEHR2.data.custom_types import MixedTensorDataset, ValueAssociatedTensorData
from TransEHR2.modules import MaskedTokenDiscriminator, MaskedTokenGenerator, TransformerHawkesProcess
from TransEHR2.modules import EventDataEncoder, ValueDataEncoder
from TransEHR2.utils import (
    calc_time_diff, resolve_lookup_embeddings, sample_non_event_time_diff
)


class ELECTRA(torch.nn.Module):
    """The pretraining model: a masked generator, a discriminator, and
    optionally a Transformer Hawkes process over the event stream.

    ``hawkes`` is optional because the THP term can be switched off. Its
    log-likelihood integrates an intensity over inter-event gaps, so a
    broad dynamic range of timestamps makes it diverge, and no reweighting
    fixes that -- it is the shape of the loss, not its scale. Passing None
    leaves the event stream out of *pretraining* only; the downstream
    classifier still encodes it, starting from a random initialization
    rather than from pretrained weights.
    """

    def __init__(
        self,
        generator: MaskedTokenGenerator,
        discriminator: MaskedTokenDiscriminator,
        hawkes: Optional[TransformerHawkesProcess] = None,
        use_lookup: bool = False,
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.hawkes = hawkes
        self.use_lookup = use_lookup

    def _extract_masked_targets(
        self,
        value_data: ValueAssociatedTensorData,
        record_masks: Dict[str, Dict[str, Tensor]]
    ) -> Dict[str, Dict[str, Union[List[Tensor], Tensor]]]:
        """
        Extract original values at masked positions before in-place batch modification.
        
        This method captures the ground truth values that will be needed for generator loss
        computation. By extracting only the masked values (rather than deep-copying the entire
        batch), we significantly reduce VRAM usage during the forward pass.
        
        For numeric features, we store the masked value components (sparse extraction).
        For categorical features, we store the masked one-hot encoded values.
        For lookup features, we store complete embedding vectors at masked positions since
        the cosine similarity loss requires the full embedding for normalization.
        
        Args:
            value_data: Original value-associated data from the batch, containing numeric,
                categorical, and/or lookup features with their values and indicators.
            record_masks: Masks indicating which records and feature values were hidden from
                the generator. A value of 1 indicates a record was masked, 0 indicates not.
                
        Returns:
            Dictionary with the same structure as value_data, but containing only the values
            at masked positions. The structure is:
            {
                'numeric': {
                    'values': List[Tensor],  # Flattened masked values per feature
                    'indicators': Tensor     # Masked indicator values (if predict_indicators)
                },
                'categorical': {
                    'values': List[Tensor],  # Masked one-hot values per feature  
                    'indicators': Tensor     # Masked indicator values (if predict_indicators)
                },
                'lookup': {
                    'embedded_values': List[Tensor],  # Full embeddings at masked positions
                    'indicators': Tensor              # Masked indicator values (if predict_indicators)
                }
            }
        """
        masked_targets = {}
        
        for feat_type in ['numeric', 'categorical', 'ordinal', 'multilabel', 'lookup']:
            if feat_type not in value_data or feat_type not in record_masks:
                continue

            masked_targets[feat_type] = {'values': []}

            if feat_type == 'numeric':
                # For numeric features, extract only the masked value components.
                # Each feature may have different dimensionality, so we store per-feature tensors.
                for i, orig_vals in enumerate(value_data[feat_type]['values']):
                    # value_mask shape: (batch_size, max_ts_len, feature_dim)
                    value_mask = record_masks[feat_type]['values'][i].bool()
                    # Extract masked values as a 1D tensor of all masked components
                    # This is memory-efficient as we only store ~15-25% of original values
                    masked_targets[feat_type]['values'].append(orig_vals[value_mask].clone())

            elif feat_type == 'categorical':
                # For categorical features, extract the full one-hot vectors at masked positions.
                # We need the complete one-hot encoding to compute cross-entropy loss.
                feature_masks = record_masks[feat_type]['indicators']  # (batch_size, max_ts_len, n_cat_feats)
                for i, orig_vals in enumerate(value_data[feat_type]['values']):
                    # feat_mask shape: (batch_size, max_ts_len)
                    feat_mask = feature_masks[:, :, i].bool()
                    # Extract complete categorical features at masked (batch, timestep) positions
                    # Result shape: (n_masked_positions, n_classes)
                    masked_targets[feat_type]['values'].append(orig_vals[feat_mask].clone())

            elif feat_type == 'multilabel':
                # For multilabel features, extract masked value components (like numeric).
                # Each class bit is independently masked and predicted via sigmoid + BCE.
                for i, orig_vals in enumerate(value_data[feat_type]['values']):
                    value_mask = record_masks[feat_type]['values'][i].bool()
                    masked_targets[feat_type]['values'].append(orig_vals[value_mask].clone())

            elif feat_type == 'ordinal':
                # For ordinal features, extract the full one-hot vectors at masked positions.
                # Values are stored as (batch, max_ts, n_levels) one-hot encodings; a zero
                # row encodes an unknown / out-of-domain value.
                feature_masks = record_masks[feat_type]['indicators']  # (batch_size, max_ts_len, n_ord_feats)
                for i, orig_vals in enumerate(value_data[feat_type]['values']):
                    # feat_mask shape: (batch_size, max_ts_len)
                    feat_mask = feature_masks[:, :, i].bool()
                    # Extract complete ordinal one-hot vectors at masked (batch, timestep) positions
                    # Result shape: (n_masked_positions, n_levels)
                    masked_targets[feat_type]['values'].append(orig_vals[feat_mask].clone())

            elif feat_type == 'lookup':
                # For lookup features, we must store the COMPLETE embedding vectors at masked positions.
                # The cosine similarity loss requires the full embedding for proper L2 normalization;
                # storing only masked components would corrupt the similarity computation.
                if 'embedded_values' in value_data[feat_type]:
                    # embedded_values: [(batch_size, max_ts_len, D_f) x n_lookup_feats]
                    orig_embeddings = value_data[feat_type]['embedded_values']
                    feature_masks = record_masks[feat_type]['indicators']  # (batch_size, max_ts_len, n_lookup_feats)

                    masked_targets[feat_type]['embedded_values'] = []
                    for i, feat_embeddings in enumerate(orig_embeddings):
                        # feat_mask shape: (batch_size, max_ts_len)
                        feat_mask = feature_masks[:, :, i].bool()
                        # Extract complete embeddings at masked positions
                        # Result shape: (n_masked_positions, D_f)
                        masked_targets[feat_type]['embedded_values'].append(
                            feat_embeddings[feat_mask].clone()
                        )
            
            # Extract masked indicators if the generator predicts them
            if self.generator.predict_indicators and value_data[feat_type].get('indicators') is not None:
                indicator_mask = record_masks[feat_type]['indicators'].bool()
                masked_targets[feat_type]['indicators'] = (
                    value_data[feat_type]['indicators'][indicator_mask].clone()
                )
        
        return masked_targets

    def _prepare_discriminator_input_inplace(
        self,
        value_data: ValueAssociatedTensorData,
        gen_output: ValueAssociatedTensorData,
        record_masks: Dict[str, Dict[str, Tensor]]
    ) -> Dict[str, Dict[str, Tensor]]:
        """Write the generator's guesses into the masked cells, in place, and label each cell.

        The batch becomes the discriminator's input, so call `_extract_masked_targets` first: after
        this the original values at masked cells are gone. Only masked cells are touched, and only
        by indexed writes, so nothing batch-sized is materialized.

        What is written at a masked cell:

        * **Sampled, not argmax.** A categorical or ordinal value is drawn from the generator's class
          distribution, a multilabel bit and an indicator from Bernoulli(sigmoid) -- ELECTRA samples
          its replacements. The most likely class every time would leave the discriminator only the
          generator's systematic misses to find, and would never write a rare class, so a rare class
          would read as "real" by construction. Numerics and lookup embeddings are point estimates.
        * **Detached.** ELECTRA trains the two networks on separate objectives; gradients from the
          discriminator do not reach the generator through its replacements. Undetached, the joint
          minimisation rewards the generator for writing replacements that are *easy* to spot.
          (The sampled writes carry no gradient anyway.)
        * **Consistent with its indicator.** The indicator written is the sampled one when the
          generator predicts indicators, and the original otherwise. Wherever it is 0 the cell gets
          the missing state -- the all-zero row every family uses for absence -- since "absent, with
          a value" is a combination no real record has and would give the cell away. Wherever it
          is 1 at a cell that was really absent, the whole predicted row is written, not the
          subsampled components, which would leave a partial vector no record has either.

        Returns:
            ``{family: {'indicators': (batch, max_ts_len, n_features)}}``, 1 where the cell the
            discriminator sees differs from the original and 0 elsewhere -- the discriminator's
            targets. A masked cell the generator reproduced exactly is real, as in ELECTRA; labelling
            it generated would ask the discriminator to tell apart two identical inputs.
        """
        labels = {}
        for feat_type in ['numeric', 'categorical', 'ordinal', 'multilabel', 'lookup']:
            # The generator output only has keys for feature types it predicts, and only for the
            # types present in the batch.
            if feat_type not in gen_output or feat_type not in value_data:
                continue
            entry = value_data[feat_type]
            value_key = 'embedded_values' if feat_type == 'lookup' else 'values'
            cell_mask = record_masks[feat_type]['indicators'].bool()  # (batch, max_ts_len, n_feats)

            original_ind = entry['indicators']
            written_ind = original_ind
            ind_logits = gen_output[feat_type].get('indicators')
            if self.generator.predict_indicators and ind_logits is not None:
                written_ind = original_ind.clone()
                drawn = torch.bernoulli(torch.sigmoid(ind_logits.detach()[cell_mask]))
                written_ind[cell_mask] = drawn.to(written_ind.dtype)
            changed = (written_ind != original_ind) & cell_mask

            for i, dest in enumerate(entry[value_key]):
                cell = cell_mask[..., i]
                if not cell.any():
                    continue
                rows = cell.nonzero(as_tuple=True)  # (batch indices, timestep indices)
                old = dest[rows]  # (n_masked, D)
                guess = self._sample_replacement(
                    feat_type, gen_output[feat_type][value_key][i].detach()[rows]
                ).to(dest.dtype)
                components = record_masks[feat_type][value_key][i].bool()[rows]
                new = torch.where(components, guess, old)
                present = written_ind[..., i][rows] != 0
                was_absent = original_ind[..., i][rows] == 0
                fake_presence = present & was_absent
                new[fake_presence] = guess[fake_presence]
                new[~present] = 0
                changed[rows + (i,)] |= (new != old).any(dim=-1)
                dest[rows] = new

            if written_ind is not original_ind:
                original_ind.copy_(written_ind)
            labels[feat_type] = {'indicators': changed.to(original_ind.dtype)}
        return labels

    @staticmethod
    def _sample_replacement(feat_type: str, prediction: Tensor) -> Tensor:
        """One replacement row per masked cell from the generator's ``(n_masked, D)`` output."""
        if feat_type == 'categorical':
            drawn = torch.distributions.Categorical(logits=prediction).sample()
            return torch.nn.functional.one_hot(drawn, num_classes=prediction.shape[-1])
        if feat_type == 'ordinal':
            # The ordinal head is a cumulative link model and already emits class probabilities.
            # Rounding can leave a hair below zero; Categorical renormalises the rest.
            drawn = torch.distributions.Categorical(probs=prediction.clamp_min(0)).sample()
            return torch.nn.functional.one_hot(drawn, num_classes=prediction.shape[-1])
        if feat_type == 'multilabel':
            return torch.bernoulli(torch.sigmoid(prediction))
        return prediction

    def compute_conditional_intensity(self, encodings, prev_event_times, time_diff):
        """Wrapper to access hawkes submodule's method from parent model.
        
        For compatibility with model sharding using Accelerate.
        """
        return self.hawkes.compute_conditional_intensity(encodings, prev_event_times, time_diff)

    def compute_initial_intensity(self, batch_size):
        """Wrapper to access hawkes submodule's method from parent model.

        For compatibility with model sharding using Accelerate.
        """
        return self.hawkes.compute_initial_intensity(batch_size)

    def forward(
        self,
        batch: MixedTensorDataset,
        record_masks: Dict[str, Dict[str, List[List[Tensor]]]],
        device: str,
        trace_grads: bool = False,
        compute_intensities: bool = False,
        thp_loss_mc_samples: int = 100
    ) -> Dict[str, Union[Tensor, Tuple[Tensor, Tensor], Dict[str, Dict[str, Tensor]]]]:
        """
        Forward pass through the ELECTRA model.
        
        Args:
            batch: MixedTensorDataset containing value data, event data, static data, and targets
            record_masks: Masks indicating which records to simulate with generator
            device: The device where tensors will be sent.
            trace_grads: Whether to trace gradients through the LLM for XAI
            compute_intensities: Whether to compute conditional and initial intensities for the Hawkes process
            thp_loss_mc_samples: Number of Monte Carlo samples for THP loss estimation
            
        Returns:
            Dict containing:
                - 'hawkes_encodings': Event sequence encodings (if event_data present)
                - 'hawkes_predictions': Tuple of (event_type_pred, time_pred) (if event_data present)
                - 'thp_intensities': Dict of intensity values for THP loss (if compute_intensities)
                - 'generator': Generator output predictions
                - 'discriminator': Discriminator output predictions
                - 'masked_targets': Extracted original values at masked positions for generator loss
                - 'discriminator_labels': ``{family: {'indicators': labels}}``, 1 where the
                  discriminator's input differs from the original; MaskedDiscriminatorLoss's targets
        """
        outputs = {}
        
        # Process event data if the Hawkes process is switched on and there
        # is an event stream to give it.
        if self.hawkes is not None and 'event_data' in batch:
            event_enc, event_pred = self.hawkes(batch['event_data'])
            outputs['hawkes_encodings'] = event_enc
            outputs['hawkes_predictions'] = event_pred  # A tuple of (event_type_prediction, time_prediction)

            if compute_intensities:
                batch_size = event_enc.size(0)
                event_data = batch['event_data']
                event_times = event_data['times']
                event_non_padding_masks = event_data['masks']
                
                # Compute time differences
                time_diff_obs = calc_time_diff(event_times, event_non_padding_masks, device=device)
                # Sample inter-event time differences for Monte Carlo integration
                time_diff_samples = sample_non_event_time_diff(
                    time_diff_obs[:, 1:], n=thp_loss_mc_samples, device=device
                )
                
                # Compute intensity values
                obs_initial_intensities = self.hawkes.compute_initial_intensity(batch_size)
                obs_conditional_intensities = self.hawkes.compute_conditional_intensity(
                    encodings=event_enc[:, :-1, :],
                    prev_event_times=event_times[:, :-1],
                    time_diff=time_diff_obs[:, 1:]
                )
                sampled_intensities = self.hawkes.compute_conditional_intensity(
                    encodings=event_enc[:, :-1, :],
                    prev_event_times=event_times[:, :-1],
                    time_diff=time_diff_samples
                )
                
                # Store intensity values in outputs
                outputs['thp_intensities'] = {
                    'obs_initial': obs_initial_intensities,
                    'obs_conditional': obs_conditional_intensities,
                    'sampled': sampled_intensities,
                    'time_diff_obs': time_diff_obs,
                    'time_diff_samples': time_diff_samples
                }
            
        value_data = batch['val_data']  # Extract the ValueAssociatedTensorData from the batch
        # Reduce the lookup family's slots to one vector per feature per timestep, once and here.
        # It has to happen before the targets are extracted, because the pooled
        # vector is what the generator reconstructs, and before the generator's predictions
        # replace the masked positions, because that substitution is the discriminator's input.
        if 'lookup' in value_data:
            resolve_lookup_embeddings(value_data['lookup'])

        # MEMORY OPTIMIZATION: Extract masked target values BEFORE in-place modification.
        # This stores only the values needed for generator loss (~15-25% of batch) rather than
        # deep-copying the entire batch, significantly reducing VRAM usage.
        masked_targets = self._extract_masked_targets(value_data, record_masks)
        outputs['masked_targets'] = masked_targets

        # Generate predictions for masked values
        gen_output = self.generator(value_data, record_masks)
        outputs['generator'] = gen_output
        
        # Prepare input for discriminator by replacing masked values with generated ones IN-PLACE.
        # After this call, value_data contains generated values at masked positions, and the
        # labels say which cells now differ from the original -- the discriminator's targets.
        outputs['discriminator_labels'] = self._prepare_discriminator_input_inplace(
            value_data, gen_output, record_masks)
        
        # Get discriminator predictions using the modified batch
        disc_output = self.discriminator(value_data, batch.get('static_data', None))
        outputs['discriminator'] = disc_output

        return outputs
    

class MixedClassifier(torch.nn.Module):
    """A classifier that combines event and value-associated time series data for prediction.
    
    This model processes both event sequences and value-associated data through separate encoders,
    aggregates their outputs, optionally incorporates static data, and makes final predictions.
    """
    
    def __init__(
        self,
        event_encoder: EventDataEncoder,
        val_encoder: ValueDataEncoder,
        d_event_enc: int,
        d_val_enc: int,
        d_statics: int,
        num_classes: int,
        aggr: str = 'max',
        use_lookup: bool = False,
        head: Optional[torch.nn.Module] = None,
    ):
        """Initialize MixedClassifier.

        Args:
            event_encoder: Encoder for event-associated data, or None to leave the event stream out
                of the head's input altogether (``USE_EVENT_ENCODER: False``). The branch is then
                skipped even though collation still delivers ``event_data``, and ``d_event_enc``
                must be 0.
            val_encoder: Encoder for value-associated time series data
            d_event_enc: Dimensionality of event encoder output
            d_val_enc: Dimensionality of time series encoder output
            d_statics: Dimensionality of static data (0 if no static data)
            num_classes: Number of output classes
            aggr: Aggregation method ('max', 'mean' or 'none') for sequence-level encoding. 'none'
                reads the most recent record: the last timestep the mask keeps.
            use_lookup: If True, the model will expect pre-computed lookup embeddings in the input batch.
            head: Module mapping the pooled encoding to logits. Defaults to the
                two-layer stack this class has always used, which returns
                `num_classes` logits. A DeepHitHead goes here instead, because a
                competing-risks head needs per-cause capacity that a shared
                32-unit bottleneck cannot give it.
        """

        super().__init__()
        if event_encoder is None and d_event_enc:
            raise ValueError(f'd_event_enc is {d_event_enc} but there is no event encoder')
        self.event_encoder = event_encoder
        self.val_encoder = val_encoder
        d_enc = self.encoding_width(d_event_enc, d_val_enc, d_statics)
        if head is None:
            head = torch.nn.Sequential(
                torch.nn.Linear(d_enc, 32),
                torch.nn.GELU(),
                torch.nn.Linear(32, num_classes),
            )
        self.head = head
        self.aggr = aggr
        self.use_lookup = use_lookup

    @staticmethod
    def encoding_width(d_event_enc: int, d_val_enc: int, d_statics: int) -> int:
        """Width of the vector the head receives.

        `forward` concatenates the event encoding, the value encoding and the
        static block in that order, so this is their sum. It exists as a
        method because a caller building its own head has to size it, and
        two copies of this sum would drift the moment the concatenation
        changes.
        """
        return d_event_enc + d_val_enc + d_statics

    @staticmethod
    def _most_recent(encodings: Tensor, masks: Tensor) -> Tensor:
        """Each row's encoding at the last timestep its mask keeps, or zeros if it keeps none.

        Series are right-aligned with padding on the left, so the most recent record is the
        rightmost kept column -- not column ``n_kept - 1``, which counts from the wrong end. Read
        off the mask rather than assumed to be the final column, so a timestep masked out after
        collation (TSR's deletion) hands the readout to the record before it.
        """
        n_steps = masks.shape[1]
        positions = torch.arange(n_steps, device=masks.device).expand_as(masks)
        last = torch.where(masks.bool(), positions, torch.full_like(positions, -1)).max(dim=1).values
        picked = encodings[torch.arange(encodings.shape[0], device=encodings.device), last.clamp(min=0)]
        return torch.where((last >= 0).unsqueeze(-1), picked, torch.zeros_like(picked))

    def forward(self, batch: MixedTensorDataset, trace_grads: bool = False) -> Tensor:
        """Forward pass through the mixed classifier.
        
        Args:
            batch (MixedTensorDataset): MixedTensorDataset containing event data, value data, and optionally static data
            trace_grads (bool): Whether to trace gradients through the LLM

        Returns:
            Tensor: Classification logits of shape (batch_size, num_classes)
        """

        embeddings = []
        
        # Process event data if the model encodes it. Collation delivers event_data either way.
        if self.event_encoder is not None and 'event_data' in batch:

            event_data = batch['event_data']

            event_indicators = event_data['indicators']
            event_times = event_data['times']
            event_masks = event_data['masks']
            
            # Pass the event data through the encoder
            #     Shape: (batch_size, max_ts_length, d_event_enc)
            event_enc = self.event_encoder(event_indicators, event_times, event_masks)

            # Aggregate event encodings across the time dimension (dim=1)
            #     event_enc final shape: (batch_size, d_event_enc)
            # NOTE nan_to_num: when every episode in a batch has padding at a
            # given timestep, the attention softmax receives all -inf and
            # produces NaN.  Replacing with 0 before masking is safe because
            # those positions are padding for all episodes and would be zeroed
            # out regardless.  Without this, NaN * 0 = NaN (IEEE 754)
            # propagates through the sum/max aggregation.
            event_enc = torch.nan_to_num(event_enc, nan=0.0)
            event_enc = event_enc * event_masks[..., None].float()  # Zero out padding embeddings
            n_obs_records = event_masks.sum(dim=-1, keepdim=True).clamp(min=1)  # Clamp to avoid errors

            if self.aggr == 'max':
                # Padding embeddings were zeroed just above, so a plain max returns 0 for any
                # channel whose observed values are all negative -- a value no record produced,
                # and one that depends on how much padding the episode happens to carry. Filling
                # padding with -inf means it can never win. Rows that are entirely padding come
                # back all -inf and are sent to the zero vector, which is what the other branches
                # give them too.
                event_enc = event_enc.masked_fill(event_masks[..., None] == 0, float('-inf'))
                event_enc, _ = torch.max(event_enc, dim=1)
                event_enc = torch.nan_to_num(event_enc, neginf=0.0)
            elif self.aggr == 'mean':
                event_enc = torch.sum(event_enc, dim=1) / n_obs_records
            elif self.aggr == 'none':
                # The most recent record; zeros if there are none
                event_enc = self._most_recent(event_enc, event_masks)
            
            embeddings.append(event_enc)
        
        # Process value-associated data if available
        if 'val_data' in batch:

            val_data = batch['val_data']

            val_times = val_data['times']
            val_masks = val_data['masks']
            
            # Combine all feature types along a single axis for the encoder
            inds_to_concat = []
            vals_to_concat = []
            
            # Process numeric features
            if 'numeric' in val_data and val_data['numeric']['values']:
                # Extract numeric feature indicators
                numeric_inds = val_data['numeric']['indicators']
                inds_to_concat.append(numeric_inds)
                # Extract and concatenate numeric feature values
                numeric_vals = torch.cat(val_data['numeric']['values'], dim=2)
                vals_to_concat.append(numeric_vals)
            
            # Process categorical features
            if 'categorical' in val_data and val_data['categorical']['values']:
                # Extract categorical feature indicators
                categorical_inds = val_data['categorical']['indicators']
                inds_to_concat.append(categorical_inds)
                # Extract and concatenate categorical feature values
                categorical_vals = torch.cat(val_data['categorical']['values'], dim=2)
                vals_to_concat.append(categorical_vals)

            # Process ordinal features
            if 'ordinal' in val_data and val_data['ordinal']['values']:
                # Extract ordinal feature indicators
                ordinal_inds = val_data['ordinal']['indicators']
                inds_to_concat.append(ordinal_inds)
                # Extract and concatenate ordinal feature values
                ordinal_vals = torch.cat(val_data['ordinal']['values'], dim=2)
                vals_to_concat.append(ordinal_vals)

            # Process multilabel features
            if 'multilabel' in val_data and val_data['multilabel']['values']:
                multilabel_inds = val_data['multilabel']['indicators']
                inds_to_concat.append(multilabel_inds)
                multilabel_vals = torch.cat(val_data['multilabel']['values'], dim=2)
                vals_to_concat.append(multilabel_vals)

            # Process lookup features
            if self.use_lookup and 'lookup' in val_data:
                # Extract lookup feature indicators, and pool each feature's slots. The pool is
                # in the forward pass so that a gradient reaches an individual drug slot, which
                # is what makes per-DIN attribution possible (sections 4.3, 5.1).
                lookup_inds = val_data['lookup']['indicators']
                inds_to_concat.append(lookup_inds)
                # A per-feature list, so extend rather than append.
                vals_to_concat.extend(resolve_lookup_embeddings(val_data['lookup']))
        
            if inds_to_concat and vals_to_concat:
                # Concatenate the tensors for numeric and categorical features along the feature dimension
                #   combined_val_indicators shape: (batch_size, max_timeseries_length, n_num_feats + n_cat_feats)
                #   combined_val_values shape: (batch_size, max_timeseries_length, total_feat_dim)
                combined_val_indicators = torch.cat(inds_to_concat, dim=2)
                combined_val_values = torch.cat(vals_to_concat, dim=2)
                
                # Pass through time series encoder
                val_enc = self.val_encoder(
                    combined_val_indicators, 
                    combined_val_values, 
                    val_times, 
                    val_masks
                )
                
                # Aggregate time series encodings across the time dimension (dim=1)
                #     val_enc final shape: (batch_size, d_val_enc)
                val_enc = torch.nan_to_num(val_enc, nan=0.0)
                val_enc = val_enc * val_masks[..., None].float() # Zero out padding embeddings
                n_obs_records = val_masks.sum(dim=-1, keepdim=True).clamp(min=1)  # Clamp to avoid errors

                if self.aggr == 'max':
                    # See the event branch: zeroed padding would win any channel whose observed
                    # values are all negative, so exclude it with -inf instead.
                    val_enc = val_enc.masked_fill(val_masks[..., None] == 0, float('-inf'))
                    val_enc, _ = torch.max(val_enc, dim=1)
                    val_enc = torch.nan_to_num(val_enc, neginf=0.0)
                elif self.aggr == 'mean':
                    # Exclude padding timesteps from mean calculation
                    val_enc = torch.sum(val_enc, dim=1) / n_obs_records
                elif self.aggr == 'none':
                    # The most recent record; zeros if there are none
                    val_enc = self._most_recent(val_enc, val_masks)

                embeddings.append(val_enc)
        
        # Add static data if available
        if 'static_data' in batch and batch['static_data'] is not None:
            embeddings.append(batch['static_data'])
        
        # Combine all embeddings
        if len(embeddings) > 1:
            enc = torch.cat(embeddings, dim=1)
        else:
            enc = embeddings[0]
        
        # Final classification layers
        return self.head(enc)
    