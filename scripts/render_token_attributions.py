"""Per-token attribution scores for the text cells TSR ranks highest.

Closes the loop between the two stages. `tsr_scores` ranks the cells of an episode and hands
back `lookup_grads`, the gradient of the target with respect to each text cell's pooled
embedding; `attribution.text_cell_attributions` turns one of those into a score per token. This
script picks the cells worth explaining, runs stage 2 over them, and writes the tokens out with
their scores.

The scoring function is a flag. `--scorer` names one of `attribution.SCORERS`:

    grad_x_input      signed; what TSR's own R(.) uses, so a token map and the cell score
                      above it answer the same question. The default.
    abs_grad_x_input  its magnitude, for ranking without regard to direction
    grad_l2           unsigned sensitivity: how much any change to this token would move
                      the target, regardless of which way the token actually points
    grad_l1           the same, less dominated by a single large component

They differ only in how a per-token vector is collapsed to a number -- the encoder backward is
identical -- so a run can be repeated under another scorer for the cost of the backward alone,
and the maps are comparable because they came from one gradient.

Output
------

A CSV at ``--output``: ``episode_id``, ``timestep``, ``feature``, ``cell_score`` (the TSR score
of the cell the tokens come from), ``token_index``, ``token_id``, ``token``, ``score``. One row
per non-padding token of each explained cell.

**That file contains record text**, which is why it is a file and not stdout: the summary
printed here is counts alone, in the house style of ``ibd_onset.py`` and ``episode_occupancy.py``.
Treat the CSV as patient data.

Cost
----

Stage 1 dominates and is TSR's: at the open gate one episode is thousands of perturbations. The
token pass is one encoder backward per explained cell, so ``--top-cells`` sets a cost that is
small beside it. Start with a handful of episodes.

Usage:
    python scripts/render_token_attributions.py dataset.yaml experiment.yaml <name> \
        --fold fold0 --split test --episodes 8 --top-cells 5 \
        --scorer grad_x_input --output tokens.csv
"""

import argparse
import csv
import numpy as np
import os
import pickle
import torch
import yaml

import _path  # noqa: F401  (repository root on sys.path)

from TransEHR2.attribution import SCORERS, text_cell_attributions
from TransEHR2.cli import resolve_device
from TransEHR2.constants import LLM_NAME
from TransEHR2.data.preprocessing import (collate_tensorized,
                                          lookup_feat_widths,
                                          prepare_dataloaders,
                                          value_encoder_dims)
from TransEHR2.models import MixedClassifier
from TransEHR2.modules import (DeepHitHead, EventDataEncoder,
                               GradientTraceableLLM, ValueDataEncoder)
from TransEHR2.survival import TimeGrid, cif_logit_target
from TransEHR2.tsr import tsr_scores
from TransEHR2.utils import move_batch_to_device

TEXT = 'text'


def load_config(path):
    with open(path, 'r') as handle:
        return yaml.safe_load(handle)


def load_tokens_table(data_dir):
    """``text_tokens.npy`` as a memory map, and the pad id ``embed.py`` recorded."""
    tables = os.path.join(data_dir, 'lookup_tables')
    tokens_path = os.path.join(tables, 'text_tokens.npy')
    if not os.path.exists(tokens_path):
        raise FileNotFoundError(
            f'{tokens_path} is missing. It is written by embed.py, and a token score cannot '
            f'be rendered without the ids the embedding was built from.'
        )
    # embed.py records the resolved pad id in metadata.pkl, beside LLM_NAME and
    # MAX_TOKEN_LENGTH, so a tokenizer change that invalidated text_tokens.npy is detectable.
    # The mask is derived from it, and guessing would mask the wrong columns.
    meta_path = os.path.join(data_dir, 'extracted', 'metadata.pkl')
    with open(meta_path, 'rb') as handle:
        metadata = pickle.load(handle)
    if 'pad_token_id' not in metadata:
        raise KeyError(
            f"{meta_path} carries no 'pad_token_id'. embed.py writes it when it builds the "
            f"text table; an extraction that has not been embedded cannot be rendered."
        )
    return np.load(tokens_path, mmap_mode='r'), int(metadata['pad_token_id'])


def cell_rows(dataset, episode_index, lookup_pos):
    """``{timestep: row in the text table}`` for one episode's text feature.

    The batch cannot answer this. ``MixedDataset`` gathers a lookup record's embedding out of
    the table in ``__getitem__`` -- ``lookup_tables[f][csr['values'][...]]`` -- so what reaches
    the model is the vector and the row index is left behind in the CSR. It is read back here
    from the same arrays, which is also what keeps the timestep axis honest: the CSR carries a
    timestep per record and collation places each at that index, so the key is directly the
    timestep TSR scored.
    """
    csr = dataset.lookup_csr[lookup_pos]
    start = int(csr['offsets'][episode_index])
    end = int(csr['offsets'][episode_index + 1])
    if end <= start:
        return {}
    timesteps = np.asarray(csr['timesteps'][start:end]).tolist()
    values = np.asarray(csr['values'][start:end]).tolist()
    return dict(zip(timesteps, values))


def text_feature_positions(index, lookup_feat_types):
    """``{position in the lookup family: position in `index`}`` for text features only.

    Drug cells have a pooled embedding and a gradient too, but their slots are ClinVec rows
    rather than token sequences, so there is nothing to render them onto.
    """
    positions = {}
    lookup_seen = 0
    for i, (family, _) in enumerate(index):
        if family != 'lookup':
            continue
        if lookup_feat_types[lookup_seen] == TEXT:
            positions[lookup_seen] = i
        lookup_seen += 1
    return positions


def top_cells(scores, occupied, feature_column, k):
    """The ``k`` highest-scoring occupied timesteps of one feature, descending."""
    column = scores[:, feature_column].clone()
    column[~occupied[:, feature_column]] = float('-inf')
    k = min(k, int(occupied[:, feature_column].sum()))
    if k <= 0:
        return torch.empty(0, dtype=torch.long)
    return torch.topk(column, k).indices


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Write per-token attribution scores for an episode\'s top text cells.')
    parser.add_argument('dataset_config')
    parser.add_argument('experiment_config')
    parser.add_argument('experiment_name')
    parser.add_argument('--model_dir', default='./models')
    parser.add_argument('--fold', default='fold0')
    parser.add_argument('--split', default='test', choices=('train', 'val', 'test'))
    parser.add_argument('--episodes', type=int, default=8,
                        help='How many episodes to explain, from the start of the split')
    parser.add_argument('--top-cells', type=int, default=5,
                        help='Text cells per episode, highest TSR score first')
    parser.add_argument('--scorer', default='grad_x_input', choices=sorted(SCORERS))
    parser.add_argument('--cause', type=int, default=0,
                        help='Position of the cause in the grid, not its EVENT_TYPE')
    parser.add_argument('--horizon-bin', type=int, default=0)
    parser.add_argument('--quantile', type=float, default=0.0,
                        help="TSR's time-relevance gate")
    parser.add_argument('--chunk', type=int, default=64)
    parser.add_argument('--ig-steps', type=int, default=0)
    parser.add_argument('--device', default=None)
    parser.add_argument('--output', default='token_attributions.csv')
    args = parser.parse_args(argv)

    if args.ig_steps:
        print('NOTE: --ig-steps integrates stage 1, and the gradients reaching stage 2 are\n'
              '      integrated to match (tsr._relevance averages them without the 1/alpha).\n'
              '      The token scores are then integrated gradients too, not grad x input.')

    device = resolve_device(args.device)
    dataset_config = load_config(args.dataset_config)
    experiment_config = load_config(args.experiment_config)
    data_dir = dataset_config['DATA_DIR']

    grid = TimeGrid(
        cuts_days=experiment_config.get('TIME_GRID_CUTS_DAYS'),
        causes=experiment_config.get('MODELLED_CAUSES'),
        brier_integration_days=experiment_config.get('BRIER_INTEGRATION_DAYS'))
    target = cif_logit_target(args.cause, args.horizon_bin, grid.n_causes, grid.n_bins)

    tokens_table, pad_token_id = load_tokens_table(data_dir)
    llm = GradientTraceableLLM(use_gradient_checkpointing=False).to(device)
    llm.model.eval()
    for parameter in llm.model.parameters():
        parameter.requires_grad_(False)
    print(f'encoder: {LLM_NAME}')

    model = build_classifier(dataset_config, experiment_config, grid, data_dir)
    weights = os.path.join(args.model_dir, args.experiment_name, args.fold,
                           'pretrained', 'finetuned_readmission.pt')
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=False))
    model = model.to(device).eval()

    loaders = prepare_dataloaders(data_dir, args.fold, batch_size=1, num_workers=0)
    names = ('train', 'val', 'test') if len(loaders) == 3 else ('train', 'test')
    dataset = dict(zip(names, loaders))[args.split].dataset

    # Episodes are taken by dataset index and collated one at a time, not drawn from the
    # DataLoader: the train loader shuffles, and the CSR lookup below is keyed by dataset
    # index, so an episode has to be identified by the index it was read at.
    feat_types = dataset.lookup_feat_types
    rows_written, episodes_done, cells_done = 0, 0, 0

    with open(args.output, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['episode_id', 'timestep', 'feature', 'cell_score',
                         'token_index', 'token_id', 'token', 'score'])

        for episode_index in range(min(args.episodes, len(dataset))):
            batch = collate_tensorized([dataset[episode_index]])
            batch = move_batch_to_device(batch, device)
            out = tsr_scores(model, batch, target=target, quantile=args.quantile,
                             chunk=args.chunk, ig_steps=args.ig_steps)
            positions = text_feature_positions(out['index'], feat_types)

            episode_id = dataset.episode_ids[episode_index]

            for lookup_pos, column in positions.items():
                steps = top_cells(out['scores'][0], out['occupied'][0], column,
                                  args.top_cells)
                if not len(steps):
                    continue
                # A text feature has one unweighted slot, so the gradient with respect to
                # the pooled embedding is the gradient for that row's own vector.
                by_timestep = cell_rows(dataset, episode_index, lookup_pos)
                steps = torch.tensor([t for t in steps.tolist() if t in by_timestep],
                                     dtype=torch.long)
                if not len(steps):
                    continue
                rows = torch.tensor([by_timestep[t] for t in steps.tolist()],
                                    dtype=torch.long)
                grads = out['lookup_grads'][lookup_pos][0, steps]
                scores = text_cell_attributions(
                    llm, tokens_table, rows.cpu(), grads, pad_token_id,
                    score=SCORERS[args.scorer])
                cells_done += len(steps)

                ids = np.asarray(tokens_table[rows.cpu().numpy()])
                for cell, step in enumerate(steps.tolist()):
                    keep = ids[cell] != pad_token_id
                    tokens = llm.tokenizer.convert_ids_to_tokens(
                        ids[cell][keep].tolist())
                    cell_score = float(out['scores'][0, step, column])
                    for position, (token_id, token) in enumerate(
                            zip(ids[cell][keep].tolist(), tokens)):
                        writer.writerow([
                            episode_id, step, feat_types[lookup_pos], f'{cell_score:.6g}',
                            position, token_id, token,
                            f'{float(scores[cell, position]):.6g}'])
                        rows_written += 1
            episodes_done += 1

    print(f'episodes explained: {episodes_done}')
    print(f'text cells:         {cells_done}')
    print(f'token rows written: {rows_written}')
    print(f'scorer:             {args.scorer}')
    print(f'\nWrote {args.output}. It carries record text -- treat it as patient data.')
    return 0


def build_classifier(dataset_config, experiment_config, grid, data_dir) -> MixedClassifier:
    """The finetuned classifier's architecture, rebuilt to load weights into.

    NOTE: this is the third copy of this construction -- ``run_experiment.py`` and
    ``dump_finetuned_predictions.py`` each carry one, both as closures inside ``main``. They
    should be one shared builder. Unifying them touches the script that runs the training, so
    it is deliberately not done here; it is a tidy-up of its own.
    """
    variable_properties = load_config(dataset_config['VARIABLE_PROPERTIES_PATH'])
    lookup_dims = lookup_feat_widths(os.path.join(data_dir, 'extracted'))
    n_val_feats, tot_val_feat_dim = value_encoder_dims(
        variable_properties, dataset_config['VALUED_FEATS'], lookup_dims)

    def setting(key, default=None):
        return experiment_config.get(key, default)

    d_disc = setting('DISCRIMINATOR_ENCODER_D_MODEL', 256)
    d_thp = setting('THP_ENCODER_D_MODEL', 256)
    static_dim = 0

    return MixedClassifier(
        event_encoder=EventDataEncoder(
            num_types=len(dataset_config['EVENT_FEATS']), d_model=d_thp,
            d_inner=setting('THP_ENCODER_D_INNER', 256),
            n_layers=setting('THP_ENCODER_N_LAYERS', 1),
            n_head=setting('THP_ENCODER_N_HEADS', 2),
            d_k=setting('THP_ENCODER_D_K', 128), d_v=setting('THP_ENCODER_D_V', 64),
            dropout=setting('THP_ENCODER_DROPOUT', 0.1),
            normalize_before=setting('THP_ENCODER_NORM_FIRST', True)),
        val_encoder=ValueDataEncoder(
            n_features=n_val_feats, feat_dim=tot_val_feat_dim, d_model=d_disc,
            n_heads=setting('DISCRIMINATOR_ENCODER_N_HEADS', 2),
            n_encoder_blocks=setting('DISCRIMINATOR_ENCODER_N_ENCODER_BLOCKS', 1),
            dim_feedforward=setting('DISCRIMINATOR_ENCODER_DIM_FEEDFORWARD', 256),
            dropout=setting('DISCRIMINATOR_ENCODER_DROPOUT', 0.1),
            activation=setting('DISCRIMINATOR_ENCODER_ACTIVATION', 'gelu'),
            norm=setting('DISCRIMINATOR_ENCODER_NORM', 'LayerNorm'),
            normalize_before=setting('DISCRIMINATOR_ENCODER_NORM_FIRST', True)),
        d_event_enc=d_thp, d_val_enc=d_disc, d_statics=static_dim,
        num_classes=grid.n_causes * grid.n_bins,
        aggr=setting('PREDICTOR_AGGREGATION_METHOD', 'mean'),
        use_lookup=bool(lookup_dims),
        head=DeepHitHead(
            d_in=MixedClassifier.encoding_width(d_thp, d_disc, static_dim),
            n_causes=grid.n_causes, n_bins=grid.n_bins,
            d_shared=setting('DEEPHIT_HEAD_D_SHARED', [256, 128]),
            d_cause=setting('DEEPHIT_HEAD_D_CAUSE', [128, 64]),
            dropout=setting('DEEPHIT_HEAD_DROPOUT', 0.1)),
    )


if __name__ == '__main__':
    raise SystemExit(main())
