"""Sweep one pretraining hyperparameter at a time, on fold 0.

A coordinate sweep, not a grid: each hyperparameter named in
``HYPERPARAMETERS_TO_TUNE`` is varied over its list while the others sit at
their own list's first entry, which is therefore the default. The cost is
that interactions are invisible; the benefit is that k hyperparameters cost
sum(len) runs instead of prod(len), and a pretraining run is not cheap.

Selection is on the *test* split, deliberately. A sweep chooses between
settings, and the validation split has to stay unspent so that the run
chosen here can still select its own epoch on something it has not seen.

One process, one GPU. To spread the sweep, give each process its own
hyperparameter:

    for hp in DISC_LOSS_WEIGHT RECORD_MASK_RATIO; do
        CUDA_VISIBLE_DEVICES=$i python scripts/tune_hyperparameters.py \\
            dataset.yaml tune.yaml --only $hp &
    done; wait

Usage:
    python scripts/tune_hyperparameters.py <dataset_config> <experiment_config> \\
        [--only HP ...] [--device cuda:0] [--num_workers N]
"""

import argparse
import gc
import os
import torch
import yaml

from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, List

import _path  # noqa: F401  (repository root on sys.path)

from TransEHR2.cli import resolve_device
from TransEHR2.data.preprocessing import (
    compute_static_feat_dims, lookup_feat_widths, prepare_dataloaders,
    value_encoder_dims
)
from TransEHR2.models import ELECTRA
from TransEHR2.modules import (
    EventDataEncoder, MaskedTokenDiscriminator, MaskedTokenGenerator,
    TransformerHawkesProcess, ValueDataEncoder
)
from TransEHR2.routines import pretrain_with_hyperparameter
from TransEHR2.utils import convert_to_python_types, create_timer



# The sweep runs on one fold. Tuning on all of them would spend the
# cross-validation structure on hyperparameter selection, which is what it
# is there to be independent of.
FOLD_NAME = 'fold0'

# Pretraining knobs a sweep may vary. Anything outside this set named in
# HYPERPARAMETERS_TO_TUNE is refused rather than silently ignored: a typo
# would otherwise run the whole sweep at the default and report it as a
# result.
TUNABLE = {
    'DISC_LOSS_WEIGHT', 'THP_LOSS_NLL_WEIGHT', 'THP_LOSS_MC_SAMPLES',
    'THP_PRED_LOSS_TYPE_WT', 'THP_PRED_LOSS_TIME_WT', 'RECORD_MASK_RATIO',
    'OBS_UNOBS_SAMPLE_RATIO', 'CMPNT_MASK_RATIO', 'PRETRAIN_LEARNING_RATE',
    'PRETRAIN_LR_HALF_LIFE',
}


class HyperparameterSweep:
    """The settings each trial runs under.

    ``experiment_config`` gives a *list* for every tuned hyperparameter and
    a scalar for everything else. The first entry of each list is that
    hyperparameter's default, so the trials for one hyperparameter differ
    from each other in exactly one place.
    """

    def __init__(self, to_tune: List[str], experiment_config: dict):
        unknown = [hp for hp in to_tune if hp not in TUNABLE]
        if unknown:
            raise ValueError(
                f"HYPERPARAMETERS_TO_TUNE names {unknown}, which this script "
                f"cannot vary. Tunable: {sorted(TUNABLE)}"
            )
        missing = [hp for hp in to_tune if hp not in experiment_config]
        if missing:
            raise ValueError(f"{missing} are named for tuning but absent "
                             f"from the experiment config")

        self.to_tune = to_tune
        self.values = {hp: list(experiment_config[hp]) for hp in to_tune}
        self.defaults = {hp: self.values[hp][0] for hp in to_tune}

    def settings_for(self, hp_name: str, value: Any) -> Dict[str, Any]:
        """Every tuned hyperparameter's value for one trial."""
        settings = dict(self.defaults)
        settings[hp_name] = value
        return settings


def load_results(path: str, experiment_name: str) -> Dict[str, Any]:
    """Previous trials, so a restarted sweep does not repeat them."""
    blank = {'experiment': experiment_name, 'fold': FOLD_NAME,
             'task': 'pretrain'}
    if not os.path.exists(path):
        return blank
    try:
        with open(path, 'r') as f_in:
            return yaml.safe_load(f_in) or blank
    except Exception as e:
        print(f"Warning: could not read {path}: {e}")
        return blank


def record_trial(
    results: Dict[str, Any],
    hp_name: str,
    value: Any,
    losses: Dict[str, Any],
    sweep: HyperparameterSweep,
    seed_defaults: bool,
    path: str
) -> bool:
    """Store one trial's losses and write the file back.

    ``seed_defaults`` fills in every hyperparameter's default entry from
    the first trial, which is run with all of them at their defaults. It
    saves one redundant run per hyperparameter, and it is only correct
    once -- hence the flag, which this returns cleared.
    """
    safe = convert_to_python_types(losses)
    results.setdefault(hp_name, {})[value] = safe

    if seed_defaults:
        for hp in sweep.to_tune:
            default = sweep.defaults[hp]
            results.setdefault(hp, {}).setdefault(default, dict(safe))
        seed_defaults = False

    with open(path, 'w') as f_out:
        yaml.dump(results, f_out, default_flow_style=False, sort_keys=False)
    return seed_defaults


def main():
    parser = argparse.ArgumentParser(
        description='Sweep pretraining hyperparameters one at a time')
    parser.add_argument('dataset_config', type=str,
                        help='YAML file for dataset parameters')
    parser.add_argument('experiment_config', type=str,
                        help='YAML file for experiment parameters')
    parser.add_argument('--only', type=str, nargs='+', default=None,
                        help='Sweep only these hyperparameters. Default is '
                             'every name in HYPERPARAMETERS_TO_TUNE.')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to train on, e.g. cuda:0')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Worker processes for data loading')
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Sweeping on {device}")

    with open(args.dataset_config, 'r') as f_in:
        dataset_config = yaml.safe_load(f_in)
    DATA_DIR = dataset_config['DATA_DIR']
    VARIABLE_PROPERTIES_PATH = dataset_config['VARIABLE_PROPERTIES_PATH']
    VALUED_FEATS = dataset_config['VALUED_FEATS']
    EVENT_FEATS = dataset_config['EVENT_FEATS']
    STATIC_FEATS = dataset_config['STATIC_FEATS']

    with open(args.experiment_config, 'r') as f_in:
        cfg = yaml.safe_load(f_in)
    EXPERIMENT_NAME = cfg['EXPERIMENT_NAME']
    BATCH_SIZE = cfg['BATCH_SIZE']
    PREDICT_INDICATORS = cfg['PREDICT_INDICATORS']
    MODEL_DIR = cfg['MODEL_DIR']
    USE_THP = cfg.get('USE_THP', True)
    USE_THP_PRED_LOSS = cfg.get('USE_THP_PRED_LOSS', True)
    PRETRAIN_TOTAL_EPOCH = cfg.get('PRETRAIN_TOTAL_EPOCH', 1000)

    sweep = HyperparameterSweep(cfg['HYPERPARAMETERS_TO_TUNE'], cfg)
    to_sweep = args.only or sweep.to_tune
    unknown = [hp for hp in to_sweep if hp not in sweep.to_tune]
    if unknown:
        raise ValueError(f"--only names {unknown}, which are not in "
                         f"HYPERPARAMETERS_TO_TUNE")

    timer = create_timer(results_dir=f'./log/timing/{EXPERIMENT_NAME}',
                         experiment_name=EXPERIMENT_NAME)
    timer.start_total_timing()
    timer.start_fold(FOLD_NAME)

    with open(VARIABLE_PROPERTIES_PATH, 'r') as f_in:
        variable_properties = yaml.safe_load(f_in)
    static_dim = sum(
        compute_static_feat_dims(variable_properties, STATIC_FEATS))
    numeric_feat_dims, categorical_class_cnts = [], []
    ordinal_features, multilabel_class_cnts = [], []
    for feature in VALUED_FEATS:
        kind = variable_properties[feature]['type']
        if kind == 'numeric':
            numeric_feat_dims.append(variable_properties[feature]['size'])
        elif kind == 'categorical':
            categorical_class_cnts.append(
                len(variable_properties[feature]['category_map']))
        elif kind == 'ordinal':
            ordinal_features.append(
                len(variable_properties[feature]['category_map']))
        elif kind == 'multilabel':
            multilabel_class_cnts.append(variable_properties[feature]['size'])

    # The whole lookup family, whenever the extraction carries it: see
    # run_experiment.py.
    lookup_dims = lookup_feat_widths(os.path.join(DATA_DIR, 'extracted'))
    use_lookup = bool(lookup_dims)
    n_val_feats, tot_val_feat_dim = value_encoder_dims(
        variable_properties, VALUED_FEATS, lookup_dims)
    n_event_types = len(EVENT_FEATS)

    log_dir = f'./log/{EXPERIMENT_NAME}/{FOLD_NAME}/pretrained'
    evaluation_dir = f'{MODEL_DIR}/{EXPERIMENT_NAME}/pretrained/evaluation'
    evaluation_fp = f'{evaluation_dir}/evaluation_pretrained.yaml'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(evaluation_dir, exist_ok=True)

    loaders = prepare_dataloaders(
        DATA_DIR, FOLD_NAME, BATCH_SIZE, num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if args.num_workers > 0 else 1)
    train_loader, test_loader = loaders[0], loaders[-1]

    def build_value_encoder(prefix: str) -> ValueDataEncoder:
        return ValueDataEncoder(
            n_features=n_val_feats, feat_dim=tot_val_feat_dim,
            d_model=cfg[f'{prefix}_ENCODER_D_MODEL'],
            n_heads=cfg[f'{prefix}_ENCODER_N_HEADS'],
            n_encoder_blocks=cfg[f'{prefix}_ENCODER_N_ENCODER_BLOCKS'],
            dim_feedforward=cfg[f'{prefix}_ENCODER_DIM_FEEDFORWARD'],
            dropout=cfg[f'{prefix}_ENCODER_DROPOUT'],
            activation=cfg[f'{prefix}_ENCODER_ACTIVATION'],
            norm=cfg[f'{prefix}_ENCODER_NORM'],
            normalize_before=cfg.get(f'{prefix}_ENCODER_NORM_FIRST', True))

    def build_electra() -> ELECTRA:
        generator = MaskedTokenGenerator(
            encoder=build_value_encoder('GENERATOR'),
            d_model=cfg['GENERATOR_D_MODEL'],
            numeric_dims=numeric_feat_dims,
            categorical_classes=categorical_class_cnts,
            ordinal_features=ordinal_features or None,
            multilabel_classes=multilabel_class_cnts or None,
            lookup_dims=lookup_dims,
            predict_indicators=PREDICT_INDICATORS,
            dim_feedforward=cfg['GENERATOR_DIM_FEEDFORWARD'])
        discriminator = MaskedTokenDiscriminator(
            encoder=build_value_encoder('DISCRIMINATOR'),
            d_model=cfg['DISCRIMINATOR_ENCODER_D_MODEL'],
            n_numeric_features=len(numeric_feat_dims),
            n_categorical_features=len(categorical_class_cnts),
            n_ordinal_features=len(ordinal_features),
            n_multilabel_features=len(multilabel_class_cnts),
            n_lookup_features=len(lookup_dims),
            n_static_features=static_dim,
            dim_feedforward=cfg['DISCRIMINATOR_DIM_FEEDFORWARD'])
        hawkes = TransformerHawkesProcess(
            encoder=EventDataEncoder(
                num_types=n_event_types, d_model=cfg['THP_ENCODER_D_MODEL'],
                d_inner=cfg['THP_ENCODER_D_INNER'],
                n_layers=cfg['THP_ENCODER_N_LAYERS'],
                n_head=cfg['THP_ENCODER_N_HEADS'],
                d_k=cfg['THP_ENCODER_D_K'], d_v=cfg['THP_ENCODER_D_V'],
                dropout=cfg['THP_ENCODER_DROPOUT'],
                normalize_before=cfg.get('THP_ENCODER_NORM_FIRST', True)),
            num_types=n_event_types) if USE_THP else None
        return ELECTRA(generator=generator, discriminator=discriminator,
                       hawkes=hawkes, use_lookup=use_lookup)

    results = load_results(evaluation_fp, EXPERIMENT_NAME)
    seed_defaults = True

    for hp_name in to_sweep:
        values = sweep.values[hp_name]
        already = list(results.get(hp_name, {}))
        pending = [v for v in values if v not in already]
        print(f'\nTesting hyperparameter: {hp_name}')
        print(f'Values not yet tested: {pending}')

        for value in pending:
            settings = sweep.settings_for(hp_name, value)
            print(f'\n{"="*60}')
            print(f'TESTING: {hp_name} = {value}')
            for name, val in settings.items():
                if name != hp_name:
                    print(f'  {name} = {val} (DEFAULT)')
            print(f'{"="*60}\n')

            electra = build_electra()
            writer = SummaryWriter(log_dir)

            timer.start_phase('pretrain')
            try:
                _, best_test_losses = pretrain_with_hyperparameter(
                    hp_name=hp_name,
                    hp_value=value,
                    model=electra,
                    loaders=[train_loader, test_loader],
                    writer=writer,
                    learning_rate=settings.get(
                        'PRETRAIN_LEARNING_RATE',
                        cfg.get('PRETRAIN_LEARNING_RATE', 2e-3)),
                    device=device,
                    lr_half_life=settings.get(
                        'PRETRAIN_LR_HALF_LIFE',
                        cfg.get('PRETRAIN_LR_HALF_LIFE', None)),
                    total_epoch=PRETRAIN_TOTAL_EPOCH,
                    disc_loss_weight=settings.get(
                        'DISC_LOSS_WEIGHT', cfg.get('DISC_LOSS_WEIGHT', 1.0)),
                    use_thp=USE_THP,
                    thp_loss_nll_weight=settings.get(
                        'THP_LOSS_NLL_WEIGHT',
                        cfg.get('THP_LOSS_NLL_WEIGHT', 1e-2)),
                    thp_loss_mc_samples=settings.get(
                        'THP_LOSS_MC_SAMPLES',
                        cfg.get('THP_LOSS_MC_SAMPLES', 100)),
                    use_thp_pred_loss=USE_THP_PRED_LOSS,
                    thp_pred_loss_type_wt=settings.get(
                        'THP_PRED_LOSS_TYPE_WT',
                        cfg.get('THP_PRED_LOSS_TYPE_WT', 1.0)),
                    thp_pred_loss_time_wt=settings.get(
                        'THP_PRED_LOSS_TIME_WT',
                        cfg.get('THP_PRED_LOSS_TIME_WT', 1e-6)),
                    record_mask_ratio=settings.get(
                        'RECORD_MASK_RATIO',
                        cfg.get('RECORD_MASK_RATIO', 0.15)),
                    obs_unobs_sample_ratio=settings.get(
                        'OBS_UNOBS_SAMPLE_RATIO',
                        cfg.get('OBS_UNOBS_SAMPLE_RATIO', 5.0)),
                    cmpnt_mask_ratio=settings.get(
                        'CMPNT_MASK_RATIO',
                        cfg.get('CMPNT_MASK_RATIO', 0.25)),
                    checkpoint_dir=f'./checkpoints/{EXPERIMENT_NAME}/'
                                   f'{FOLD_NAME}/pretrained',
                    timer=timer,
                    ordinal_features=ordinal_features or None,
                )
            finally:
                writer.close()
            timer.end_phase('pretrain')

            seed_defaults = record_trial(
                results, hp_name, value, best_test_losses, sweep,
                seed_defaults, evaluation_fp)

            del electra, writer
            gc.collect()
            torch.cuda.empty_cache()

    best = {'experiment': EXPERIMENT_NAME, 'fold': FOLD_NAME,
            'task': 'pretrain', 'best_hyperparameters': {}}
    for hp_name in sweep.to_tune:
        trials = results.get(hp_name, {})
        if not trials:
            continue
        value, losses = min(trials.items(),
                            key=lambda kv: kv[1]['Optimization_Loss'])
        best['best_hyperparameters'][hp_name] = {
            'value': value, 'Optimization_Loss': losses['Optimization_Loss']}

    best_fp = os.path.join(evaluation_dir, 'best_hyperparameters.yaml')
    with open(best_fp, 'w') as f_out:
        yaml.dump(best, f_out)
    print(f'Best hyperparameters written to: {best_fp}')

    timer.end_fold()
    timer.print_final_summary()


if __name__ == '__main__':
    main()
