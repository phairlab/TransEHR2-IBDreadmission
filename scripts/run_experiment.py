"""Run one experiment: pretrain, then fit the competing-risks head, per fold.

One process, one GPU. Parallelism across folds is parallelism across
*processes* -- the folds share nothing but the extracted arrays, which are
memory-mapped read-only, so five shells or one SLURM array does the job
that FSDP used to:

    for f in 0 1 2 3 4; do
        CUDA_VISIBLE_DEVICES=$f python scripts/run_experiment.py \\
            dataset.yaml experiment.yaml --folds fold$f &
    done; wait

Usage:
    python scripts/run_experiment.py <dataset_config> <experiment_config> \\
        [--folds fold0 fold1] [--device cuda:0] [--force_pretrain] \\
        [--num_workers 0] [--mem_test_mode]
"""

import argparse
import gc
import os
import torch
import yaml

from torch.utils.tensorboard import SummaryWriter

import _path  # noqa: F401  (repository root on sys.path)

from TransEHR2.cli import TASK, get_fold_names, resolve_device
from TransEHR2.data.preprocessing import (
    compute_static_feat_dims, lookup_feat_widths, prepare_dataloaders,
    value_encoder_dims
)
from TransEHR2.losses import DeepHitLoss
from TransEHR2.models import ELECTRA, MixedClassifier
from TransEHR2.modules import (
    DeepHitHead, EventDataEncoder, MaskedTokenDiscriminator,
    MaskedTokenGenerator, TransformerHawkesProcess, ValueDataEncoder
)
from TransEHR2.routines import (
    evaluate_finetuned_model, finetune_model, pretrain_model
)
from TransEHR2.survival import (
    DEFAULT_BRIER_INTEGRATION_DAYS, DEFAULT_CAUSES, DEFAULT_CUTS_DAYS,
    TimeGrid, format_days
)
from TransEHR2.utils import (
    convert_to_python_types, create_timer,
    format_finetuning_performance_table
)


def pretraining_is_done(pretrained_dir: str, expect_event_encoder: bool) -> bool:
    """Whether ``pretrained_dir`` holds what finetuning will ask for.

    The test is the encoder files, not "is there a .pt here": the finetuned
    model and the encoders are written into this same directory, so any
    looser test calls pretraining finished as soon as *anything* has been
    saved. Creates the directory when it does not exist, so the caller can
    write into it straight afterwards.
    """
    if not os.path.exists(pretrained_dir):
        os.makedirs(pretrained_dir, exist_ok=True)
        return False

    needed = ['value_encoder.pt']
    if expect_event_encoder:
        needed.append('event_encoder.pt')
    return all(os.path.exists(os.path.join(pretrained_dir, f))
               for f in needed)


def main():
    parser = argparse.ArgumentParser(
        description='Pretrain and fit the competing-risks model, per fold'
    )
    parser.add_argument(
        'dataset_config', type=str,
        help='YAML file that specifies parameters for the dataset'
    )
    parser.add_argument(
        'experiment_config', type=str,
        help='YAML file that specifies parameters for the experiment'
    )
    parser.add_argument(
        '--folds', type=str, nargs='+', default=None,
        help='Which folds to run. Default is every fold in DATA_DIR. Give '
             'one fold per process to spread a run across GPUs.'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device to train on, e.g. cuda:0. Default picks CUDA, then '
             'MPS, then CPU.'
    )
    parser.add_argument(
        '--force_pretrain', action='store_true',
        help='Pretrain even if pretrained weights are found'
    )
    parser.add_argument(
        '--num_workers', type=int, default=0,
        help='Worker processes for data loading. Default 0 (main process).'
    )
    parser.add_argument(
        '--mem_test_mode', action='store_true',
        help='Run two batches per phase and report peak memory. Useful for '
             'finding the batch size a card will take.'
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Training on {device}")

    with open(args.dataset_config, 'r') as f_in:
        dataset_config = yaml.safe_load(f_in)
    DATA_DIR = dataset_config['DATA_DIR']
    VARIABLE_PROPERTIES_PATH = dataset_config['VARIABLE_PROPERTIES_PATH']
    VALUED_FEATS = dataset_config['VALUED_FEATS']
    EVENT_FEATS = dataset_config['EVENT_FEATS']
    STATIC_FEATS = dataset_config['STATIC_FEATS']

    with open(args.experiment_config, 'r') as f_in:
        experiment_config = yaml.safe_load(f_in)
    EXPERIMENT_NAME = experiment_config['EXPERIMENT_NAME']
    BATCH_SIZE = experiment_config['BATCH_SIZE']
    PREDICT_INDICATORS = experiment_config['PREDICT_INDICATORS']
    GENERATOR_ENCODER_D_MODEL = experiment_config['GENERATOR_ENCODER_D_MODEL']
    GENERATOR_ENCODER_N_HEADS = experiment_config['GENERATOR_ENCODER_N_HEADS']
    GENERATOR_ENCODER_N_ENCODER_BLOCKS = experiment_config['GENERATOR_ENCODER_N_ENCODER_BLOCKS']
    GENERATOR_ENCODER_DIM_FEEDFORWARD = experiment_config['GENERATOR_ENCODER_DIM_FEEDFORWARD']
    GENERATOR_ENCODER_DROPOUT = experiment_config['GENERATOR_ENCODER_DROPOUT']
    GENERATOR_ENCODER_ACTIVATION = experiment_config['GENERATOR_ENCODER_ACTIVATION']
    GENERATOR_ENCODER_NORM = experiment_config['GENERATOR_ENCODER_NORM']
    GENERATOR_ENCODER_NORM_FIRST = experiment_config.get('GENERATOR_ENCODER_NORM_FIRST', True)
    DISCRIMINATOR_ENCODER_D_MODEL = experiment_config['DISCRIMINATOR_ENCODER_D_MODEL']
    DISCRIMINATOR_ENCODER_N_HEADS = experiment_config['DISCRIMINATOR_ENCODER_N_HEADS']
    DISCRIMINATOR_ENCODER_N_ENCODER_BLOCKS = experiment_config['DISCRIMINATOR_ENCODER_N_ENCODER_BLOCKS']
    DISCRIMINATOR_ENCODER_DIM_FEEDFORWARD = experiment_config['DISCRIMINATOR_ENCODER_DIM_FEEDFORWARD']
    DISCRIMINATOR_ENCODER_DROPOUT = experiment_config['DISCRIMINATOR_ENCODER_DROPOUT']
    DISCRIMINATOR_ENCODER_ACTIVATION = experiment_config['DISCRIMINATOR_ENCODER_ACTIVATION']
    DISCRIMINATOR_ENCODER_NORM = experiment_config['DISCRIMINATOR_ENCODER_NORM']
    DISCRIMINATOR_ENCODER_NORM_FIRST = experiment_config.get('DISCRIMINATOR_ENCODER_NORM_FIRST', True)
    THP_ENCODER_D_MODEL = experiment_config['THP_ENCODER_D_MODEL']
    THP_ENCODER_D_INNER = experiment_config['THP_ENCODER_D_INNER']
    THP_ENCODER_N_LAYERS = experiment_config['THP_ENCODER_N_LAYERS']
    THP_ENCODER_N_HEADS = experiment_config['THP_ENCODER_N_HEADS']
    THP_ENCODER_D_K = experiment_config['THP_ENCODER_D_K']
    THP_ENCODER_D_V = experiment_config['THP_ENCODER_D_V']
    THP_ENCODER_DROPOUT = experiment_config['THP_ENCODER_DROPOUT']
    THP_ENCODER_NORM_FIRST = experiment_config.get('THP_ENCODER_NORM_FIRST', True)
    GENERATOR_D_MODEL = experiment_config['GENERATOR_D_MODEL']
    GENERATOR_DIM_FEEDFORWARD = experiment_config['GENERATOR_DIM_FEEDFORWARD']
    DISCRIMINATOR_DIM_FEEDFORWARD = experiment_config['DISCRIMINATOR_DIM_FEEDFORWARD']
    PREDICTOR_AGGREGATION_METHOD = experiment_config['PREDICTOR_AGGREGATION_METHOD']
    MODEL_DIR = experiment_config['MODEL_DIR']
    PRETRAIN_LEARNING_RATE = experiment_config.get('PRETRAIN_LEARNING_RATE', 2e-3)

    # The decay factor was applied on a fixed multi-epoch cadence, so the same value means a
    # different schedule now that the scheduler steps every epoch. Refuse it rather than
    # silently reinterpret it -- an ignored key would leave the rate constant for the whole
    # run without saying so.
    stale_decay = [key for key in ('PRETRAIN_LEARNING_RATE_DECAY', 'FINETUNE_LEARNING_RATE_DECAY')
                   if key in experiment_config]
    if stale_decay:
        raise ValueError(
            f"{args.experiment_config} carries {', '.join(stale_decay)}, which no longer "
            f"has an effect. The learning rate schedule is set by PRETRAIN_LR_HALF_LIFE and "
            f"FINETUNE_LR_HALF_LIFE, in epochs, applied as lr(e) = lr0 * 0.5 ** (e / H). A "
            f"factor g formerly applied every I epochs is a half-life of I * ln(0.5) / ln(g)."
        )

    PRETRAIN_LR_HALF_LIFE = experiment_config.get('PRETRAIN_LR_HALF_LIFE', None)
    PRETRAIN_TOTAL_EPOCH = experiment_config.get('PRETRAIN_TOTAL_EPOCH', 1000)
    DISC_LOSS_WEIGHT = experiment_config.get('DISC_LOSS_WEIGHT', 1.0)
    USE_THP = experiment_config.get('USE_THP', True)
    THP_LOSS_NLL_WEIGHT = experiment_config.get('THP_LOSS_NLL_WEIGHT', 1e-2)
    THP_LOSS_MC_SAMPLES = experiment_config.get('THP_LOSS_MC_SAMPLES', 100)
    USE_THP_PRED_LOSS = experiment_config.get('USE_THP_PRED_LOSS', True)
    THP_PRED_LOSS_TYPE_WT = experiment_config.get('THP_PRED_LOSS_TYPE_WT', 1.0)
    THP_PRED_LOSS_TIME_WT = experiment_config.get('THP_PRED_LOSS_TIME_WT', 1e-6)
    RECORD_MASK_RATIO = experiment_config.get('RECORD_MASK_RATIO', 0.15)
    OBS_UNOBS_SAMPLE_RATIO = experiment_config.get('OBS_UNOBS_SAMPLE_RATIO', 5.0)
    CMPNT_MASK_RATIO = experiment_config.get('CMPNT_MASK_RATIO', 0.25)
    FINETUNE_LEARNING_RATE = experiment_config.get('FINETUNE_LEARNING_RATE', 2e-4)
    FINETUNE_TOTAL_EPOCH = experiment_config.get('FINETUNE_TOTAL_EPOCH', 500)
    FINETUNE_LR_HALF_LIFE = experiment_config.get('FINETUNE_LR_HALF_LIFE', None)
    FINETUNE_SELECTION_METRIC = experiment_config.get(
        'FINETUNE_SELECTION_METRIC', 'brier')

    # The competing-risks head and its time grid.
    TIME_GRID_CUTS_DAYS = experiment_config.get(
        'TIME_GRID_CUTS_DAYS', list(DEFAULT_CUTS_DAYS))
    MODELLED_CAUSES = experiment_config.get(
        'MODELLED_CAUSES', list(DEFAULT_CAUSES))
    BRIER_INTEGRATION_DAYS = experiment_config.get(
        'BRIER_INTEGRATION_DAYS', DEFAULT_BRIER_INTEGRATION_DAYS)
    DEEPHIT_RANK_WEIGHT = experiment_config.get('DEEPHIT_RANK_WEIGHT', 1.0)
    DEEPHIT_SIGMA = experiment_config.get('DEEPHIT_SIGMA', 0.1)
    DEEPHIT_CAUSE_WEIGHTS = experiment_config.get('DEEPHIT_CAUSE_WEIGHTS', None)
    DEEPHIT_HEAD_D_SHARED = experiment_config.get('DEEPHIT_HEAD_D_SHARED', [256, 128])
    DEEPHIT_HEAD_D_CAUSE = experiment_config.get('DEEPHIT_HEAD_D_CAUSE', [128, 64])
    DEEPHIT_HEAD_DROPOUT = experiment_config.get('DEEPHIT_HEAD_DROPOUT', 0.1)

    grid = TimeGrid(TIME_GRID_CUTS_DAYS, MODELLED_CAUSES,
                    BRIER_INTEGRATION_DAYS)
    print(f"Time grid: {grid.n_bins} bins, {', '.join(grid.labels())}")
    print(f"Integrated Brier score: discharge to "
          f"{format_days(grid.brier_integration_days)}, over a monotone "
          f"cubic interpolation of the cumulative incidence")
    print(f"Modelled causes: {', '.join(grid.cause_names)} "
          f"(any EVENT_TYPE not listed is censoring)")

    timer = create_timer(
        results_dir=f'./log/timing/{EXPERIMENT_NAME}',
        experiment_name=EXPERIMENT_NAME
    )
    timer.start_total_timing()

    # Get number of valued features and their sizes, class counts of categorical features,
    # number of event types. These are needed as arguments for model initialization.
    with open(VARIABLE_PROPERTIES_PATH, 'r') as f_in:
        variable_properties = yaml.safe_load(f_in)
    # Width of the stored static_data array. NOT len(STATIC_FEATS): a
    # categorical static is one-hot and occupies `size` columns, so count and
    # width diverge the moment any static has size > 1. Derived from the same
    # helper the extraction uses, so the two cannot drift apart.
    static_dim = sum(
        compute_static_feat_dims(variable_properties, STATIC_FEATS)
    )
    numeric_feat_dims = []    # The dimension of each numeric feature
    categorical_class_cnts = []   # Number of classes for each categorical feature
    ordinal_features = []     # Number of levels for each ordinal feature
    multilabel_class_cnts = []    # Number of classes for each multilabel feature
    for feature in VALUED_FEATS:
        if variable_properties[feature]['type'] == 'numeric':
            numeric_feat_dims.append(variable_properties[feature]['size'])
        elif variable_properties[feature]['type'] == 'categorical':
            categorical_class_cnts.append(
                len(variable_properties[feature]['category_map'])
            )
        elif variable_properties[feature]['type'] == 'ordinal':
            ordinal_features.append(
                len(variable_properties[feature]['category_map'])
            )
        elif variable_properties[feature]['type'] == 'multilabel':
            multilabel_class_cnts.append(
                variable_properties[feature]['size']
            )

    fold_name_list = args.folds or get_fold_names(DATA_DIR)
    if not fold_name_list:
        raise FileNotFoundError(f'No fold directories found in {DATA_DIR}')

    # The lookup family is used whenever the extraction carries it: the
    # dataset config's TEXT_FEATS and DRUG_FEATS decide that, and there
    # is no experiment-level switch over them. Sizing therefore reads
    # the extracted root rather than the config's feature lists --
    # DRUG_FEATS has no VALUED_FEATS entry and no timeseries.csv column,
    # so a width counted over text alone is short by one feature and by
    # ClinVec's embedding width, and the first forward dies in the
    # indicator projection.
    lookup_dims = lookup_feat_widths(os.path.join(DATA_DIR, 'extracted'))
    use_lookup = bool(lookup_dims)
    n_val_feats, tot_val_feat_dim = value_encoder_dims(
        variable_properties, VALUED_FEATS, lookup_dims
    )
    n_event_types = len(EVENT_FEATS)

    def build_value_encoder(d_model, n_heads, n_blocks, dim_ff, dropout,
                            activation, norm, norm_first):
        return ValueDataEncoder(
            n_features=n_val_feats, feat_dim=tot_val_feat_dim, d_model=d_model,
            n_heads=n_heads, n_encoder_blocks=n_blocks, dim_feedforward=dim_ff,
            dropout=dropout, activation=activation, norm=norm,
            normalize_before=norm_first)

    def build_event_encoder():
        return EventDataEncoder(
            num_types=n_event_types, d_model=THP_ENCODER_D_MODEL,
            d_inner=THP_ENCODER_D_INNER, n_layers=THP_ENCODER_N_LAYERS,
            n_head=THP_ENCODER_N_HEADS, d_k=THP_ENCODER_D_K,
            d_v=THP_ENCODER_D_V, dropout=THP_ENCODER_DROPOUT,
            normalize_before=THP_ENCODER_NORM_FIRST)

    def build_predictor():
        """A fresh competing-risks model with untrained encoders."""
        return MixedClassifier(
            event_encoder=build_event_encoder(),
            val_encoder=build_value_encoder(
                DISCRIMINATOR_ENCODER_D_MODEL, DISCRIMINATOR_ENCODER_N_HEADS,
                DISCRIMINATOR_ENCODER_N_ENCODER_BLOCKS,
                DISCRIMINATOR_ENCODER_DIM_FEEDFORWARD,
                DISCRIMINATOR_ENCODER_DROPOUT, DISCRIMINATOR_ENCODER_ACTIVATION,
                DISCRIMINATOR_ENCODER_NORM, DISCRIMINATOR_ENCODER_NORM_FIRST),
            d_event_enc=THP_ENCODER_D_MODEL,
            d_val_enc=DISCRIMINATOR_ENCODER_D_MODEL,
            d_statics=static_dim,
            num_classes=grid.n_causes * grid.n_bins,
            aggr=PREDICTOR_AGGREGATION_METHOD,
            use_lookup=use_lookup,
            head=DeepHitHead(
                d_in=MixedClassifier.encoding_width(
                    THP_ENCODER_D_MODEL, DISCRIMINATOR_ENCODER_D_MODEL,
                    static_dim),
                n_causes=grid.n_causes, n_bins=grid.n_bins,
                d_shared=DEEPHIT_HEAD_D_SHARED, d_cause=DEEPHIT_HEAD_D_CAUSE,
                dropout=DEEPHIT_HEAD_DROPOUT),
        )

    for fold_name in fold_name_list:
        timer.start_fold(fold_name)

        dataloader_list = prepare_dataloaders(
            DATA_DIR,
            fold_name,
            BATCH_SIZE,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2 if args.num_workers > 0 else 1,
        )
        if len(dataloader_list) == 3:
            train_loader, val_loader, test_loader = dataloader_list
        else:
            raise RuntimeError(
                f'{fold_name} has no validation split. Both finetuning and '
                f'pretraining select their epoch on it; rebuild the folds '
                f"with IBDdataprep's split.py."
            )

        # ---------------------------------------------------------- pretrain
        model_save_dir = f'{MODEL_DIR}/{EXPERIMENT_NAME}/{fold_name}/pretrained'

        if not args.force_pretrain and pretraining_is_done(model_save_dir,
                                                           USE_THP):
            print(f"\nPretrained encoders found in {model_save_dir}, "
                  f"skipping pretraining.\n")
        else:
            print("\nStarting pretraining from scratch.\n")

            generator = MaskedTokenGenerator(
                encoder=build_value_encoder(
                    GENERATOR_ENCODER_D_MODEL, GENERATOR_ENCODER_N_HEADS,
                    GENERATOR_ENCODER_N_ENCODER_BLOCKS,
                    GENERATOR_ENCODER_DIM_FEEDFORWARD,
                    GENERATOR_ENCODER_DROPOUT, GENERATOR_ENCODER_ACTIVATION,
                    GENERATOR_ENCODER_NORM, GENERATOR_ENCODER_NORM_FIRST),
                d_model=GENERATOR_D_MODEL,
                numeric_dims=numeric_feat_dims,
                categorical_classes=categorical_class_cnts,
                ordinal_features=ordinal_features if ordinal_features else None,
                multilabel_classes=multilabel_class_cnts if multilabel_class_cnts else None,
                lookup_dims=lookup_dims,
                predict_indicators=PREDICT_INDICATORS,
                dim_feedforward=GENERATOR_DIM_FEEDFORWARD
            )
            discriminator = MaskedTokenDiscriminator(
                encoder=build_value_encoder(
                    DISCRIMINATOR_ENCODER_D_MODEL, DISCRIMINATOR_ENCODER_N_HEADS,
                    DISCRIMINATOR_ENCODER_N_ENCODER_BLOCKS,
                    DISCRIMINATOR_ENCODER_DIM_FEEDFORWARD,
                    DISCRIMINATOR_ENCODER_DROPOUT,
                    DISCRIMINATOR_ENCODER_ACTIVATION,
                    DISCRIMINATOR_ENCODER_NORM,
                    DISCRIMINATOR_ENCODER_NORM_FIRST),
                d_model=DISCRIMINATOR_ENCODER_D_MODEL,
                n_numeric_features=len(numeric_feat_dims),
                n_categorical_features=len(categorical_class_cnts),
                n_ordinal_features=len(ordinal_features),
                n_multilabel_features=len(multilabel_class_cnts),
                n_lookup_features=len(lookup_dims),
                n_static_features=static_dim,
                dim_feedforward=DISCRIMINATOR_DIM_FEEDFORWARD
            )
            # With the Hawkes process switched off there is no event branch in
            # pretraining at all, so its encoder is neither built nor saved and
            # the predictor below starts its own from scratch.
            hawkes = TransformerHawkesProcess(
                encoder=build_event_encoder(), num_types=n_event_types
            ) if USE_THP else None

            electra = ELECTRA(
                generator=generator,
                discriminator=discriminator,
                hawkes=hawkes,
                use_lookup=use_lookup,
            )

            log_dir = f'./log/{EXPERIMENT_NAME}/{fold_name}/pretrained'
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir)

            timer.start_phase('pretrain')
            pretrain_model(
                model=electra,
                save_path=f'{model_save_dir}/pretrained.pt',
                loaders=dataloader_list,
                writer=writer,
                learning_rate=PRETRAIN_LEARNING_RATE,
                device=device,
                lr_half_life=PRETRAIN_LR_HALF_LIFE,
                total_epoch=PRETRAIN_TOTAL_EPOCH,
                disc_loss_weight=DISC_LOSS_WEIGHT,
                use_thp=USE_THP,
                thp_loss_nll_weight=THP_LOSS_NLL_WEIGHT,
                thp_loss_mc_samples=THP_LOSS_MC_SAMPLES,
                use_thp_pred_loss=USE_THP_PRED_LOSS,
                thp_pred_loss_type_wt=THP_PRED_LOSS_TYPE_WT,
                thp_pred_loss_time_wt=THP_PRED_LOSS_TIME_WT,
                record_mask_ratio=RECORD_MASK_RATIO,
                obs_unobs_sample_ratio=OBS_UNOBS_SAMPLE_RATIO,
                cmpnt_mask_ratio=CMPNT_MASK_RATIO,
                checkpoint_dir=f'./checkpoints/{EXPERIMENT_NAME}/{fold_name}/pretrained',
                timer=timer,
                mem_test_mode=args.mem_test_mode,
                ordinal_features=ordinal_features if ordinal_features else None
            )
            timer.end_phase('pretrain')

            writer.close()
            del electra, generator, discriminator, hawkes, writer
            gc.collect()
            torch.cuda.empty_cache()

        # ---------------------------------------------------------- finetune
        finetuned_model_path = f'{model_save_dir}/finetuned_{TASK}.pt'
        evaluation_dir = (f'{MODEL_DIR}/{EXPERIMENT_NAME}/{fold_name}/{TASK}'
                          f'/evaluation')
        evaluation_file = f'{evaluation_dir}/evaluation_{TASK}.yaml'

        if os.path.exists(evaluation_file):
            print(f"\nEvaluation for {fold_name} already completed, "
                  f"skipping.\n")
            timer.end_fold()
            continue

        best_train_scores = None
        best_val_scores = None

        if os.path.exists(finetuned_model_path):
            print("\nFinetuned model found, skipping finetuning.\n")
        else:
            predictor = load_pretrained_encoders(
                build_predictor(), model_save_dir, expect_event_encoder=USE_THP)

            log_dir = f'./log/{EXPERIMENT_NAME}/{fold_name}/finetuned_{TASK}'
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir)

            timer.start_phase('finetune')
            best_train_scores, best_val_scores = finetune_model(
                model=predictor,
                save_path=finetuned_model_path,
                loaders=dataloader_list,
                grid=grid,
                writer=writer,
                learning_rate=FINETUNE_LEARNING_RATE,
                device=device,
                rank_weight=DEEPHIT_RANK_WEIGHT,
                sigma=DEEPHIT_SIGMA,
                cause_weights=DEEPHIT_CAUSE_WEIGHTS,
                selection_metric=FINETUNE_SELECTION_METRIC,
                lr_half_life=FINETUNE_LR_HALF_LIFE,
                total_epoch=FINETUNE_TOTAL_EPOCH,
                checkpoint_dir=f'./checkpoints/{EXPERIMENT_NAME}/{fold_name}/finetuned',
                timer=timer,
                mem_test_mode=args.mem_test_mode
            )
            timer.end_phase('finetune')

            writer.close()
            del predictor, writer
            gc.collect()
            torch.cuda.empty_cache()

        # ---------------------------------------------------------- evaluate
        predictor = build_predictor()
        # strict, deliberately. The saved head's width is n_causes * n_bins,
        # so a TIME_GRID_CUTS_DAYS edited between finetuning and scoring
        # shows up here as a shape mismatch. Loading loosely would leave a
        # randomly initialized head and report its scores.
        predictor.load_state_dict(
            torch.load(finetuned_model_path, map_location='cpu',
                       weights_only=False))
        predictor = predictor.to(device)

        best_test_scores = evaluate_finetuned_model(
            model=predictor,
            loader=test_loader,
            grid=grid,
            loss_fn=DeepHitLoss(
                n_causes=grid.n_causes, n_bins=grid.n_bins,
                sigma=DEEPHIT_SIGMA, rank_weight=DEEPHIT_RANK_WEIGHT,
                cause_weights=DEEPHIT_CAUSE_WEIGHTS).to(device),
            device=device,
            prefix='test',
            mem_test_mode=args.mem_test_mode
        )

        del predictor
        gc.collect()
        torch.cuda.empty_cache()

        os.makedirs(evaluation_dir, exist_ok=True)
        evaluation_data = {
            'task': TASK,
            'fold': fold_name,
            'experiment': EXPERIMENT_NAME,
            'time_grid_cuts_days': list(map(float, grid.cuts)),
            'modelled_causes': list(grid.cause_names),
            'brier_integration_days': grid.brier_integration_days,
            'use_thp': USE_THP,
            'selection_metric': FINETUNE_SELECTION_METRIC,
            'train_scores': convert_to_python_types(best_train_scores or {}),
            'validation_scores': convert_to_python_types(best_val_scores or {}),
            'test_scores': convert_to_python_types(best_test_scores),
        }
        with open(evaluation_file, 'w') as f_out:
            yaml.dump(evaluation_data, f_out, default_flow_style=False,
                      indent=2)
        print(f"Saved evaluation results to {evaluation_file}\n")

        print("\n" + format_finetuning_performance_table(
            train_scores=best_train_scores,
            val_scores=best_val_scores,
            test_scores=best_test_scores,
            title=f'{fold_name}: Competing-Risks Model Performance'
        ) + "\n")

        del dataloader_list, train_loader, val_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()

        timer.end_fold()

    timer.print_final_summary()


def load_pretrained_encoders(
    predictor: MixedClassifier,
    pretrained_dir: str,
    expect_event_encoder: bool = True
) -> MixedClassifier:
    """Load the pretrained encoder weights into a fresh predictor.

    The value encoder is required: without it finetuning starts from
    nothing and the pretraining stage was pointless. The event encoder is
    only there when the Hawkes process ran, so its absence is expected
    under ``USE_THP: False`` and is reported rather than raised.
    """
    value_path = os.path.join(pretrained_dir, 'value_encoder.pt')
    event_path = os.path.join(pretrained_dir, 'event_encoder.pt')

    if not os.path.exists(value_path):
        raise FileNotFoundError(
            f"Encoder weights not found in {pretrained_dir}. Expected "
            f"value_encoder.pt. Make sure pretraining completed."
        )
    predictor.val_encoder.load_state_dict(
        torch.load(value_path, map_location='cpu', weights_only=False))
    print(f"\nLoaded value encoder weights from {pretrained_dir}")

    if os.path.exists(event_path):
        predictor.event_encoder.load_state_dict(
            torch.load(event_path, map_location='cpu', weights_only=False))
        print(f"Loaded event encoder weights from {pretrained_dir}\n")
    elif expect_event_encoder:
        raise FileNotFoundError(
            f"event_encoder.pt not found in {pretrained_dir}, but this run "
            f"pretrained with USE_THP: True. Pretraining did not finish, or "
            f"the config changed between the two stages."
        )
    else:
        print("No pretrained event encoder (USE_THP is off); the event "
              "encoder will train from scratch during finetuning.\n")

    return predictor


if __name__ == '__main__':
    main()
