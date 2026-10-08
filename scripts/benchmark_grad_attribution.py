#!/usr/bin/env python
"""What stage 2 of token attribution costs: wall time and peak memory, per batch and per sequence.

Token-level attribution re-runs the text encoder with a local backward, for the records being
explained only (``TransEHR2/attribution.py``). The open question is whether that fits the
single-GPU arrangement the model was restructured to get, so this measures the thing that
decides it: a forward plus a backward to the input embedding layer, at the batch sizes and
sequence lengths the study actually uses.

Two modes are timed against each other:

``embed``       forward under ``no_grad``, which is what ``scripts/embed.py`` already does once
                per unique string. The reference.
``attribute``   forward with a graph plus ``autograd.grad`` to ``inputs_embeds``. The cost of
                the strategy.

Their ratio is the headline: it says what attribution costs relative to a pass the study has
already paid for and timed.

    python scripts/benchmark_grad_attribution.py --batch-sizes 1,4,8,16
    python scripts/benchmark_grad_attribution.py --tokens data/lookup_tables/text_tokens.npy
    python scripts/benchmark_grad_attribution.py --checkpointing --batch-sizes 16,32

What this does not measure
--------------------------

Stage 1 -- TransEHR2's own forward and backward over the stored lookup rows -- is unchanged by
any of this and is already timed by every training run, so it is not repeated here. Peak memory
below is the encoder's alone; the two are co-resident only during an explanation pass, and the
encoder is the larger of the two.

Real data
---------

``--tokens`` reads ``text_tokens.npy``, a derivative of the cohort, and is therefore **run by
the operator, not by an assistant**. It is used for one thing: drawing a realistic distribution
of sequence lengths. Nothing from the file is printed -- the output is timings, byte counts and
length percentiles. Without it, lengths are synthetic and the timings are a worst case, since
every sequence runs at full width.
"""

import argparse
import statistics
import sys
import time

import _path  # noqa: F401  (repository root on sys.path)

import numpy as np
import torch

from TransEHR2.attribution import token_attributions
from TransEHR2.constants import LLM_NAME, MAX_TOKEN_LENGTH, TEXT_POOLING


def pick_device(requested):
    if requested != 'auto':
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def reset_peak_memory(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mib(device):
    """Peak allocated bytes, or None where the backend does not report one."""
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    if device.type == 'mps':
        # MPS has no peak counter; this is the live figure, which undercounts a
        # transient. Treated as indicative only -- the cluster run is the measurement.
        return torch.mps.current_allocated_memory() / 1024 ** 2
    return None


def sequence_lengths(tokens_path, n, max_length, pad_id, rng):
    """A sample of real sequence lengths, or full-width ones when no table is given."""
    if tokens_path is None:
        return np.full(n, max_length, dtype=np.int64)
    table = np.load(tokens_path, mmap_mode='r')
    rows = rng.choice(table.shape[0], size=min(n, table.shape[0]), replace=False)
    lengths = np.array([int((table[r] != pad_id).sum()) for r in sorted(rows)])
    if lengths.size < n:
        lengths = np.resize(lengths, n)
    return np.clip(lengths, 1, max_length)


def make_batch(lengths, vocab_size, width, device, rng):
    """Random ids at the given lengths, right-padded to ``width``.

    The ids are random because timing depends on shape, not on content: an encoder's cost is
    the same whatever the tokens say. Using real ids would put cohort data in the benchmark's
    memory for no gain.
    """
    ids = torch.from_numpy(
        rng.integers(1, vocab_size, size=(len(lengths), width))).long()
    mask = torch.zeros(len(lengths), width, dtype=torch.long)
    for row, length in enumerate(lengths):
        mask[row, :length] = 1
        ids[row, length:] = 0
    return ids.to(device), mask.to(device)


def time_mode(llm, mode, ids, mask, width, repeats, device):
    """Median seconds per batch over ``repeats``, after one warmup."""
    upstream = torch.randn(ids.size(0), llm.model.config.hidden_size, device=device)
    timings = []
    for run in range(repeats + 1):
        synchronize(device)
        if run == 1:
            reset_peak_memory(device)       # after warmup, so allocator caching settles
        start = time.perf_counter()
        if mode == 'embed':
            with torch.no_grad():
                llm(ids, attention_mask=mask)
        else:
            token_attributions(llm, ids, mask, upstream)
        synchronize(device)
        if run:
            timings.append(time.perf_counter() - start)
    return statistics.median(timings), peak_memory_mib(device)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Time and measure stage 2 of token attribution")
    parser.add_argument('--model', default=LLM_NAME,
                        help="encoder path or hub id; defaults to constants.LLM_NAME")
    parser.add_argument('--pooling', default=TEXT_POOLING, choices=('cls', 'mean'))
    parser.add_argument('--batch-sizes', default='1,4,8,16',
                        help="comma-separated")
    parser.add_argument('--width', type=int, default=MAX_TOKEN_LENGTH,
                        help="padded sequence width; the second axis of text_tokens.npy")
    parser.add_argument('--tokens', default=None,
                        help="text_tokens.npy, for a realistic length distribution. "
                             "Operator only -- see the module docstring")
    parser.add_argument('--pad-id', type=int, default=0,
                        help="pad token id in --tokens, from extracted/metadata.pkl")
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--checkpointing', action='store_true',
                        help="gradient checkpointing: less activation memory, more compute")
    parser.add_argument('--dtype', default='bfloat16',
                        choices=('bfloat16', 'float16', 'float32'))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--project', type=int, default=None,
                        help="project the attribute rate onto this many sequences")
    parser.add_argument('--tsr', default=None, metavar='T,N',
                        help="timesteps and features of one explained episode, for the TSR "
                             "budget section")
    parser.add_argument('--stage1-ms', type=float, default=None,
                        help="measured ms for one TransEHR2 forward+backward over stored "
                             "lookup rows; TSR's passes are all of this kind")
    parser.add_argument('--gate-quantile', type=float, default=0.0,
                        help="TSR's time-relevance gate; the fraction of timesteps that get a "
                             "feature profile is 1 minus this, and it sets the pass count. 0, "
                             "the default, measures every cell at 1 + T + T*N")
    parser.add_argument('--render-cells', type=int, default=None,
                        help="text cells per episode whose tokens are actually rendered; "
                             "defaults to the timestep count, i.e. all of them")
    parser.add_argument('--ig-steps', default='0', metavar='M[,M...]',
                        help="Use integrated gradients as TSR's R(.) over this many path "
                             'points instead of grad x input, which is 0. Multiplies every '
                             'pass by it. A list reports one episode table per setting, '
                             'against the same model and batch, so the methods are comparable '
                             'rather than two jobs apart.')
    parser.add_argument('--density', type=float, default=1.0,
                        help='Fraction of (timestep, feature) cells the synthetic episode '
                             'observes. TSR skips the rest, since deleting what is not there '
                             'moves nothing, so this is the dominant term in the real cost. '
                             '1.0, the default, is the worst case and not a realistic one.')
    parser.add_argument('--steps', default=None, metavar='T[,T...]',
                        help='Episode lengths to measure, overriding the T in --stage1. '
                             'MAX_EPISODE_LEN_STEPS is the whole episode; a shorter window '
                             'truncates it, and T enters the dominant T*N term.')
    parser.add_argument('--dataset-config', default=None, metavar='PATH',
                        help='Size the stage-1 model from this dataset config and its '
                             'VARIABLE_PROPERTIES_PATH, so N and every feature width are the '
                             "study's rather than --feat-width repeated. Needs both files, "
                             'which live on the cluster.')
    parser.add_argument('--stage1', default=None, metavar='T,N',
                        help="time a TransEHR2 forward+backward at this shape, against batch "
                             "size: the pass TSR spends all its time in")
    parser.add_argument('--chunk', type=int, default=64,
                        help='How many TSR perturbations share one forward and backward when '
                             'the episode is measured end to end.')
    parser.add_argument('--stage1-batches', default='1,8,32',
                        help="batch sweep for --stage1")
    parser.add_argument('--encoder-blocks', type=int, default=1,
                        help="Encoder blocks in the value encoder, i.e. the experiment "
                             "config's DISCRIMINATOR_ENCODER_N_ENCODER_BLOCKS. The classifier "
                             'attribution runs over is built from the discriminator settings, '
                             "so the generator's count does not enter this cost.")
    parser.add_argument('--d-shared', default='256,128',
                        help="DeepHit head's shared stack widths, per layer "
                             '(DEEPHIT_HEAD_D_SHARED).')
    parser.add_argument('--d-cause', default='128,64',
                        help="DeepHit head's per-cause stack widths, per layer "
                             '(DEEPHIT_HEAD_D_CAUSE).')
    parser.add_argument('--d-model', type=int, default=256,
                        help="value encoder width; experiment2_text.yaml's")
    parser.add_argument('--feat-width', type=int, default=4,
                        help="per-feature value width for the synthetic --stage1 batch")
    parser.add_argument('--drug-width', type=int, default=128,
                        help="Width of a drug feature's pooled embedding, ClinVec's by "
                             'default. Only used with --dataset-config.')
    parser.add_argument('--text-width', type=int, default=1024,
                        help="lookup embedding width for the synthetic --stage1 batch")
    parser.add_argument('--skip-encoder', action='store_true',
                        help="skip the encoder table; --stage1 only")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)

    device = pick_device(args.device)
    rng = np.random.default_rng(args.seed)
    batch_sizes = [int(b) for b in args.batch_sizes.split(',')]

    if args.skip_encoder:
        if not args.stage1:
            parser.error('--skip-encoder leaves nothing to run without --stage1 T,N')
        report_stage1(args, device)
        return 0

    from TransEHR2.modules import GradientTraceableLLM
    llm = GradientTraceableLLM(
        args.model, max_length=args.width, pooling=args.pooling,
        use_gradient_checkpointing=args.checkpointing,
        dtype=getattr(torch, args.dtype), device_map=str(device))
    llm.model.eval()

    parameters = sum(p.numel() for p in llm.model.parameters())
    print(f"model        {args.model}")
    print(f"  parameters {parameters / 1e6:.0f} M   hidden "
          f"{llm.model.config.hidden_size}   pooling {args.pooling}")
    print(f"  device     {device}   dtype {args.dtype}   "
          f"checkpointing {args.checkpointing}")

    lengths = sequence_lengths(args.tokens, max(batch_sizes) * 4, args.width,
                               args.pad_id, rng)
    source = args.tokens or 'synthetic (full width)'
    print(f"  lengths    {source}: median {int(np.median(lengths))}, "
          f"p95 {int(np.percentile(lengths, 95))}, max {int(lengths.max())}")
    print()

    header = f"{'batch':>6}  {'mode':>10}  {'s/batch':>9}  {'ms/seq':>8}  {'seq/s':>8}  {'peak MiB':>9}"
    print(header)
    print('-' * len(header))

    rates = {}
    for batch_size in batch_sizes:
        for mode in ('embed', 'attribute'):
            chosen = rng.choice(lengths, size=batch_size, replace=False)
            ids, mask = make_batch(chosen, llm.model.config.vocab_size,
                                   args.width, device, rng)
            try:
                seconds, peak = time_mode(llm, mode, ids, mask, args.width,
                                          args.repeats, device)
            except torch.cuda.OutOfMemoryError:
                print(f"{batch_size:>6}  {mode:>10}  {'OOM':>9}")
                torch.cuda.empty_cache()
                continue
            rate = batch_size / seconds
            rates.setdefault(mode, {})[batch_size] = rate
            peak_text = f"{peak:9.0f}" if peak is not None else f"{'-':>9}"
            print(f"{batch_size:>6}  {mode:>10}  {seconds:9.4f}  "
                  f"{1000 * seconds / batch_size:8.2f}  {rate:8.1f}  {peak_text}")

    best = max(rates.get('attribute', {}).items(), key=lambda kv: kv[1], default=None)
    if best and rates.get('embed'):
        batch_size, rate = best
        reference = rates['embed'].get(batch_size)
        print()
        print(f"best attribute rate  {rate:.1f} seq/s at batch {batch_size}")
        if reference:
            print(f"  cost over a no-grad forward at the same batch: "
                  f"{reference / rate:.2f}x")
        if args.project:
            print(f"  {args.project} sequences: {args.project / rate / 60:.1f} min")
        if args.tsr:
            report_tsr_budget(args, rate)
    if args.stage1:
        report_stage1(args, device)
    return 0


def report_tsr_budget(args, attribute_rate):
    """What one explained EPISODE costs -- a whole patient timeline, not a patient-minute.

    T is that episode's timestep count and N its feature count, so this is the cost of
    explaining one patient end to end. A patient-minute is a single timestep within it, and
    is what ContrastiveBMMB calls a "record" -- the two senses collide, so neither word is
    used bare here.

    TSR's R(.) is a scalar per (feature, timestep), so every one of its passes is a TransEHR2
    forward and backward over the stored lookup rows -- the encoder is not in that graph. The
    encoder is run once per text cell whose tokens are rendered, after TSR has chosen them. The
    two costs are therefore reported separately, because they scale with different things and
    only one of them is what this script measures.

    The pass count follows the reference implementation, which recomputes the feature profile at
    every timestep that clears the gate: 1 + T + (1 - q)*T*N, not 1 + T + N.
    """
    steps, features = (int(v) for v in args.tsr.split(','))
    passing = round(steps * (1.0 - args.gate_quantile))
    passes = 1 + steps + passing * features
    rendered = args.render_cells if args.render_cells is not None else steps

    print()
    print(f"TSR budget for one episode (whole patient timeline) at T={steps}, N={features}, "
          f"gate q={args.gate_quantile}")
    print(f"  stage 1  {passes} passes (1 + T + {passing}*N), encoder not involved")
    if args.stage1_ms:
        stage1 = passes * args.stage1_ms / 1000
        print(f"           {stage1:.1f} s at {args.stage1_ms:.0f} ms/pass")
    else:
        print("           supply --stage1-ms to cost these; one TransEHR2 "
              "forward+backward each")
    stage2 = rendered / attribute_rate
    print(f"  stage 2  {rendered} text cells x 1 encoder pass = {stage2:.1f} s "
          f"at {attribute_rate:.1f} seq/s")
    if args.stage1_ms:
        print(f"  total    {passes * args.stage1_ms / 1000 + stage2:.1f} s per episode")



# ---------------------------------------------------------------------------
# Stage 1: the TransEHR2 pass TSR actually spends its time in
# ---------------------------------------------------------------------------

def dataset_widths(config_path, drug_width=128):
    """``(steps, per-feature value widths)`` for the real study, from its two config files.

    `run_experiment` sizes the value encoder as ``sum(variable_properties[f]['size'])`` over
    ``VALUED_FEATS``, plus one feature per member of the lookup family; this reads the same two
    numbers so the benchmarked shape cannot drift from the trained one. A categorical feature is
    one-hot and occupies ``size`` columns, so count and width diverge as soon as any feature has
    size > 1 -- which is why repeating a single ``--feat-width`` is a stand-in rather than a
    measurement.

    The widths are returned one per feature rather than summed because TSR deletes one feature
    at a time: the per-pass cost follows the total, but the per-pass *overhead* follows how many
    separate tensors there are.

    ``DRUG_FEATS`` is counted, at `drug_width` each. Extraction writes one indicator array per
    lookup type and `datasets.MixedDataset` collates every one of them, so a batch carries the
    drug column alongside the text one -- the data's feature count is what TSR perturbs, and
    leaving it out would understate N. It joins the list as an
    ordinary value column because stage 1 runs over *stored* lookup rows: the pooling that makes
    a lookup feature different has already happened, so for cost it is a column of that width.
    """
    import yaml

    with open(config_path) as handle:
        config = yaml.safe_load(handle)
    with open(config['VARIABLE_PROPERTIES_PATH']) as handle:
        properties = yaml.safe_load(handle)
    widths = [properties[name]['size'] for name in config['VALUED_FEATS']]
    widths += [drug_width] * len(config.get('DRUG_FEATS') or [])
    return config['MAX_EPISODE_LEN_STEPS'], widths


def build_stage1(steps, widths, d_model, text_width, device, n_causes, n_bins,
                 n_blocks=1, d_shared=(256, 128), d_cause=(128, 64)):
    """A MixedClassifier and a batch at the given shape, for timing only.

    Random weights, because a forward and backward cost the same whatever the weights say. The
    shape is what has to be real: `experiment2_text.yaml` sets d_model 256, two heads, one
    encoder block, so this is small enough that a batch-1 pass is probably latency-bound rather
    than compute-bound -- which is the whole question, since TSR's loops are parallel over
    perturbations and only a latency-bound pass has headroom to exploit.

    The event branch is omitted: TSR never perturbs it, so it adds a constant per-pass cost that
    would be the same for every variant. That makes this a lower bound on the real per-pass time.

    The head is the real `DeepHitHead` at `experiment2_text.yaml`'s widths, because the target
    being differentiated is read off its output: the attribution scalar is the log-odds of
    readmission by a horizon, not the training loss, so the pass has to produce a (cause, bin)
    block for `survival.cif_logit_target` to reduce.
    """
    from TransEHR2.models import MixedClassifier
    from TransEHR2.modules import DeepHitHead, EventDataEncoder, ValueDataEncoder

    features = len(widths) + 1                     # one lookup feature, the text one
    feat_dim = sum(widths) + text_width
    val_encoder = ValueDataEncoder(
        n_features=features, feat_dim=feat_dim, d_model=d_model, n_heads=2,
        n_encoder_blocks=n_blocks, dim_feedforward=d_model, dropout=0.0,
        norm='LayerNorm')
    width = MixedClassifier.encoding_width(d_event_enc=0, d_val_enc=d_model, d_statics=0)
    model = MixedClassifier(
        event_encoder=EventDataEncoder(num_types=1, d_model=16, d_inner=32, n_layers=1,
                                       n_head=2, d_k=8, d_v=8, dropout=0.0),
        val_encoder=val_encoder, d_event_enc=0, d_val_enc=d_model, d_statics=0,
        num_classes=n_causes * n_bins, aggr='mean', use_lookup=True,
        head=DeepHitHead(d_in=width, n_causes=n_causes, n_bins=n_bins,
                         d_shared=d_shared, d_cause=d_cause,
                         dropout=0.0)).to(device)
    model.eval()
    return model


def stage1_batch(batch_size, steps, widths, text_width, device, density=1.0):
    """A batch at the given shape, with `density` of its cells observed.

    An unobserved cell is zero in both the value and the indicator, which is how extraction
    leaves one and what makes TSR's skip exact. Density is the single largest term in the real
    cost -- a patient-minute carries a handful of a 148-feature set -- so a benchmark that
    leaves it at 1.0 measures a worst case no episode exhibits.
    """
    def observed(n_features):
        if density >= 1.0:
            return torch.ones(batch_size, steps, n_features, device=device)
        return (torch.rand(batch_size, steps, n_features, device=device)
                < density).float()

    numeric_inds = observed(len(widths))
    lookup_inds = observed(1)
    return {
        'val_data': {
            'times': torch.arange(steps, device=device).float().repeat(batch_size, 1),
            'masks': torch.ones(batch_size, steps, device=device),
            'numeric': {
                'indicators': numeric_inds,
                'values': [torch.randn(batch_size, steps, width, device=device)
                           * numeric_inds[..., feature:feature + 1]
                           for feature, width in enumerate(widths)],
            },
            'lookup': {
                'indicators': lookup_inds,
                'slot_values': [torch.randn(batch_size, steps, text_width, device=device)
                                * lookup_inds],
                'doses': [None], 'masks': [None],
            },
        }
    }


def report_stage1(args, device):
    """Time one TSR pass -- a TransEHR2 forward and backward -- against batch size.

    The number that decides whether integrated gradients is affordable. If ms/pass is flat in
    batch size the pass is latency-bound and TSR's perturbations can be batched almost for free;
    if ms/pass/item is flat instead, it is compute-bound and batching buys nothing.

    The scalar differentiated is `survival.cif_logit_target` -- the log-odds of unplanned
    readmission by one horizon -- so the measured pass is the one deployment would take. Each
    horizon needs its own backward, so the second table is what deciding not to pick a single
    horizon actually costs.

    Every shape in `--steps` is reported separately, because T enters the dominant T*N term and
    a truncated window is a real lever on the cost rather than a rounding of it.
    """
    from TransEHR2.survival import DEFAULT_CAUSES, DEFAULT_CUTS_DAYS

    declared_steps, features = (int(v) for v in args.stage1.split(','))
    n_causes, n_bins = len(DEFAULT_CAUSES), len(DEFAULT_CUTS_DAYS)
    cause = DEFAULT_CAUSES.index('readmission')

    if args.dataset_config:
        config_steps, widths = dataset_widths(args.dataset_config, args.drug_width)
        source = f"{args.dataset_config} ({len(widths)} valued and drug + 1 text)"
    else:
        config_steps, widths = declared_steps, [args.feat_width] * (features - 1)
        source = f"--feat-width {args.feat_width} repeated ({features} features)"

    if args.steps:
        sweep = [int(v) for v in args.steps.split(',')]
    elif args.dataset_config:
        sweep = [config_steps]
    else:
        sweep = [declared_steps]

    for steps in sweep:
        _report_one_shape(args, device, steps, widths, source, cause, n_causes,
                          n_bins, DEFAULT_CUTS_DAYS)


def _report_one_shape(args, device, steps, widths, source, cause, n_causes, n_bins,
                      cuts_days):
    """The two tables for one episode length: per-pass against batch size, then the episode."""
    from TransEHR2.survival import cif_logit_target
    from TransEHR2.tsr import feature_index, saliency

    features = len(widths) + 1
    d_shared = [int(v) for v in args.d_shared.split(',')]
    d_cause = [int(v) for v in args.d_cause.split(',')]
    model = build_stage1(steps, widths, args.d_model, args.text_width, device,
                         n_causes, n_bins, args.encoder_blocks, d_shared, d_cause)

    def measure(batch_size, horizon_bin):
        batch = stage1_batch(batch_size, steps, widths, args.text_width, device,
                             args.density)
        index = feature_index(batch)
        target = cif_logit_target(cause, horizon_bin, n_causes, n_bins)
        timings = []
        for run in range(args.repeats + 1):
            synchronize(device)
            if run == 1:
                reset_peak_memory(device)
            start = time.perf_counter()
            saliency(model, batch, target=target, timestep=0, index=index)
            synchronize(device)
            if run:
                timings.append(time.perf_counter() - start)
        return statistics.median(timings), peak_memory_mib(device)

    print()
    print(f"stage 1: TransEHR2 forward+backward, T={steps}, N={features}, "
          f"feat_dim={sum(widths) + args.text_width}, d_model={args.d_model}, "
          f"{args.encoder_blocks} encoder block(s), DeepHit head {n_causes}x{n_bins} "
          f"shared {args.d_shared} cause {args.d_cause}")
    print(f"widths: {source}")
    print(f"target: log-odds of readmission by {cuts_days[-1]:g} d (the last bin)")
    header = f"{'batch':>6}  {'ms/pass':>9}  {'ms/pass/item':>13}  {'peak MiB':>9}"
    print(header)
    print('-' * len(header))

    best_ms_item = None
    for batch_size in [int(b) for b in args.stage1_batches.split(',')]:
        seconds, peak = measure(batch_size, n_bins - 1)
        ms_item = 1000 * seconds / batch_size
        if best_ms_item is None or ms_item < best_ms_item:
            best_ms_item = ms_item
        peak_text = f"{peak:9.0f}" if peak is not None else f"{'-':>9}"
        print(f"{batch_size:>6}  {1000 * seconds:9.2f}  {ms_item:13.2f}  {peak_text}")

    for ig_steps in [int(v) for v in args.ig_steps.split(',')]:
        report_horizons(args, model, steps, widths, n_causes, n_bins, best_ms_item,
                        cuts_days, device, ig_steps)


def report_horizons(args, model, steps, widths, n_causes, n_bins, ms_item,
                    cuts_days, device, ig_steps):
    """Run TSR end to end, once per horizon, and time it.

    Measured rather than projected. The projection this replaces multiplied a counted number of
    perturbations by a measured per-item time, which assumed the tiling itself is free; it is
    not, so the two are printed side by side and the gap is what batching costs to set up.

    A horizon is a different scalar and so a different backward over the same forward. They are
    timed separately rather than assumed equal because the log-odds of an early bin reduces over
    fewer cells than a late one.

    Called once per `--ig-steps` setting against the same model and the same batch, so grad x
    input and integrated gradients differ in nothing but R(.) -- comparing across jobs would
    also be comparing across a fresh random model and whatever else the cluster was doing.
    """
    from TransEHR2.survival import cif_logit_target
    from TransEHR2.tsr import tsr_scores

    cause = 0
    features = len(widths) + 1
    batch = stage1_batch(1, steps, widths, args.text_width, device, args.density)

    method = (f"IG x{ig_steps}" if ig_steps else 'grad x input')
    dense = (ig_steps or 1) * (1 + steps + steps * features)
    print()
    print(f"measured episode, {method}, gate q={args.gate_quantile}, "
          f"chunk {args.chunk}, density {args.density:g}")
    header = (f"{'horizon':>10}  {'bin':>4}  {'passes':>7}  {'calls':>6}  "
              f"{'s/episode':>10}  {'peak MiB':>9}")
    print(header)
    print('-' * len(header))

    total = 0.0
    for horizon_bin in range(n_bins):
        target = cif_logit_target(cause, horizon_bin, n_causes, n_bins)
        synchronize(device)
        reset_peak_memory(device)
        start = time.perf_counter()
        result = tsr_scores(model, batch, target=target, quantile=args.gate_quantile,
                            chunk=args.chunk, ig_steps=ig_steps)
        synchronize(device)
        seconds = time.perf_counter() - start
        total += seconds
        # Only where the backend keeps a real high-water mark. MPS reports the live figure, and
        # every chunk has been freed by the time tsr_scores returns, so it would read near zero
        # and mean nothing -- unlike in the per-pass table, where the batch is still alive.
        peak = peak_memory_mib(device) if device.type == 'cuda' else None
        peak_text = f"{peak:9.0f}" if peak is not None else f"{'-':>9}"
        calls = result['calls']
        passes_taken = result['passes']
        days = cuts_days[horizon_bin]
        label = f"{days:g} d" if days < 365 else f"{days / 365.25:.3g} y"
        print(f"{label:>10}  {horizon_bin:>4}  {result['passes']:>7}  {calls:>6}  "
              f"{seconds:10.1f}  {peak_text}")

    print()
    print(f"TSR for one episode at T={steps}, N={features}, encoder not involved")
    print(f"  {dense} passes if every cell were observed; "
          f"{100 * (1 - passes_taken / dense):.0f}% skipped as empty")
    print(f"  one horizon     {total / n_bins:.1f} s measured, "
          f"{passes_taken * ms_item / 1000:.1f} s projected from the passes actually taken "
          f"at {ms_item:.2f} ms/pass/item")
    print(f"  all {n_bins} horizons  {total:.1f} s measured")


if __name__ == '__main__':
    sys.exit(main())
