#!/bin/bash
#SBATCH --job-name=diagnose
#SBATCH --partition=gpu-h200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --gpus=h200:1
#SBATCH --nodes=1
#SBATCH --time=0-02:30:00
#SBATCH --output=log/diagnose_%j.out
#SBATCH --error=log/diagnose_%j.err
#SBATCH --qos=normal

#SBATCH --mail-user=pr3@ualberta.ca
#SBATCH --mail-type=END,FAIL

# Everything worth measuring about where a pretraining step's time goes, in one
# job, against the re-extracted cohort.
#
#     sbatch SLURM/slurm_diagnostics.sh                 # from the project root
#     sbatch TransEHR2/SLURM/slurm_diagnostics.sh       # the tracked copy
#     BATCH_SIZE=200 WORKERS=0,4,8,16,24 sbatch SLURM/slurm_diagnostics.sh
#
# On the performance-dx branch the normally untracked SLURM scripts are tracked,
# so they arrive by pull rather than by scp. That branch is a dead end: nothing
# here is meant to reach main.
#
# ONE GPU. Nothing here is distributed; the GPU is only needed for the last two
# sections, and the first three are CPU and I/O bound by construction.
#
# RUN IT ALONE. The loader sections measure storage throughput, so a concurrent
# job on the same node reading the same filesystem makes every number a
# measurement of the contention rather than of the pipeline. That is also the
# confound to rule out in the 1-vs-2-block head-to-head, where both arms ran at
# once and came back equal.
#
# Sections, in order:
#
#   0. warm     the whole cohort read once, so the sweeps below are comparable
#   1. raw      episode rows straight off the memmaps -- a sample, warm
#   2. loader   the real prepare_dataloaders path, swept over worker counts
#   3. device   host-to-device copy for one batch
#   4. padding  how much of what was read is padding, and what a shorter
#               extraction would win
#   5. stage 1  the GPU compute floor at 1 and at 2 encoder blocks, data
#               resident, which is the block question with the I/O removed
#   6. phase    the real pretraining step attributed phase by phase, at 0 and
#               at 8 dataloader workers -- the only section that is not a proxy
#
# Sections 1-4 answer where the time goes around the model. What they cannot see
# is the step itself: for that, run pretraining with TRANSEHR2_PHASE_TIMING set to
# a report interval and read the proportions it prints. That path synchronizes to
# time GPU phases, so its totals run long -- compare them against an
# uninstrumented run, not against these.
#
# Read 2 against 1: if they are close, the storage is the limit and more workers
# will not help. Read 5 against the epoch time: if the compute floor is a small
# fraction of it, the model is not what the epochs are measuring.
#
# Section 0 exists because the sweep arms are otherwise not exchangeable: the
# first reads cold and leaves the cohort resident for the rest. Two runs of the
# same sweep, one on a node a previous job had warmed and one on a cold node,
# differed 2.3x at 8 workers for no other reason. With the cohort read once up
# front every arm measures the warm path, which is the one a training run is on
# after its first epoch -- and section 0's own rate is the cold cost of that
# first epoch. Compare the size it prints against --mem to confirm the cohort
# stays resident at all.

set -uo pipefail

# This copy is tracked inside the repo, so it can be submitted from the project
# root (where the untracked SLURM/ also lives) or from the repo itself. sbatch
# runs a spooled copy, so $0 is no guide -- the submit directory is, and which
# of the two it is shows in what sits beside it.
SUBMIT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ -d "${SUBMIT}/TransEHR2" ]; then
    PROJECT_ROOT="${PROJECT_ROOT:-${SUBMIT}}"
elif [ -f "${SUBMIT}/scripts/diagnose_dataloading.py" ]; then
    PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "${SUBMIT}")}"
else
    PROJECT_ROOT="${PROJECT_ROOT:-${SUBMIT}}"
fi
TRANSEHR2_DIR="${TRANSEHR2_DIR:-${PROJECT_ROOT}/TransEHR2}"
TRANSEHR2_VENV="${TRANSEHR2_VENV:-${TRANSEHR2_DIR}/venv/TransEHR2}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
DATASET_CONFIG="${DATASET_CONFIG:-TransEHR2/configs/datasets/RMT23345.yaml}"
FOLD="${FOLD:-fold0}"
BATCH_SIZE="${BATCH_SIZE:-200}"
# Enough to outrun the prefetch queue. With 16 workers and prefetch_factor 2 the loader
# hands back 32 batches before producing anything, so a 20-batch run measured the queue
# draining and reported 588 GiB/s. The timing is over the whole run now, but the run still
# has to be long enough for production to dominate it.
BATCHES="${BATCHES:-200}"
WORKERS="${WORKERS:-0,4,8,16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
STEPS="${STEPS:-300}"
N_FEATURES="${N_FEATURES:-149}"
CHUNK="${CHUNK:-1024}"
DENSITY="${DENSITY:-0.056}"
BATCH_SIZES="${BATCH_SIZES:-50,100,200,400}"
# Each LazyArray opens its memmap once per process, so a short profile charges ~165 one-time
# opens to a couple of hundred items and reads as if opening files were a per-item cost.
PROFILE_ITEMS="${PROFILE_ITEMS:-2000}"
STAGES="${STAGES:-all}"
SMI_INTERVAL="${SMI_INTERVAL:-10}"

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-<none>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
mkdir -p log

if [ ! -f "${TRANSEHR2_VENV}/bin/activate" ]; then
    echo "ERROR: no virtualenv at ${TRANSEHR2_VENV}. Override TRANSEHR2_VENV." >&2
    exit 1
fi
if [ ! -d "${DATA_DIR}/extracted" ]; then
    echo "ERROR: no ${DATA_DIR}/extracted. Re-extract before running this." >&2
    exit 1
fi

# Power is the better tell than utilisation: GPU-Util counts any kernel being
# resident, so a trivial kernel looping reads 100%, whereas draw against the cap
# tracks how much of the card is actually working.
nvidia-smi --query-gpu=timestamp,utilization.gpu,power.draw,memory.used \
           --format=csv -l "${SMI_INTERVAL}" > "log/diagnose_gpu_${SLURM_JOB_ID:-0}.csv" &
SMI_PID=$!
trap 'kill ${SMI_PID} 2>/dev/null' EXIT

cd "${TRANSEHR2_DIR}" || exit 1
# shellcheck disable=SC1091
source "${TRANSEHR2_VENV}/bin/activate" || exit 1

echo
echo "########## 1-4: where the step's time goes ##########"
python scripts/diagnose_dataloading.py \
    --data-dir "${DATA_DIR}" \
    --fold "${FOLD}" \
    --batch-size "${BATCH_SIZE}" \
    --batches "${BATCHES}" \
    --workers "${WORKERS}" \
    --prefetch-factor "${PREFETCH_FACTOR}" \
    --batch-sizes "${BATCH_SIZES}" \
    --profile-items "${PROFILE_ITEMS}" \
    --stages "${STAGES}"
status=$?

echo
echo "########## 5: GPU compute floor, 1 vs 2 encoder blocks ##########"
for blocks in 1 2; do
    echo
    echo "---------- ${blocks} encoder block(s) ----------"
    python scripts/benchmark_grad_attribution.py \
        --skip-encoder \
        --stage1 "${STEPS},${N_FEATURES}" \
        --stage1-batches 256,1024,2048 \
        --steps "${STEPS}" \
        --chunk "${CHUNK}" \
        --density "${DENSITY}" \
        --encoder-blocks "${blocks}" \
        --dataset-config "${DATASET_CONFIG}" \
        --ig-steps 0
    status=$(( status || $? ))
done

echo
echo "########## 6: the real pretraining step, phase by phase ##########"
#
# Sections 1-5 are proxies: the loader measured without a model, the model
# measured without a loader. This is the step itself, and it is the only thing
# that can confirm the share the rest imply.
#
# Run at two worker counts because run_experiment.py defaults --num_workers to
# 0, and section 2 puts 0 workers at 2.7x the cost of 4. If the pretraining jobs
# have been taking that default, the cheapest available change is a flag.
#
# The config is generated, not edited: PRETRAIN_TOTAL_EPOCH and
# FINETUNE_TOTAL_EPOCH come down to 1, and MODEL_DIR and EXPERIMENT_NAME are
# redirected into the log directory so a timing run cannot overwrite a real
# checkpoint. --force_pretrain because a run that finds existing weights skips
# pretraining, which is the half being timed.
#
# The timer synchronizes CUDA to attribute GPU phases, so the step total runs
# long by construction. Read the proportions; read the total against section 5.

PHASE_WORKERS="${PHASE_WORKERS:-0,8}"
PHASE_EVERY="${PHASE_EVERY:-100}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-TransEHR2/configs/experiments/experiment1_baseline.yaml}"
PHASE_DIR="${PROJECT_ROOT}/log/phase_timing_${SLURM_JOB_ID:-0}"

if [ ! -f "${EXPERIMENT_CONFIG}" ]; then
    echo "SKIPPED: no ${EXPERIMENT_CONFIG} to derive a timing config from." >&2
else
    mkdir -p "${PHASE_DIR}"
    python - "${EXPERIMENT_CONFIG}" "${PHASE_DIR}" <<'PYEOF'
import sys
import yaml

source, phase_dir = sys.argv[1], sys.argv[2]
with open(source) as handle:
    config = yaml.safe_load(handle)
config['PRETRAIN_TOTAL_EPOCH'] = 1
config['FINETUNE_TOTAL_EPOCH'] = 1
config['MODEL_DIR'] = phase_dir
config['EXPERIMENT_NAME'] = 'phase_timing'
with open(f'{phase_dir}/experiment.yaml', 'w') as handle:
    yaml.safe_dump(config, handle)
print(f"timing config written to {phase_dir}/experiment.yaml")
PYEOF
    for workers in $(echo "${PHASE_WORKERS}" | tr ',' ' '); do
        echo
        echo "---------- pretraining, ${workers} dataloader worker(s) ----------"
        TRANSEHR2_PHASE_TIMING="${PHASE_EVERY}" \
        python scripts/run_experiment.py \
            "${DATASET_CONFIG}" \
            "${PHASE_DIR}/experiment.yaml" \
            --folds "${FOLD}" \
            --num_workers "${workers}" \
            --force_pretrain
        status=$(( status || $? ))
    done
    echo
    echo "Timing checkpoints are under ${PHASE_DIR} and are not a trained model;"
    echo "remove it once the numbers above have been read."
fi

echo
echo "GPU sampling written to log/diagnose_gpu_${SLURM_JOB_ID:-0}.csv"
echo "Job finished at $(date) with status ${status}"
exit ${status}
