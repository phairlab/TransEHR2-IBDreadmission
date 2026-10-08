#!/bin/bash
#SBATCH --job-name=diagnose
#SBATCH --partition=gpu-h200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --gpus=h200:1
#SBATCH --nodes=1
#SBATCH --time=0-01:30:00
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
#   1. raw      episode rows straight off the memmaps -- the storage ceiling
#   2. loader   the real prepare_dataloaders path, swept over worker counts
#   3. device   host-to-device copy for one batch
#   4. padding  how much of what was read is padding, and what a shorter
#               extraction would win
#   5. stage 1  the GPU compute floor at 1 and at 2 encoder blocks, data
#               resident, which is the block question with the I/O removed
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
BATCHES="${BATCHES:-20}"
WORKERS="${WORKERS:-0,4,8,16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
STEPS="${STEPS:-300}"
N_FEATURES="${N_FEATURES:-149}"
CHUNK="${CHUNK:-1024}"
DENSITY="${DENSITY:-0.056}"
BATCH_SIZES="${BATCH_SIZES:-50,100,200,400}"
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
echo "GPU sampling written to log/diagnose_gpu_${SLURM_JOB_ID:-0}.csv"
echo "Job finished at $(date) with status ${status}"
exit ${status}
