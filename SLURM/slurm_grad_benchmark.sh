#!/bin/bash
#SBATCH --job-name=gradbench
#SBATCH --partition=gpu-h200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --gpus=h200:1
#SBATCH --nodes=1
#SBATCH --time=0-01:00:00
#SBATCH --output=log/gradbench_%j.out
#SBATCH --error=log/gradbench_%j.err
#SBATCH --qos=normal

#SBATCH --mail-user=pr3@ualberta.ca
#SBATCH --mail-type=END,FAIL

# Times stage 2 of token attribution -- an encoder forward plus a local backward to the input
# embedding layer -- at the batch sizes and sequence widths the study uses.
#
#   sbatch SLURM/slurm_grad_benchmark.sh
#   BATCH_SIZES=1,2,4 CHECKPOINTING=1 sbatch SLURM/slurm_grad_benchmark.sh
#   TOKENS=data/lookup_tables/text_tokens.npy sbatch SLURM/slurm_grad_benchmark.sh
#   TSR=300,40 STAGE1_MS=25 sbatch SLURM/slurm_grad_benchmark.sh
#   STAGE1=300,40 SKIP_ENCODER=1 CHUNK=256 sbatch SLURM/slurm_grad_benchmark.sh
#   STAGE1=500,148 SKIP_ENCODER=1 STEPS=300,500 CHUNK=256 \
#       DATASET_CONFIG=TransEHR2/configs/datasets/RMT23345.yaml \
#       sbatch SLURM/slurm_grad_benchmark.sh
#
# ONE GPU, DELIBERATELY, and that holds for stage 1 as much as for stage 2. TSR has no
# collective in it -- every perturbation is independent, and the batched implementation
# exploits that inside one card -- so a multi-card request would leave every card but one
# idle. What more cards buy is more episodes at once, which is the explanation run and not
# this benchmark; that parallelism is a job array over episodes, not an allocation.
#
# For stage 2: if batch 1 OOMs at --width 2048, that is the finding; rerun with
# CHECKPOINTING=1 before concluding anything.
#
# STAGE1 times the TransEHR2 forward+backward TSR spends all its passes in, then runs one
# episode end to end per horizon. SKIP_ENCODER=1 skips the stage-2 LLM entirely, which is the
# fast path and the one that needs no checkpoint on disk.
#
# IG_STEPS swaps grad x input for integrated gradients as TSR's R(.), multiplying every pass
# by it. DENSITY sets what fraction of the synthetic episode's cells are observed: TSR skips
# the rest, because deleting a cell that was never there cannot move the map, and at EHR
# density that skip is most of the cost. 1.0 is the worst case, not a realistic one -- set it
# from the real occupancy, or read the skipped percentage off a run against real episodes.
#
# DATASET_CONFIG sizes that model from the study's own feature set -- 147 valued features plus
# the text superfeature, each at the width variable_properties gives it -- instead of repeating
# one --feat-width. Without it the shape is a stand-in, and TSR costs 1 + T + T*N, so N being
# wrong is a linear error in the headline number. STEPS sweeps the episode length; 500 is
# MAX_EPISODE_LEN_STEPS, and a shorter window is a real lever because T enters the T*N term.
#
# The default --time of one hour is sized for stage 2. The full shape is ~74.5k perturbations
# per horizon and six horizons, so raise it if the job is cut off; the horizons cost the same
# as each other, which two measured shapes now agree on, so one horizon is a defensible
# shortcut.
#
# Requires the grad-benchmark branch, whose constants.py points LLM_NAME at ContrastiveBMMB's
# A3 checkpoint. Check that path exists on this machine before submitting -- a missing
# directory fails at load, several minutes in.
#
# TOKENS reads text_tokens.npy for a realistic length distribution. It is cohort-derived: the
# script prints length percentiles and timings only, never a row, but it is operator-run by the
# same rule the dedup report follows. Without it every sequence runs at full width, which is a
# worst case rather than a wrong one.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
TRANSEHR2_DIR="${TRANSEHR2_DIR:-${PROJECT_ROOT}/TransEHR2}"
TRANSEHR2_VENV="${TRANSEHR2_VENV:-${TRANSEHR2_DIR}/venv/TransEHR2}"
BATCH_SIZES="${BATCH_SIZES:-1,2,4,8,16}"
WIDTH="${WIDTH:-2048}"
REPEATS="${REPEATS:-5}"
DTYPE="${DTYPE:-bfloat16}"
PROJECT="${PROJECT:-100000}"
TOKENS="${TOKENS:-}"
TSR="${TSR:-}"
STAGE1_MS="${STAGE1_MS:-}"
RENDER_CELLS="${RENDER_CELLS:-}"
CHECKPOINTING="${CHECKPOINTING:-}"
STAGE1="${STAGE1:-}"
STAGE1_BATCHES="${STAGE1_BATCHES:-1,8,32,64,128,256}"
CHUNK="${CHUNK:-}"
STEPS="${STEPS:-}"
IG_STEPS="${IG_STEPS:-}"
DENSITY="${DENSITY:-}"
DATASET_CONFIG="${DATASET_CONFIG:-}"
GATE_QUANTILE="${GATE_QUANTILE:-}"
SKIP_ENCODER="${SKIP_ENCODER:-}"

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-<none>}"
echo "BATCH_SIZES=${BATCH_SIZES}  WIDTH=${WIDTH}  DTYPE=${DTYPE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
mkdir -p log

if [ ! -f "${TRANSEHR2_VENV}/bin/activate" ]; then
    echo "ERROR: no virtualenv at ${TRANSEHR2_VENV}. Override TRANSEHR2_VENV." >&2
    exit 1
fi

cd "${TRANSEHR2_DIR}" || exit 1
branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
if [ "${branch}" != "grad-benchmark" ]; then
    echo "WARNING: on branch '${branch}', not grad-benchmark." >&2
fi

# shellcheck disable=SC1091
source "${TRANSEHR2_VENV}/bin/activate" || exit 1

# Built as an array so that an unset optional flag contributes no argument at all. Under
# `set -u` an unquoted empty variable expands to an empty string, which argparse reads as a
# positional and rejects.
cmd=(python scripts/benchmark_grad_attribution.py
     --batch-sizes "${BATCH_SIZES}"
     --width "${WIDTH}"
     --repeats "${REPEATS}"
     --dtype "${DTYPE}"
     --project "${PROJECT}")
[ -n "${TOKENS}" ] && cmd+=(--tokens "${PROJECT_ROOT}/${TOKENS}")
[ -n "${CHECKPOINTING}" ] && cmd+=(--checkpointing)
[ -n "${TSR}" ] && cmd+=(--tsr "${TSR}")
[ -n "${STAGE1_MS}" ] && cmd+=(--stage1-ms "${STAGE1_MS}")
[ -n "${RENDER_CELLS}" ] && cmd+=(--render-cells "${RENDER_CELLS}")
[ -n "${STAGE1}" ] && cmd+=(--stage1 "${STAGE1}" --stage1-batches "${STAGE1_BATCHES}")
[ -n "${CHUNK}" ] && cmd+=(--chunk "${CHUNK}")
[ -n "${STEPS}" ] && cmd+=(--steps "${STEPS}")
[ -n "${IG_STEPS}" ] && cmd+=(--ig-steps "${IG_STEPS}")
[ -n "${DENSITY}" ] && cmd+=(--density "${DENSITY}")
[ -n "${DATASET_CONFIG}" ] && cmd+=(--dataset-config "${DATASET_CONFIG}")
[ -n "${GATE_QUANTILE}" ] && cmd+=(--gate-quantile "${GATE_QUANTILE}")
[ -n "${SKIP_ENCODER}" ] && cmd+=(--skip-encoder)

echo "+ ${cmd[*]}"
"${cmd[@]}"
status=$?

echo "Job finished at $(date) with status ${status}"
exit ${status}
