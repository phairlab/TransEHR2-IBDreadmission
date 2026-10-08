#!/bin/bash
#SBATCH --job-name=blockcount
#SBATCH --partition=gpu-h200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --gpus=h200:2
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --output=log/blockcount_%j.out
#SBATCH --error=log/blockcount_%j.err
#SBATCH --qos=normal

#SBATCH --mail-user=pr3@ualberta.ca
#SBATCH --mail-type=END,FAIL

# Times 20 pretraining epochs with 1 vs 2 encoder blocks in the generator and
# discriminator, on fold0, and reports the per-epoch average for each.
#
#   sbatch SLURM/slurm_blockcount_timing.sh
#
# Development only, untracked. NOT part of the hyperparameter tuning
# procedure -- it is a cost measurement, so the only output that matters is
# seconds per epoch. No finetuning runs; the driver stops as soon as
# pretraining returns.
#
# TWO GPUs, ONE JOB, ON PURPOSE. The two arms run at the same time on one
# node so that they see the same hardware, the same filesystem and the same
# neighbours. Two separate jobs could land on different nodes, and a 10%
# difference between nodes is the same size as the difference being
# measured.
#
# The cost of that choice: the two processes share CPU, host memory
# bandwidth and the data path. If pretraining turns out to be input-bound
# rather than GPU-bound, contention will compress the gap between the arms
# and the second block will look cheaper than it is. --cpus-per-task is
# sized at 8 per process to make that unlikely rather than impossible. The
# train/val split in the summary is the thing to read if the two arms come
# out suspiciously close: if val time barely moves with block count, the run
# was not compute-bound.
#
# Both arms are driven from one base config with a single knob, so they
# cannot drift apart in anything but the block count. The materialised
# config for each arm is written beside its results.
#
# The driver deletes its own checkpoint, log and model directories before
# starting -- a resumed pretraining checkpoint would skip epochs and time
# the wrong number of them. It refuses to delete anything not named
# blockcount_timing_*.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
TRANSEHR2_DIR="${TRANSEHR2_DIR:-${PROJECT_ROOT}/TransEHR2}"
TRANSEHR2_VENV="${TRANSEHR2_VENV:-${TRANSEHR2_DIR}/venv/TransEHR2}"
DRIVER="${DRIVER:-${PROJECT_ROOT}/development/scripts/blockcount_timing.py}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/development/scripts/blockcount_timing.yaml}"
DATASET_CONFIG="${DATASET_CONFIG:-${TRANSEHR2_DIR}/TransEHR2/configs/datasets/RMT23345.yaml}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/development/scripts/blockcount_timing_out}"
FOLD="${FOLD:-fold0}"
NUM_WORKERS="${NUM_WORKERS:-4}"

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-<none>}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

cd "${PROJECT_ROOT}" || exit 1
mkdir -p log "${OUT_DIR}"

if [ ! -f "${TRANSEHR2_VENV}/bin/activate" ]; then
    echo "ERROR: no virtualenv at ${TRANSEHR2_VENV}. Override TRANSEHR2_VENV." >&2
    exit 1
fi

n_gpus=$(nvidia-smi --list-gpus | wc -l)
if [ "${n_gpus}" -lt 2 ]; then
    echo "ERROR: ${n_gpus} GPU(s) visible, need 2. The arms must run " \
         "together on one node for the comparison to mean anything." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "${TRANSEHR2_VENV}/bin/activate" || exit 1

run_arm() {
    local blocks=$1
    local gpu=$2
    local logfile="log/blockcount_blocks${blocks}_${SLURM_JOB_ID:-local}.out"
    echo "arm: ${blocks} block(s) on GPU ${gpu} -> ${logfile}"
    CUDA_VISIBLE_DEVICES="${gpu}" python "${DRIVER}" \
        --blocks "${blocks}" \
        --config "${CONFIG}" \
        --dataset-config "${DATASET_CONFIG}" \
        --transehr2-dir "${TRANSEHR2_DIR}" \
        --out-dir "${OUT_DIR}" \
        --fold "${FOLD}" \
        --num-workers "${NUM_WORKERS}" \
        > "${logfile}" 2>&1
}

run_arm 1 0 &
pid1=$!
run_arm 2 1 &
pid2=$!

wait ${pid1}; status1=$?
wait ${pid2}; status2=$?
echo "arm 1 block exited ${status1}, arm 2 blocks exited ${status2}"

if [ ${status1} -eq 0 ] && [ ${status2} -eq 0 ]; then
    echo
    python "${DRIVER}" --summarise \
        "${OUT_DIR}/blocks1.json" "${OUT_DIR}/blocks2.json"
    status=0
else
    echo "ERROR: an arm failed; see the per-arm logs in log/." >&2
    status=1
fi

echo "Job finished at $(date) with status ${status}"
exit ${status}
