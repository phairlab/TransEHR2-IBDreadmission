#!/bin/bash
#SBATCH --job-name=embed
#SBATCH --partition=gpu-h200
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=h200:1
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --output=log/embed_%j.out
#SBATCH --error=log/embed_%j.err
#SBATCH --qos=normal

#SBATCH --mail-user=pr3@ualberta.ca
#SBATCH --mail-type=END,FAIL

# Builds the global lookup tables: text_embeddings.npy, text_tokens.npy and
# drug_embeddings.npy under data/lookup_tables.
#
#   sbatch SLURM/slurm_embed.sh
#   TABLES=drug sbatch SLURM/slurm_embed.sh     # ClinVec only, loads no LLM
#
# MUST RUN AFTER EXTRACTION, AND AFTER EVERY EXTRACTION THAT REBUILT THE ROOT.
#
# extract_data.py assigns the unique-string row order and writes it to text_strings.pkl;
# embed.py embeds that list in that order, and the arrays' `text_values` are positions in it.
# A stale text_embeddings.npy therefore has the right shape and the wrong contents, and nothing
# downstream can tell. ./prepare_data.sh --chain queues this behind the extraction that
# invalidated it.
#
# ONE GPU, AND THE TIME LIMIT IS STILL A GUESS.
#
# TransEHR2/constants.py now names ContrastiveBMMB's A3 encoder -- BioClinical ModernBERT
# large, 396M parameters, under 1 GB at bf16. One card is ample. This asked for two when the
# encoder was meta-llama/Llama-3.1-70B, ~140 GB against an H200's 141, where device_map='auto'
# had to shard across a second card to leave room for activations. None of that applies to an
# encoder-only model, and a second card now buys nothing: embed.py runs one batch at a time and
# has no data parallelism to spend it on.
#
# 64G rather than 250G for the same reason, plus a correction. build_text_tables opens both
# text_tokens.npy and text_embeddings.npy with np.lib.format.open_memmap and fills them a batch
# at a time, so neither is ever held whole -- the tens of gigabytes they run to are page cache,
# which is reclaimable and not a reservation. What actually needs resident memory is
# text_strings.pkl, unpickled whole: at section 4.5's ~3.95 M unique strings and a few KB for a
# record filling all 25 diagnosis and 20 intervention slots, that is tens of GB at the worst.
# 64G covers it with headroom. If this is ever killed rather than failing, it was the pickle.
#
# The walltime is deliberately NOT cut to match the faster encoder. 12 h is now generous --
# expect a few hours -- but embed.py has no resume: a job that dies at the limit has written a
# partial text_embeddings.npy with the right shape and the wrong contents past the last batch,
# which nothing downstream can detect. Trading queue priority for that is the wrong trade.
# Watch the first run's throughput, and if you do lower this, lower it to twice what it
# reports rather than to what it reports.

set -uo pipefail

# Submitted from the project root, so SLURM_SUBMIT_DIR is that root and there is no cluster
# path to hardcode -- the same file works on the cluster and against a local checkout.
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
TRANSEHR2_DIR="${TRANSEHR2_DIR:-${PROJECT_ROOT}/TransEHR2}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
DATASET_CONFIG="${DATASET_CONFIG:-${TRANSEHR2_DIR}/TransEHR2/configs/datasets/RMT23345.yaml}"
TRANSEHR2_VENV="${TRANSEHR2_VENV:-${TRANSEHR2_DIR}/venv/TransEHR2}"
TABLES="${TABLES:-all}"
BATCH_SIZE="${BATCH_SIZE:-64}"

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-<none>}"
echo "TABLES=${TABLES}  BATCH_SIZE=${BATCH_SIZE}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
mkdir -p log

if [ ! -d "${DATA_DIR}/extracted" ]; then
    echo "ERROR: no ${DATA_DIR}/extracted. Extraction has to run first:" >&2
    echo "       ./prepare_data.sh" >&2
    exit 1
fi
if [ "${TABLES}" != "drug" ] && [ ! -f "${DATA_DIR}/extracted/text_strings.pkl" ]; then
    echo "ERROR: no text_strings.pkl beside the arrays, so there is no string order to" >&2
    echo "       embed against. Re-run extraction." >&2
    exit 1
fi
if [ ! -f "${TRANSEHR2_VENV}/bin/activate" ]; then
    echo "ERROR: no virtualenv at ${TRANSEHR2_VENV}. Override TRANSEHR2_VENV." >&2
    exit 1
fi

cd "${TRANSEHR2_DIR}" || exit 1
# shellcheck disable=SC1091
source "${TRANSEHR2_VENV}/bin/activate"
export PYTHONPATH="${PYTHONPATH:-}:${TRANSEHR2_DIR}"

echo "  python: $(command -v python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
python -c "from TransEHR2.constants import LLM_NAME, MAX_TOKEN_LENGTH; \
print(f'Text encoder: {LLM_NAME} at {MAX_TOKEN_LENGTH} tokens')"

START=$SECONDS
python scripts/embed.py "${DATASET_CONFIG}" \
    --data_dir "${DATA_DIR}" \
    --tables "${TABLES}" \
    --batch-size "${BATCH_SIZE}"
STATUS=$?
DURATION=$((SECONDS - START))
printf "embed.py finished in %02d:%02d:%02d with status %s\n" \
    $((DURATION / 3600)) $((DURATION % 3600 / 60)) $((DURATION % 60)) "${STATUS}"
[ ${STATUS} -eq 0 ] || exit ${STATUS}

echo ""
echo "Tables in ${DATA_DIR}/lookup_tables:"
ls -la "${DATA_DIR}/lookup_tables" 2>/dev/null || echo "  (none written)"
echo "EMBED_RUN_OK"
