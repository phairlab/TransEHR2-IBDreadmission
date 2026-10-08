#!/bin/bash
#SBATCH --job-name=repair_ids
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --time=0-01:00:00
#SBATCH --output=log/repair_ids_%j.out
#SBATCH --error=log/repair_ids_%j.err
#SBATCH --qos=normal

#SBATCH --mail-user=pr3@ualberta.ca
#SBATCH --mail-type=END,FAIL

# Names the patients whose data/root directory has to be rebuilt after the two backshift
# fixes (IBDdataprep 62c7352 and b926286), writing them where build_root's --patients-file
# can read them.
#
#     sbatch SLURM/slurm_repair_ids.sh
#     WINDOW_DAYS=3 sbatch SLURM/slurm_repair_ids.sh     # a wider superset
#
# WHY THIS IS A JOB AND NOT A LOGIN-NODE COMMAND
#
# The only expensive thing it does is read RMT23345_AMB_prepared.csv, and only for PATID and
# TIMESTAMP -- but AMB carries TEXT_SUPERFEATURE, so the table is multi-GB and pandas'
# usecols discards the other columns only after the parser has built the row. The reads are
# chunked, which bounds the peak, but not far enough below a login node's cap to rely on.
# 16G is generous for what it does; the job is minutes long, so it costs little to ask.
#
# WHAT IT DOES NOT DO
#
# It writes a list. Nothing is rebuilt, nothing is deleted, and data/root is not opened. The
# rebuild is a separate submission, and the script prints the exact command for it.
#
# THE FAILURE LOG IS NOT OPTIONAL
#
# Patients build_root skipped have no directory at all, and they are the ones the fixes exist
# for. They are recoverable only from the failure log of the run that skipped them, so the
# newest log/build_root_*.log is picked up automatically and its absence is an error rather
# than a quietly shorter list. Set FAILURE_LOG to choose a different one, or
# ALLOW_NO_FAILURE_LOG=1 if you have already rebuilt the skipped patients another way.

set -uo pipefail

# Submitted from the project root, so SLURM_SUBMIT_DIR is that root and there is no cluster
# path to hardcode -- the same file works on the cluster and against a local checkout.
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
TRANSEHR2_DIR="${TRANSEHR2_DIR:-${PROJECT_ROOT}/TransEHR2}"
IBDDATAPREP_DIR="${IBDDATAPREP_DIR:-${PROJECT_ROOT}/IBDdataprep}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
RAW_DIR="${RAW_DIR:-${DATA_DIR}/ibd/RMT23345}"
PREPARED_DIR="${PREPARED_DIR:-${RAW_DIR}/RMT23345_prepared}"
IBD_VENV="${IBD_VENV:-${TRANSEHR2_DIR}/venv/IBDdataprep}"
FINDER="${FINDER:-${PROJECT_ROOT}/development/scripts/backshift_repair_ids.py}"
OUT="${OUT:-${PROJECT_ROOT}/repair_ids.txt}"
WINDOW_DAYS="${WINDOW_DAYS:-1}"

cd "${PROJECT_ROOT}" || exit 1
mkdir -p log

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-<none>}"
echo "WINDOW_DAYS=${WINDOW_DAYS}"

if [ ! -f "${FINDER}" ]; then
    echo "ERROR: no finder script at ${FINDER}." >&2
    exit 1
fi
if [ ! -d "${PREPARED_DIR}" ]; then
    echo "ERROR: no prepared tables at ${PREPARED_DIR}. Override PREPARED_DIR." >&2
    exit 1
fi
if [ ! -f "${IBD_VENV}/bin/activate" ]; then
    echo "ERROR: no virtualenv at ${IBD_VENV}. Override IBD_VENV." >&2
    exit 1
fi

# Newest first, so a fresh run's log wins over the one before it.
if [ -z "${FAILURE_LOG:-}" ]; then
    FAILURE_LOG=$(ls -t log/build_root_*.log 2>/dev/null | head -1)
fi
if [ -z "${FAILURE_LOG:-}" ] || [ ! -f "${FAILURE_LOG}" ]; then
    if [ "${ALLOW_NO_FAILURE_LOG:-0}" != "1" ]; then
        echo "ERROR: no build_root failure log found under log/." >&2
        echo "       The patients it skipped were never written to data/root, so a list" >&2
        echo "       built without it would leave exactly them missing. Set FAILURE_LOG to" >&2
        echo "       the log of the run that skipped them, or ALLOW_NO_FAILURE_LOG=1 if they" >&2
        echo "       are already accounted for." >&2
        exit 1
    fi
    echo "WARNING: proceeding with no failure log (ALLOW_NO_FAILURE_LOG=1)."
    echo "         Patients build_root skipped are NOT in the list this writes."
    FAILURE_LOG="/nonexistent"
else
    echo "Failure log: ${FAILURE_LOG}"
fi

# shellcheck disable=SC1091
source "${IBD_VENV}/bin/activate"
echo "  python: $(command -v python)"
echo ""

START=$SECONDS
python "${FINDER}" \
    --prepared "${PREPARED_DIR}" \
    --failure-log "${FAILURE_LOG}" \
    --out "${OUT}" \
    --window-days "${WINDOW_DAYS}"
STATUS=$?
DURATION=$((SECONDS - START))
printf "\nbackshift_repair_ids.py finished in %02d:%02d:%02d with status %s\n" \
    $((DURATION / 3600)) $((DURATION % 3600 / 60)) $((DURATION % 60)) "${STATUS}"
[ ${STATUS} -eq 0 ] || exit ${STATUS}

echo ""
echo "Read the union percentage above before rebuilding. Targeting only pays while the list"
echo "is a small share of the cohort; past roughly a third of it, the clean rebuild"
echo "(./prepare_data.sh --scope prepared) is simpler and not much slower."
echo ""
echo "The rebuild writes into the existing data/root and must not clean it, which is why it"
echo "is build_root.py directly rather than a scope of prepare_data.sh:"
echo ""
echo "    source ${IBD_VENV}/bin/activate"
echo "    export PYTHONPATH=\"\${PYTHONPATH:-}:${IBDDATAPREP_DIR}\""
echo "    python -m IBDdataprep.build_root \\"
echo "        --prepared ${PREPARED_DIR} \\"
echo "        --root ${DATA_DIR}/root \\"
echo "        --patients-file ${OUT} \\"
echo "        --log ${PROJECT_ROOT}/log/build_root_repair.log"
echo ""
echo "then validate_root, then ./prepare_data.sh --scope labels --chain for steps 6-7."
echo "REPAIR_IDS_OK"
