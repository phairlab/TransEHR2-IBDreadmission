#!/bin/bash
#SBATCH --job-name=rebuild_patients
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --time=0-06:00:00
#SBATCH --output=log/rebuild_patients_%j.out
#SBATCH --error=log/rebuild_patients_%j.err
#SBATCH --qos=normal

#SBATCH --mail-user=pr3@ualberta.ca
#SBATCH --mail-type=END,FAIL

# Rebuilds a named subset of data/root in place, leaving every other patient directory as it
# is. For repairing the patients the backshift fixes changed (IBDdataprep 62c7352, b926286)
# without paying for a whole-cohort step 4.
#
#     sbatch SLURM/slurm_rebuild_patients.sh
#     PATIENTS_FILE=/path/to/ids.txt sbatch SLURM/slurm_rebuild_patients.sh
#     sbatch --mem=128G --time=0-12:00:00 SLURM/slurm_rebuild_patients.sh
#
# NOTHING IS DELETED, AND THAT IS THE POINT
#
# prepare_data.sh's 'prepared' scope removes data/root before rebuilding it, which is correct
# when every patient is being rewritten and fatal here -- the patients not in the list are the
# ones being kept. So this calls build_root.py directly. CLEAN_ROOT lives in the pipeline
# wrapper, not in build_root.py, which only ever writes the directories it is given.
#
# SIZING IS A GUESS, AND IT SCALES WITH THE LIST
#
# read_prepared() loads all seven prepared tables into memory at once, filtered to the
# requested patients inside the chunk loop, so peak memory tracks the list's share of the
# cohort rather than the cohort. Step 4 of the full pipeline asks for 256G to hold all of it;
# 64G is the allowance for a subset. Runtime does NOT scale the same way -- every table is
# scanned end to end whatever the list length -- so a short list is cheap in memory and still
# pays most of the I/O. If the job is killed rather than failing, it was the memory: resubmit
# with sbatch --mem=.
#
# A NON-ZERO EXIT IS INFORMATION, NOT NECESSARILY A CRASH
#
# build_root.py returns 1 when it skips a patient, having written every other one first. After
# a repair run that means the fixes did not cover everything, and the failure log names who is
# left. Read it before resubmitting: a second run of the same list will skip the same patients.

set -uo pipefail

# Submitted from the project root, so SLURM_SUBMIT_DIR is that root and there is no cluster
# path to hardcode -- the same file works on the cluster and against a local checkout.
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
TRANSEHR2_DIR="${TRANSEHR2_DIR:-${PROJECT_ROOT}/TransEHR2}"
IBDDATAPREP_DIR="${IBDDATAPREP_DIR:-${PROJECT_ROOT}/IBDdataprep}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
PREPARED_DIR="${PREPARED_DIR:-${DATA_DIR}/ibd/RMT23345_prepared}"
DATASET_CONFIG="${DATASET_CONFIG:-${TRANSEHR2_DIR}/TransEHR2/configs/datasets/RMT23345.yaml}"
VARIABLE_PROPERTIES="${VARIABLE_PROPERTIES:-${DATA_DIR}/ibd/variable_properties.yaml}"
IBD_VENV="${IBD_VENV:-${TRANSEHR2_DIR}/venv/IBDdataprep}"
PATIENTS_FILE="${PATIENTS_FILE:-${PROJECT_ROOT}/repair_ids.txt}"

cd "${PROJECT_ROOT}" || exit 1
mkdir -p log
BUILD_LOG="${PROJECT_ROOT}/log/build_root_repair_${SLURM_JOB_ID:-$$}.log"

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-<none>}"

for REQUIRED in "${PREPARED_DIR}" "${DATA_DIR}/root"; do
    if [ ! -d "${REQUIRED}" ]; then
        echo "ERROR: no ${REQUIRED}." >&2
        echo "       This repairs an existing root. To build one from nothing, run" >&2
        echo "       ./prepare_data.sh --scope prepared instead." >&2
        exit 1
    fi
done
if [ ! -f "${PATIENTS_FILE}" ]; then
    echo "ERROR: no patient list at ${PATIENTS_FILE}." >&2
    echo "       Write one with: sbatch SLURM/slurm_repair_ids.sh" >&2
    echo "       or set PATIENTS_FILE to a list you already have." >&2
    exit 1
fi
if [ ! -f "${IBD_VENV}/bin/activate" ]; then
    echo "ERROR: no virtualenv at ${IBD_VENV}. Override IBD_VENV." >&2
    exit 1
fi

# Blank and '#' lines are ignored by build_root.py, so count the way it does.
N_PATIENTS=$(sed 's/#.*//' "${PATIENTS_FILE}" | grep -c '[^[:space:]]')
if [ "${N_PATIENTS}" -eq 0 ]; then
    echo "ERROR: ${PATIENTS_FILE} names no patients. Nothing to rebuild." >&2
    exit 1
fi
N_EXISTING=$(find "${DATA_DIR}/root" -mindepth 1 -maxdepth 1 -type d | wc -l)

echo ""
echo "Rebuilding ${N_PATIENTS} patient(s) from ${PATIENTS_FILE}"
echo "  into ${DATA_DIR}/root, which holds ${N_EXISTING} directories now."
echo "  Directories not named in that list are untouched; none is deleted."
echo "  Failure detail (clinical, so not on stdout): ${BUILD_LOG}"

# shellcheck disable=SC1091
source "${IBD_VENV}/bin/activate"
export PYTHONPATH="${PYTHONPATH:-}:${IBDDATAPREP_DIR}"
echo "  python: $(command -v python)"
echo ""

START=$SECONDS
python -m IBDdataprep.build_root \
    --prepared "${PREPARED_DIR}" \
    --root "${DATA_DIR}/root" \
    --config "${DATASET_CONFIG}" \
    --variable-properties "${VARIABLE_PROPERTIES}" \
    --patients-file "${PATIENTS_FILE}" \
    --log "${BUILD_LOG}"
STATUS=$?
DURATION=$((SECONDS - START))
printf "\nbuild_root finished in %02d:%02d:%02d with status %s\n" \
    $((DURATION / 3600)) $((DURATION % 3600 / 60)) $((DURATION % 60)) "${STATUS}"

N_AFTER=$(find "${DATA_DIR}/root" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "data/root now holds ${N_AFTER} directories (was ${N_EXISTING})."

if [ ${STATUS} -ne 0 ]; then
    echo ""
    echo "Patients were still skipped. Every other one in the list was written, so this is a" >&2
    echo "shorter problem than it was, not a failed run -- but resubmitting the same list" >&2
    echo "will skip the same patients. Read ${BUILD_LOG} first." >&2
    exit ${STATUS}
fi

echo ""
echo "Next, over the whole root rather than the subset -- the invariants are cross-patient."
echo "Run it under the IBDdataprep venv, which is what step 5 of the pipeline uses; the"
echo "TransEHR2 venv carries its own pandas and is only picked up from step 7 onward:"
echo ""
echo "    source ${IBD_VENV}/bin/activate"
echo "    export PYTHONPATH=\"\${PYTHONPATH:-}:${IBDDATAPREP_DIR}\""
echo "    python -m IBDdataprep.validate_root \\"
echo "        --root ${DATA_DIR}/root \\"
echo "        --log ${PROJECT_ROOT}/log/validate_root_repair.log"
echo ""
echo "then ./prepare_data.sh --scope labels --chain for the labels, folds and arrays."
echo "REBUILD_PATIENTS_OK"
