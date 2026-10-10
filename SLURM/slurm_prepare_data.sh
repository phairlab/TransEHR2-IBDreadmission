#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --time=0-24:00:00
#SBATCH --output=log/prepare_data_%j.out
#SBATCH --error=log/prepare_data_%j.err
#SBATCH --qos=normal

#SBATCH --mail-user=pr3@ualberta.ca
#SBATCH --mail-type=END,FAIL

# The header's 24 h is the full-scope figure -- 10 h for Stage A, measured, plus a 14 h
# allowance for everything after it. ./prepare_data.sh overrides it per scope, so a bare
# sbatch is the only thing this value governs.
#
# Builds the IBD cohort from the raw RMT23345 CSVs: the per-patient root, the labels,
# the fold row indices, and the extracted arrays.
#
# Submit through ./prepare_data.sh, which runs a cheap listing pass first and then resubmits
# with the deletion enabled. A bare `sbatch` on this file deletes nothing: DRY_RUN defaults to 1.
#
#   DRY_RUN=1     list what would be removed, then stop (default)
#   DRY_RUN=0     remove it, then run the pipeline
#   CLEAN_ROOT=1  also remove data/root, the per-patient tree (default 1)
#   FIRST_STEP=N  start at step N (default 0). Steps 0-2 are Stage A (R): normalize, prepare
#                 each table, fold in the LAB variable properties. Steps 3-8 are the Python
#                 half. Use 3 to keep the prepared tables, or 6 with CLEAN_ROOT=0 to keep the
#                 per-patient root as well
#   RAW_DIR=P     the raw extract (default data/ibd/RMT23345, a symlink to the RDSS copy).
#                 Read-only: nothing here writes to it, and PREPARED_DIR sits beside it rather
#                 than under it, so the prepared tables stay on local disk.
#   CALIBRATE=1   after extraction, measure what the outlier rules would remove (default 0).
#                 Reports only. The filter itself is a cut table applied at load time, so
#                 nothing here reads or writes one.
#   N_WORKERS=N   extract_data.py worker processes (default 8, matching --cpus-per-task)
#
# WHAT THIS INVALIDATES
#
# split.py reassigns every patient to a fold, so `summary_statistics_fold{i}.npz` and anything
# trained against the old partition are void. The arrays themselves are cohort-wide and are
# rewritten here regardless; extract_data.py clears data/extracted on every run.
#
# The lookup tables are the trap. embed.py embeds `text_strings.pkl` *in the order extraction
# assigned*, and `text_values` are positions in that list -- so a re-extraction that changes
# the unique-string set silently invalidates data/lookup_tables/text_embeddings.npy without
# changing its shape. Nothing downstream would notice. Re-embed after any run that rebuilt the
# root: ./prepare_data.sh --chain queues it behind this job.
#
# Neither embedding nor training runs here. Embedding needs a GPU and is SLURM/slurm_embed.sh.

set -uo pipefail

DRY_RUN="${DRY_RUN:-1}"
CLEAN_ROOT="${CLEAN_ROOT:-1}"
FIRST_STEP="${FIRST_STEP:-0}"
CALIBRATE="${CALIBRATE:-0}"
N_WORKERS="${N_WORKERS:-8}"

# Same layout on the cluster as locally, which is what lets IBDdataprep's paths.py resolve
# every default from its own checkout and this script pass almost no paths at all.
#
# Submitted from the project root, so SLURM_SUBMIT_DIR is that root and there is no cluster
# path to hardcode -- the same file works on the cluster and against a local checkout.
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
IBDDATAPREP_DIR="${IBDDATAPREP_DIR:-${PROJECT_ROOT}/IBDdataprep}"
TRANSEHR2_DIR="${TRANSEHR2_DIR:-${PROJECT_ROOT}/TransEHR2}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
# The raw extract, read and never written. It is a symlink to the RDSS copy, so the prepared
# tables go beside it in data/ibd rather than under it: that keeps them on local disk, and it
# is where IBDdataprep/slurm_prepare_RMT23345.sh writes when one table is re-prepared on its
# own, which is only the same table if both scripts name the same directory.
RAW_DIR="${RAW_DIR:-${DATA_DIR}/ibd/RMT23345}"
PREPARED_DIR="${PREPARED_DIR:-${DATA_DIR}/ibd/RMT23345_prepared}"
DATASET_CONFIG="${DATASET_CONFIG:-${TRANSEHR2_DIR}/TransEHR2/configs/datasets/RMT23345.yaml}"
VARIABLE_PROPERTIES="${VARIABLE_PROPERTIES:-${DATA_DIR}/ibd/variable_properties.yaml}"
RESOURCES_DIR="${RESOURCES_DIR:-${DATA_DIR}/resources}"

# Stage A runs under renv: IBDdataprep/.Rprofile sources renv/activate.R, so an Rscript
# started with that directory as its working directory picks the locked library up on its own.
# That is also the working directory the scripts need, since each one sources
# helper_functions.R by a path relative to the repository root.
RSCRIPT="${RSCRIPT:-Rscript}"
# The modules that put Rscript on PATH, in load order. Two of them here: the software stack
# has to come before the R build that was compiled against it. Set to empty if Rscript is
# already on PATH, or point RSCRIPT at the interpreter to skip modules entirely.
R_MODULE="${R_MODULE:-StdEnv/2023 r/4.6.1}"

# normalize_RMT23345_datetimes.R takes the whole directory and does all seven in one call.
NORMALIZE_TABLES="${NORMALIZE_TABLES:-AMB CLM DAD DI LAB PIN REG}"
# One prepare script per table, and DI is deliberately absent: section 2.2 of the blueprint
# records that DI has no consumer, so preparing it spends time and disk on a table nothing
# downstream reads. Add it back here if that changes. SCU is absent for the opposite reason --
# prepare_RMT23345_DAD.R writes RMT23345_SCU_prepared.csv beside its own output.
PREPARE_TABLES="${PREPARE_TABLES:-AMB CLM DAD LAB PIN REG}"

# Two dependency sets: the R-side successors need pandas and pyyaml, extraction needs torch.
# Set both to the same path if one environment carries everything.
# Both live under TransEHR2/venv, whichever repository's code they serve. IBDdataprep's R half
# is Stage A and runs upstream of this script; the steps below are its Python half, so it needs
# an environment like any other.
IBD_VENV="${IBD_VENV:-${TRANSEHR2_DIR}/venv/IBDdataprep}"
TRANSEHR2_VENV="${TRANSEHR2_VENV:-${TRANSEHR2_DIR}/venv/TransEHR2}"

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-<none>}"
echo "DRY_RUN=${DRY_RUN}  CLEAN_ROOT=${CLEAN_ROOT}  FIRST_STEP=${FIRST_STEP}  CALIBRATE=${CALIBRATE}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 1
mkdir -p log

if [ ! -d "${PROJECT_ROOT}/TransEHR2" ] || [ ! -d "${PROJECT_ROOT}/IBDdataprep" ]; then
    echo "ERROR: ${PROJECT_ROOT} does not hold TransEHR2/ and IBDdataprep/, so it is not the" >&2
    echo "       project root. Submit from the root, or set PROJECT_ROOT." >&2
    exit 1
fi

for REQUIRED in "${IBDDATAPREP_DIR}" "${TRANSEHR2_DIR}" "${RAW_DIR}" "${RESOURCES_DIR}"; do
    if [ ! -d "${REQUIRED}" ]; then
        echo "ERROR: ${REQUIRED} is not a directory." >&2
        echo "       Override PROJECT_ROOT, IBDDATAPREP_DIR, TRANSEHR2_DIR, RAW_DIR or" >&2
        echo "       RESOURCES_DIR." >&2
        exit 1
    fi
done
# PREPARED_DIR is an output when the R stage runs and an input when it does not.
if [ "${FIRST_STEP}" -gt 1 ] && [ ! -d "${PREPARED_DIR}" ]; then
    echo "ERROR: FIRST_STEP=${FIRST_STEP} skips the preparation steps, but there is no" >&2
    echo "       ${PREPARED_DIR} to read. Start from step 0." >&2
    exit 1
fi
# PREPARED_DIR is an rm -rf target below when the R stage runs. A mis-set one would otherwise
# aim it somewhere expensive -- data/ibd among them, since that is its parent now and holds
# the raw symlink.
case "${PREPARED_DIR}" in
    "${RAW_DIR}"|"${DATA_DIR}/ibd"|"${DATA_DIR}"|"${PROJECT_ROOT}"|/|"")
        echo "ERROR: PREPARED_DIR is ${PREPARED_DIR}, which is not a directory this script" >&2
        echo "       may remove. It must be a subdirectory of its own." >&2
        exit 1 ;;
esac
if [ "${FIRST_STEP}" -le 1 ]; then
    MISSING_RAW=""
    for TABLE in ${NORMALIZE_TABLES}; do
        [ -f "${RAW_DIR}/RMT23345_${TABLE}.csv" ] || MISSING_RAW="${MISSING_RAW} ${TABLE}"
    done
    if [ -n "${MISSING_RAW}" ]; then
        echo "ERROR: no RMT23345_<table>.csv in ${RAW_DIR} for:${MISSING_RAW}" >&2
        echo "       normalize_RMT23345_datetimes.R reads them by that exact name." >&2
        exit 1
    fi
fi
if [ ! -f "${DATASET_CONFIG}" ]; then
    echo "ERROR: no dataset config at ${DATASET_CONFIG}. Override DATASET_CONFIG." >&2
    exit 1
fi
# Checked here rather than at the point of use. The deletion below happens before the first
# step runs, so a venv that turns out to be missing would otherwise cost the fold directories
# and labels.csv on the way to an error that was knowable up front. Same for Rscript.
if [ "${FIRST_STEP}" -le 2 ]; then
    if [ -n "${R_MODULE}" ]; then
        # `module` is a shell function the login shell's profile defines, and a batch shell is
        # not a login shell -- so it has to be initialized here before it can be called.
        if ! command -v module >/dev/null 2>&1; then
            for INIT in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh \
                        "${LMOD_PKG:-/nonexistent}/init/bash" \
                        "${MODULESHOME:-/nonexistent}/init/bash"; do
                if [ -f "${INIT}" ]; then
                    # shellcheck disable=SC1090
                    . "${INIT}" && break
                fi
            done
        fi
        if ! command -v module >/dev/null 2>&1; then
            echo "ERROR: R_MODULE=${R_MODULE} was given but no 'module' command exists in a" >&2
            echo "       batch shell here, and none of the usual init scripts were found." >&2
            echo "       Set RSCRIPT to the interpreter's full path instead -- 'module load" >&2
            echo "       ${R_MODULE} && command -v Rscript' on a login node will print it." >&2
            exit 1
        fi
        # Unquoted deliberately: R_MODULE is a load order, not one name, and `module load`
        # takes them as separate arguments. No module name contains whitespace.
        # shellcheck disable=SC2086
        module load ${R_MODULE} || {
            echo "ERROR: module load ${R_MODULE} failed. 'module avail r' on a login node" >&2
            echo "       lists the names this cluster actually has, and the stack module has" >&2
            echo "       to come first in R_MODULE." >&2
            exit 1; }
    fi
    if ! command -v "${RSCRIPT}" >/dev/null 2>&1; then
        echo "ERROR: ${RSCRIPT} not found, so Stage A cannot run." >&2
        echo "" >&2
        echo "       On a login node, find R and pick whichever of these works:" >&2
        echo "           module avail R          # or: module spider R" >&2
        echo "           command -v Rscript      # if it is already on PATH there" >&2
        echo "" >&2
        echo "       Then either name the modules, in load order, so the job loads them:" >&2
        echo "           R_MODULE='<stack> <r>' ./prepare_data.sh" >&2
        echo "       or give the interpreter's full path, which needs no module at all:" >&2
        echo "           RSCRIPT=/path/to/Rscript ./prepare_data.sh" >&2
        echo "" >&2
        echo "       Either variable reaches this job through --export=ALL, so setting it on" >&2
        echo "       the command line in front of ./prepare_data.sh is enough." >&2
        exit 1
    fi
    echo "  Rscript: $(command -v "${RSCRIPT}")"
    if [ ! -f "${IBDDATAPREP_DIR}/renv.lock" ]; then
        echo "ERROR: no renv.lock in ${IBDDATAPREP_DIR}, so Stage A has no locked library." >&2
        exit 1
    fi
fi
for VENV in "${IBD_VENV}" "${TRANSEHR2_VENV}"; do
    if [ ! -f "${VENV}/bin/activate" ]; then
        echo "ERROR: no virtualenv at ${VENV}." >&2
        echo "       Set IBD_VENV and TRANSEHR2_VENV to the environments that exist, or point" >&2
        echo "       both at one that carries pandas, pyyaml and torch together." >&2
        exit 1
    fi
done

activate() {
    # $1 = venv root. Deactivates whatever is active first, so switching between the two
    # environments mid-job cannot leave the first one's site-packages ahead of the second's.
    local venv="$1"
    # Preflight has already checked both of these. This catches one going away mid-job.
    if [ ! -f "${venv}/bin/activate" ]; then
        echo "ERROR: ${venv} was there at preflight and is not now." >&2
        exit 1
    fi
    if [ -n "${VIRTUAL_ENV:-}" ]; then
        deactivate 2>/dev/null || true
    fi
    # shellcheck disable=SC1091
    source "${venv}/bin/activate"
    echo "  venv:   ${venv}"
    echo "  python: $(command -v python)"
}

# ------------------------------------------------------------------
# What a re-run invalidates, listed before anything is touched.
# ------------------------------------------------------------------

N_FOUND=0
report_target() {
    # $1 = path. Reports it if it exists; the caller counts.
    if [ -e "$1" ]; then
        echo "    $1"
        return 0
    fi
    return 1
}

echo ""
echo "============================================================"
if [ "${DRY_RUN}" -ne 0 ]; then
    echo "Artifacts that WOULD be removed (dry run, nothing is deleted)"
else
    echo "Removing artifacts invalidated by re-running the pipeline"
fi
echo "============================================================"

echo ""
echo "  Fold row indices and their standardization statistics -- split.py reassigns every"
echo "  patient, so a fold directory written before this run describes a partition that will"
echo "  not exist afterwards:"
for FOLD_DIR in "${DATA_DIR}"/fold*; do
    report_target "${FOLD_DIR}" && N_FOUND=$((N_FOUND + 1))
done
report_target "${DATA_DIR}/labels.csv" && N_FOUND=$((N_FOUND + 1))

echo ""
echo "  Extracted arrays -- extract_data.py clears this directory itself on every run, so it"
echo "  is listed for completeness rather than removed here:"
report_target "${DATA_DIR}/extracted" && N_FOUND=$((N_FOUND + 1))

if [ "${CLEAN_ROOT}" -ne 0 ]; then
    echo ""
    echo "  Per-patient root (CLEAN_ROOT=1). Renamed aside rather than walked: a rename is one"
    echo "  metadata operation whatever the tree holds, and this one holds ~137k directories."
    report_target "${DATA_DIR}/root" && N_FOUND=$((N_FOUND + 1))
fi

if [ "${FIRST_STEP}" -le 1 ]; then
echo ""
echo "  Prepared tables -- the R stage rewrites them, and removing the directory first is"
echo "  what stops a table dropped from PREPARE_TABLES surviving as a stale file:"
report_target "${PREPARED_DIR}" && N_FOUND=$((N_FOUND + 1))
STALE_NORMALIZED=""
else
echo ""
echo "  Normalization intermediates whose prepared counterpart already exists. These are"
echo "  ${PREPARED_DIR}/*_normalized.csv, superseded by the *_prepared.csv derived from them,"
echo "  and reclaiming them is pure disk saving. One with no prepared counterpart is left"
echo "  alone: it is the input a preparation step has not consumed yet."
STALE_NORMALIZED=""
for NORMALIZED in "${PREPARED_DIR}"/RMT23345_*_normalized.csv; do
    [ -f "${NORMALIZED}" ] || continue
    if [ -f "${NORMALIZED%_normalized.csv}_prepared.csv" ]; then
        report_target "${NORMALIZED}" && N_FOUND=$((N_FOUND + 1))
        STALE_NORMALIZED="${STALE_NORMALIZED} ${NORMALIZED}"
    fi
done
[ -n "${STALE_NORMALIZED}" ] || echo "    (none)"
fi

echo ""
echo "  NOT removed, NOT written, and NOT regenerated here:"
echo "    ${RAW_DIR}/RMT23345_*.csv   (the raw extract; read-only to this pipeline)"
echo "    ${DATA_DIR}/lookup_tables   (embed.py; see the header -- re-embed after a root rebuild)"
echo "    ${TRANSEHR2_DIR}/models     (trained weights and evaluation YAMLs)"
echo "    ${DATA_DIR}/resources       (code dictionaries and ClinVec)"
echo ""
echo "${N_FOUND} existing artifact(s) found."

if [ "${DRY_RUN}" -ne 0 ]; then
    echo ""
    echo "Dry run: nothing was deleted and no pipeline step ran."
    echo "DATAPREP_DRY_RUN_OK"
    exit 0
fi

# ------------------------------------------------------------------
# Deletion. The root is renamed aside and purged on the way out, whichever way out that is.
# ------------------------------------------------------------------

STAGED_FOR_DELETION=""
PURGE_DECISION="keep"

purge_staged() {
    [ -n "${STAGED_FOR_DELETION}" ] || return 0
    if [ "${PURGE_DECISION}" = "purge" ]; then
        echo ""
        echo "Purging ${STAGED_FOR_DELETION} at $(date)"
        rm -rf -- "${STAGED_FOR_DELETION}"
        echo "Purged at $(date)"
    else
        echo ""
        echo "The pipeline did not complete. The previous root is still at:"
        echo "    ${STAGED_FOR_DELETION}"
        echo "Move it back to ${DATA_DIR}/root to restore, or delete it once you are satisfied."
    fi
}
# One caller, on the way out. The rename happens early and there are a dozen exits after it --
# a missing venv, a failed step, a SIGTERM at the time limit. Routing them individually is how
# one gets missed and an hours-old root is deleted after a failure, or kept forever after a
# success.
trap 'purge_staged' EXIT

for FOLD_DIR in "${DATA_DIR}"/fold*; do
    [ -d "${FOLD_DIR}" ] && rm -rf -- "${FOLD_DIR}"
done
rm -f -- "${DATA_DIR}/labels.csv"

if [ "${FIRST_STEP}" -le 1 ]; then
    rm -rf -- "${PREPARED_DIR}"
else
    # Word-split deliberately: the paths were accumulated into one string above and no
    # RMT23345 table name contains whitespace.
    # shellcheck disable=SC2086
    [ -n "${STALE_NORMALIZED}" ] && rm -f -- ${STALE_NORMALIZED}
fi

if [ "${CLEAN_ROOT}" -ne 0 ] && [ -d "${DATA_DIR}/root" ]; then
    STAGED_FOR_DELETION="${DATA_DIR}/root.staged.${SLURM_JOB_ID:-$$}"
    mv -- "${DATA_DIR}/root" "${STAGED_FOR_DELETION}" || exit 1
    echo ""
    echo "Renamed the previous root aside as ${STAGED_FOR_DELETION}"
fi

# ------------------------------------------------------------------
# The pipeline.
# ------------------------------------------------------------------

run_step() {
    # $1 = step number, $2 = description, rest = command.
    local step="$1" description="$2"
    shift 2
    if [ "${step}" -lt "${FIRST_STEP}" ]; then
        echo ""
        echo "Step ${step}: ${description} -- SKIPPED (FIRST_STEP=${FIRST_STEP})"
        return 0
    fi
    echo ""
    echo "============================================================"
    echo "Step ${step}: ${description} at $(date)"
    echo "============================================================"
    local step_start=$SECONDS
    "$@"
    local status=$?
    local duration=$((SECONDS - step_start))
    printf "Step %s finished in %02d:%02d:%02d with status %s\n" \
        "${step}" $((duration / 3600)) $((duration % 3600 / 60)) $((duration % 60)) "${status}"
    if [ ${status} -ne 0 ]; then
        echo "ERROR: step ${step} (${description}) failed; later steps consume its output," \
             "so stopping."
        exit ${status}
    fi
}

# Stage A. Working directory is the IBDdataprep repository root for two reasons: .Rprofile
# activates renv from there, and every prepare script sources helper_functions.R by a path
# relative to it.
cd "${IBDDATAPREP_DIR}" || exit 1

run_step 0 "Normalizing datetimes" \
    "${RSCRIPT}" IBDdataprep/normalize_RMT23345_datetimes.R \
        --input_dir "${RAW_DIR}" \
        --output_dir "${PREPARED_DIR}"

# One script per table rather than a loop inside one: they take a file in and a file out, and
# each has its own arguments. DAD writes RMT23345_SCU_prepared.csv beside its output, so SCU is
# not in PREPARE_TABLES. LAB additionally emits the variable_properties fragment step 2 folds
# in, which is why it cannot be the last thing to run before the Python half.
prepare_table() {
    # $1 = table name. Echoes nothing; exits non-zero on failure, which run_step reports.
    local table="$1"
    local in_path="${PREPARED_DIR}/RMT23345_${table}_normalized.csv"
    local out_path="${PREPARED_DIR}/RMT23345_${table}_prepared.csv"
    local script="IBDdataprep/prepare_RMT23345_${table}.R"

    [ -f "${script}" ] || { echo "ERROR: no ${script}" >&2; return 1; }
    if [ ! -f "${in_path}" ]; then
        echo "ERROR: ${in_path} was not written by step 0." >&2
        return 1
    fi

    echo ""
    echo "  ${table}:"
    case "${table}" in
        # -r for the tables that map codes through data/resources.
        AMB|CLM|DAD|PIN)
            "${RSCRIPT}" "${script}" -i "${in_path}" -o "${out_path}" \
                -r "${RESOURCES_DIR}" || return 1 ;;
        # LAB writes the variable_properties fragment as well as the table.
        LAB)
            "${RSCRIPT}" "${script}" -i "${in_path}" -o "${out_path}" \
                -p "${PREPARED_DIR}/variable_properties_LAB.yaml" || return 1 ;;
        *)
            "${RSCRIPT}" "${script}" -i "${in_path}" -o "${out_path}" || return 1 ;;
    esac
}

prepare_all() {
    local table
    for table in ${PREPARE_TABLES}; do
        prepare_table "${table}" || return 1
    done
}

run_step 1 "Preparing the per-table CSVs (${PREPARE_TABLES})" prepare_all

# Normalization is all-or-nothing -- one call, all seven tables -- so a table with no prepare
# script still gets a normalized file. Nothing will ever consume it, and the reclamation rule
# above only removes intermediates whose prepared counterpart exists, so without this it would
# sit on disk permanently.
if [ "${FIRST_STEP}" -le 1 ]; then
    for TABLE in ${NORMALIZE_TABLES}; do
        case " ${PREPARE_TABLES} " in
            *" ${TABLE} "*) continue ;;
        esac
        UNCONSUMED="${PREPARED_DIR}/RMT23345_${TABLE}_normalized.csv"
        if [ -f "${UNCONSUMED}" ]; then
            echo "  ${TABLE} is normalized but not prepared; removing ${UNCONSUMED}"
            rm -f -- "${UNCONSUMED}"
        fi
    done
fi

# Folded in before the contract check, not after: check_contract.py and build_root.py both read
# variable_properties.yaml, and the LAB features only reach it here.
run_step 2 "Folding the LAB variable properties in" \
    "${RSCRIPT}" IBDdataprep/merge_variable_properties.R \
        --properties_path "${PREPARED_DIR}/variable_properties_LAB.yaml" \
        --target_path "${VARIABLE_PROPERTIES}"

echo ""
echo "Activating the IBDdataprep environment for steps 3-6."
activate "${IBD_VENV}"
export PYTHONPATH="${PYTHONPATH:-}:${IBDDATAPREP_DIR}"

# Step 3 before the expensive Python work: it reads two YAML files and nothing else, and it
# catches the config/properties mismatch that would otherwise surface part-way through
# extraction, hours in. It has to follow step 2, which is what puts the LAB features in the
# file it checks.
run_step 3 "Checking the feature contract" \
    python -m IBDdataprep.check_contract \
        --config "${DATASET_CONFIG}" \
        --variable-properties "${VARIABLE_PROPERTIES}"

run_step 4 "Building the per-patient root" \
    python -m IBDdataprep.build_root \
        --prepared "${PREPARED_DIR}" \
        --root "${DATA_DIR}/root" \
        --config "${DATASET_CONFIG}" \
        --log "${SLURM_SUBMIT_DIR:-$(pwd)}/log/build_root_${SLURM_JOB_ID:-$$}.log"

run_step 5 "Validating the root" \
    python -m IBDdataprep.validate_root \
        --root "${DATA_DIR}/root" \
        --log "${SLURM_SUBMIT_DIR:-$(pwd)}/log/validate_root_${SLURM_JOB_ID:-$$}.log"

# ibd_onset.py and external_cause.py each read the whole root rather than
# stays.csv alone, so this step is no longer the cheap one it was. They run
# here rather than in step 4 because both predicates -- which codes name IBD,
# and which name an external cause -- are label policy: changing either must
# cost a 'labels' run, not a root rebuild.
run_step 6 "Building labels and partitioning into folds" \
    bash -c "python -m IBDdataprep.ibd_onset \
                 --root '${DATA_DIR}/root' \
                 --output '${DATA_DIR}/ibd_onset.csv' \
                 --resources '${RESOURCES_DIR}' \
                 --n-workers '${N_WORKERS}' \
             && python -m IBDdataprep.external_cause \
                 --root '${DATA_DIR}/root' \
                 --output '${DATA_DIR}/external_cause.csv' \
                 --resources '${RESOURCES_DIR}' \
                 --n-workers '${N_WORKERS}' \
             && python -m IBDdataprep.build_labels \
                 --root '${DATA_DIR}/root' --labels '${DATA_DIR}/labels.csv' \
                 --ibd-onset '${DATA_DIR}/ibd_onset.csv' \
                 --external-cause '${DATA_DIR}/external_cause.csv' \
             && python -m IBDdataprep.split \
                 --labels '${DATA_DIR}/labels.csv' --output '${DATA_DIR}'"

echo ""
echo "Activating the TransEHR2 environment for steps 7 onward."
activate "${TRANSEHR2_VENV}"
cd "${TRANSEHR2_DIR}" || exit 1
export PYTHONPATH="${PYTHONPATH:-}:${TRANSEHR2_DIR}"

run_step 7 "Extracting the cohort into arrays" \
    python scripts/extract_data.py "${DATASET_CONFIG}" \
        --data_dir "${DATA_DIR}" \
        --n_workers "${N_WORKERS}"

if [ "${CALIBRATE}" -ne 0 ]; then
    run_step 8 "Measuring what the outlier rules would remove" \
        python scripts/calibrate_outlier_filter.py "${DATA_DIR}/extracted" \
            --report "${SLURM_SUBMIT_DIR:-$(pwd)}/log/outlier_calibration_${SLURM_JOB_ID:-$$}.txt"
else
    echo ""
    echo "Step 8: outlier calibration -- SKIPPED (CALIBRATE=0)"
fi

PURGE_DECISION="purge"

echo ""
echo "============================================================"
echo "Pipeline finished at $(date)"
echo "============================================================"
echo "Arrays:  ${DATA_DIR}/extracted"
echo "Folds:   ${DATA_DIR}/fold*"
echo ""
echo "The lookup tables under ${DATA_DIR}/lookup_tables were NOT rebuilt. text_values index"
echo "text_strings.pkl in the order this extraction assigned, so if the unique-string set"
echo "changed the existing text_embeddings.npy is wrong without being the wrong shape. Rebuild:"
echo "    sbatch SLURM/slurm_embed.sh"
echo "DATAPREP_RUN_OK"
