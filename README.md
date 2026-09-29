# TransEHR2-IBDreadmission

Code for a model that predicts personalized readmission probability
distributions for patients with inflammatory bowel disease. It uses the
TransEHR2 framework to learn latent representations of longitudinal medical
records, and predicts readmission probabilities from those representations.

## About TransEHR2

TransEHR, originally presented by [Xu *et al.*](https://proceedings.mlr.press/v225/xu23a/xu23a.pdf), is a transformer neural network-based model that learns representations of medical record timeseries which can be used as input for downstream medical prediction tasks. TransEHR consists of a generator network, a discriminator network, and a transformer Hawkes process network. During self-supervised pre-training, the generator learns to simulate the values of randomly masked records. The discriminator network learns to identify which records are simulated and which ones are original. The transformer Hawkes process learns the temporal dynamics of different types of features captured in the medical records. TransEHR is pretrained to minimize the weighted sum of losses from these three networks. Finetuning is fully supervised and aims to maximize performance on a given downstream prediction task.

TransEHR2 improves upon the original TransEHR model. It supports additional data types for input, namely: vector-valued, categorical, ordinal, text and drug features. TransEHR2 also corrects known errors in Xu *et al.*'s loss calculations for the transformer Hawkes process. The loss is reformulated to express the joint conditional likelihood of the next observed event type(s) and their timestamp, rather than timestamp only, under the learned model.

## Data availability

The health records used to train and evaluate the model are restricted and,
in the interest of protecting patient privacy, cannot be publicly
distributed. Several of the supporting reference files — code-set
dictionaries, drug vocabularies, feature definitions — are used under
license and cannot be redistributed here either. [`manifest.csv`](manifest.csv)
lists every file the pipeline expects, with its checksum and where it came
from; you will need to obtain the licensed ones yourself.

## Installation

Clone the repository and create a virtual environment (optional but
advisable).

```shell
git clone https://github.com/phairlab/TransEHR2-IBDreadmission.git TransEHR2 && cd TransEHR2
python -m venv venv/TransEHR2
```

```shell
source venv/TransEHR2/bin/activate
pip install -r requirements.txt
deactivate
```

`requirements.txt` does not install NVIDIA's `transformer-engine`. If you
want fp8 precision, build and install it before the other requirements.

Text features require authorization to use Meta's Llama model. TransEHR2
obtains it through HuggingFace; the exact version is set by `LLM_NAME` in
[`TransEHR2/constants.py`](TransEHR2/constants.py). Put your HuggingFace
token in a `.env` file at the repository root as `HF_READ_TOKEN`.

## Data

Data files live **outside** the repository, by default at `../data/`
relative to the project root. Create the directory and fetch what can be
fetched:

```shell
bash scripts/setup_data.sh          # or: bash scripts/setup_data.sh /path/to/data
```

Set `SHARED_DATA_ROOT` in your shell profile to avoid passing the path each
time. Each file is verified against its checksum; the script is safe to
re-run. To register a file you have added or changed:

```shell
bash scripts/update_manifest.sh <path/relative/to/data/root> <source> <source_type>
```

`source_type` is one of `local_copy`, `local_symlink`, `download`, or
`build`. A `build` entry is produced locally rather than fetched — its
`source` column is the command that makes it — so `setup_data.sh` verifies
one when it is present and reports rather than fetches when it is not. New
files must be added to `manifest.csv` by hand, which keeps a script from
registering data that cannot legally be shared.

### 1. Prepare the records

The raw records are turned into per-patient CSVs by
[IBDdataprep](https://github.com/phairlab/IBDdataprep), a separate package
with its own environment. It writes `data/root/{PATID}/` — one
`timeseries.csv`, `stays.csv` and `drugs.csv` per patient — then a
cohort-wide `data/labels.csv` and the cross-validation folds in
`data/fold{i}/`. See that repository's README for the commands.

The row order of `labels.csv` is the study's canonical episode order.
Everything downstream indexes into it, so the folds, the extracted arrays
and any saved predictions all address the same rows.

### 2. Extract to arrays

```shell
python scripts/extract_data.py TransEHR2/configs/datasets/RMT23345.yaml
```

Reads `data/root/` and `data/labels.csv` and writes `data/extracted/` — one
cohort-wide set of memory-mapped `.npy` arrays, plus one set of
standardization statistics per fold, computed over that fold's training rows
alone. Extraction runs **once for the whole cohort**, not once per fold: a
fold is a set of row indices into these arrays.

`data/extracted/` is cleared and rewritten on every run.

### 3. Build the lookup tables

```shell
python scripts/embed.py TransEHR2/configs/datasets/RMT23345.yaml
```

Text and drug values are not stored per timestep. Each is an `int32` row
index into a table built once for the whole study, which is what makes the
text feasible: a few million unique strings are embedded once instead of
once per fold per timestep. `embed.py` writes `data/lookup_tables/` —

- `text_embeddings.npy`, the LLM's mean-pooled output per unique string;
- `text_tokens.npy`, the token ids behind them, kept for token-level
  attribution;
- `drug_embeddings.npy`, the ClinVec ATC vectors plus an all-zero row that
  unused drug slots point at.

`--tables drug` builds only the drug table and loads no LLM; `--tables text`
builds only the text pair. The tables are deliberately **not** in
`data/extracted/`, so re-extracting does not delete an artifact that took
GPU-hours to produce. Re-extracting does invalidate the text pair, though —
it reassigns the string indices — and the loader refuses a table that no
longer matches, so rebuild it after any re-extraction.

## Running an experiment

```shell
python scripts/run_experiment.py \
    TransEHR2/configs/datasets/RMT23345.yaml \
    TransEHR2/configs/experiments/experiment1_baseline.yaml
```

An experiment is pretraining, finetuning, and evaluation on a held-out test
set, repeated per fold. [`tune_hyperparameters.py`](scripts/tune_hyperparameters.py)
takes the same two config arguments and sweeps one pretraining
hyperparameter at a time instead of training once.

Parameters live in `TransEHR2/configs/`: one file per dataset under
`datasets/`, one per experiment under `experiments/`. **The paths in the
shipped configs are absolute and point at the original author's machine** —
edit `DATA_DIR`, `VARIABLE_PROPERTIES_PATH`, `CLINVEC_PATH` and `MODEL_DIR`
before your first run.

### One GPU per process

There is no data parallelism: a run occupies one GPU, and the way to use
several is to give each one its own fold, which share nothing but the
memory-mapped arrays.

```shell
for f in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$f python scripts/run_experiment.py \
        TransEHR2/configs/datasets/RMT23345.yaml \
        TransEHR2/configs/experiments/experiment1_baseline.yaml \
        --folds fold$f &
done; wait
```

`--device` picks the card directly when `CUDA_VISIBLE_DEVICES` is not how
the scheduler hands them out.

## What the model predicts

One outcome, fitted as competing risks with DeepHit on a discrete grid of
time bins: unplanned readmission, death, and out-migration. They compete
because a patient who dies cannot be readmitted, so a model scoring them
separately would describe a cohort that cannot exist.

Out-migration is a cause rather than censoring because leaving the province
is a period of missing data, not the end of the record, and this data is
already incomplete — it holds no trace of most of a patient's contacts with
the health system. Treating a departure as censoring would assume it says
nothing about readmission risk. `EVENT_TYPE` 0, administrative censoring, is
the only thing still treated as censoring.

`MODELLED_CAUSES` in the experiment config decides which `EVENT_TYPE`s get a
cause axis; anything left out of it falls back to censoring, so the
two-cause arm costs a config line rather than a code change.

The grid is `TIME_GRID_CUTS_DAYS` in the experiment config, six bins by
default: 30, 60 and 90 days, then 1, 3 and 5 years. The last cut is the
horizon; an event past it is an episode observed through the whole grid
without one.

The MIMIC fork's three tasks are gone. In-hospital mortality does not exist
in data that follows patients after discharge, and length of stay and
phenotyping were never IBD questions.

### The Hawkes process is optional

`USE_THP: False` removes the Transformer Hawkes process from pretraining.
Its log-likelihood integrates an intensity over inter-event gaps, so a broad
dynamic range of timestamps makes it diverge; that is the form of the loss,
not its scale, and no reweighting fixes it. With the switch off the event
stream still reaches the classifier — the event encoder simply learns during
finetuning rather than being handed pretrained weights.

## Scoring a trained model

```shell
python scripts/dump_finetuned_predictions.py \
    TransEHR2/configs/datasets/RMT23345.yaml \
    TransEHR2/configs/experiments/experiment1_baseline.yaml \
    experiment1_baseline --model_dir ./models
python scripts/evaluate_finetuned_predictions.py experiment1_baseline \
    --model_dir ./models \
    --experiment_config TransEHR2/configs/experiments/experiment1_baseline.yaml
```

The first writes each episode's cumulative incidence at every cut, per
cause; the second reads those CSVs back and writes one YAML per split with
the per-fold metrics. Scoring from files rather than from the training loop
means a change of metric costs a re-read rather than a re-fit.

## Tests

```shell
python -m pytest TransEHR2/tests
```

The suite needs no patient data. Most of it runs against a small cohort the
tests write themselves; the end-to-end tests run against IBDdataprep's
synthetic fixture and skip if it has not been generated.
