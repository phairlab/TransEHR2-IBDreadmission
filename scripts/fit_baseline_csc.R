# Fit the Part 1 cause-specific Cox baseline and export its parameters.
#
# For each fold in FOLDS of the experiment config, fits riskRegression::CSC,
# one Cox model per modelled cause, on the train partition, and writes under
# {MODEL_DIR}/{EXPERIMENT_NAME}/fold{i}/competing_risks/ what
# TransEHR2/csc.py rebuilds the cumulative incidences from:
#   competing_risks_{split}_csc_lp.csv, for split train, val and test:
#       idx, time_to_event (minutes) and event_type as labels.csv has them,
#       then lp_<cause>, x beta_k uncentred;
#   competing_risks_csc_baseline_hazard.csv: time_days, then cumhaz_<cause>,
#       the uncentred baseline cumulative hazard at every training event time
#       of a modelled cause.
# scripts/dump_baseline_csc_predictions.py writes the grid dumps from them.
#
# A partition is its fold{i}_{split}.csv listfile under the dataset config's
# DATA_DIR, joined to labels.csv on (PATID, STAY_INDEX), which gives idx, the
# labels.csv row; the baseline features (IBDdataprep's lace.py) are joined on
# idx. Time is fitted in days.
# An EVENT_TYPE that is not a modelled cause is censoring, and so is any
# outcome at or past the horizon, the last of TIME_GRID_CUTS_DAYS (user,
# 2026-10-18): follow-up is cut there, as TimeGrid.discretize cuts it for
# the DeepHit models, so all three models train on the same outcomes.
#
# The encoding is the plan's (Features), fitted on train and applied to all
# three partitions:
#   LINEAR_FEATURES enter as they are, and must not be missing unless they
#       are also MEDIAN_FILL_FEATURES;
#   MEDIAN_FILL_FEATURES have a missing value filled with the train median of
#       the observed ones, and gain a 0/1 "<feature>_missing";
#   CATEGORICAL_FEATURES enter as treatment dummies against their first
#       level, observed values sorted, then "missing" where one occurs.
# A missing indicator or level exists only if train has a missing value, and
# a level exists only if train has it: a val or test row that needs one train
# lacks has no coefficient, so the run stops rather than guess one.
#
# The export follows TransEHR2/tests/fixtures/make_csc_reference.R:
# predict(type = "lp", reference = "zero") and
# predictCox(centered = FALSE, type = "cumhazard"), the same centring on both.
# Doubles are written with 17 significant digits, because fwrite's 15 do not
# round-trip and csc.py compares a patient's own event time with time_days
# exactly.
#
# Prints aggregates only.
#
# Usage, from the repository root:
#   Rscript scripts/fit_baseline_csc.R \
#       [--config TransEHR2/configs/experiments/part1_csc.yaml] \
#       [--dataset_config TransEHR2/configs/datasets/RMT23345.yaml]

suppressPackageStartupMessages({
  library(data.table)
  library(optparse)
  library(prodlim)
  library(riskRegression)
  library(survival)
  library(yaml)
})

options <- parse_args(OptionParser(option_list = list(
  make_option("--config", type = "character",
              default = "TransEHR2/configs/experiments/part1_csc.yaml",
              help = "Experiment config [default: %default]"),
  make_option("--dataset_config", type = "character",
              default = "TransEHR2/configs/datasets/RMT23345.yaml",
              help = "Dataset config giving DATA_DIR [default: %default]")
)))

TASK <- "competing_risks"
SPLITS <- c("train", "val", "test")
MINUTES_PER_DAY <- 1440
# build_labels.py's EVENT_TYPE coding, as TransEHR2/survival.py's CAUSE_CODES.
CAUSE_CODES <- c(readmission = 1L, death = 2L, out_migration = 3L)
MISSING_LEVEL <- "missing"

cfg <- read_yaml(options$config)
causes <- unlist(cfg$MODELLED_CAUSES)
unknown <- setdiff(causes, names(CAUSE_CODES))
if (length(unknown) > 0) stop("unknown cause(s): ", toString(unknown))
codes <- CAUSE_CODES[causes]
horizon <- tail(unlist(cfg$TIME_GRID_CUTS_DAYS), 1)
linear <- unlist(cfg$LINEAR_FEATURES)
filled <- unlist(cfg$MEDIAN_FILL_FEATURES)
categorical <- unlist(cfg$CATEGORICAL_FEATURES)
stray <- setdiff(filled, linear)
if (length(stray) > 0) stop("not LINEAR_FEATURES: ", toString(stray))
unfilled <- setdiff(linear, filled)
data_dir <- read_yaml(options$dataset_config)$DATA_DIR

exact <- function(x) sprintf("%.17g", x)

labels <- fread(file.path(data_dir, "labels.csv"),
                select = c("PATID", "STAY_INDEX", "TIME_TO_EVENT",
                           "EVENT_TYPE"),
                colClasses = list(character = "PATID"))
labels[, idx := .I - 1L]

features <- fread(file.path(data_dir, cfg$BASELINE_FEATURES_FILE),
                  select = c("idx", linear, categorical),
                  colClasses = list(character = categorical),
                  na.strings = "")
if (anyDuplicated(features$idx)) stop("baseline features repeat an idx")

# One partition of one fold: outcomes and raw features, in idx order.
read_partition <- function(fold, split) {
  path <- file.path(data_dir, sprintf("fold%d", fold),
                    sprintf("fold%d_%s.csv", fold, split))
  listed <- fread(path, select = c("PATID", "STAY_INDEX", "TIME_TO_EVENT",
                                   "EVENT_TYPE"),
                  colClasses = list(character = "PATID"))
  rows <- labels[listed, on = c("PATID", "STAY_INDEX")]
  if (anyNA(rows$idx)) {
    stop(path, ": ", sum(is.na(rows$idx)), " rows not in labels.csv")
  }
  if (any(rows$TIME_TO_EVENT != rows$i.TIME_TO_EVENT |
            rows$EVENT_TYPE != rows$i.EVENT_TYPE)) {
    stop(path, " disagrees with labels.csv; are they from different runs?")
  }
  absent <- sum(!rows$idx %in% features$idx)
  if (absent > 0) stop(path, ": ", absent, " episodes have no features")
  rows <- features[rows[, .(idx, TIME_TO_EVENT, EVENT_TYPE)], on = "idx"]
  setorder(rows, idx)
  rows
}

# The encoding's train statistics: fill medians and category levels.
fit_encoding <- function(train) {
  medians <- sapply(filled, function(f) median(train[[f]], na.rm = TRUE))
  if (anyNA(medians)) {
    stop("no observed train value for ",
         toString(filled[is.na(medians)]))
  }
  list(
    medians = medians,
    indicated = filled[sapply(filled, function(f) anyNA(train[[f]]))],
    levels = lapply(setNames(categorical, categorical), function(f) {
      observed <- sort(unique(na.omit(train[[f]])), method = "radix")
      if (anyNA(train[[f]])) c(observed, MISSING_LEVEL) else observed
    }))
}

# The design matrix of one partition, as a data.table of numeric columns.
encode <- function(rows, encoding, split) {
  absent <- sapply(unfilled, function(f) sum(is.na(rows[[f]])))
  if (any(absent > 0)) {
    stop(split, ": missing values in ",
         toString(paste(unfilled[absent > 0], absent[absent > 0])),
         "; only MEDIAN_FILL_FEATURES may be missing")
  }
  unfillable <- setdiff(filled[sapply(filled, function(f)
    anyNA(rows[[f]]))], encoding$indicated)
  if (length(unfillable) > 0) {
    stop(split, ": ", toString(unfillable),
         " missing, but never missing in train")
  }
  columns <- c(
    lapply(setNames(linear, linear), function(f) {
      value <- as.numeric(rows[[f]])
      if (f %in% filled) fifelse(is.na(value), encoding$medians[[f]], value)
      else value
    }),
    lapply(setNames(encoding$indicated,
                    paste0(encoding$indicated, "_missing")),
           function(f) as.numeric(is.na(rows[[f]]))),
    unlist(lapply(categorical, function(f) {
      value <- fifelse(is.na(rows[[f]]), MISSING_LEVEL, rows[[f]])
      levels <- encoding$levels[[f]]
      unseen <- setdiff(unique(value), levels)
      if (length(unseen) > 0) {
        stop(split, ": ", f, " level(s) absent from train: ",
             toString(unseen))
      }
      lapply(setNames(levels[-1], paste(f, levels[-1], sep = "_")),
             function(level) as.numeric(value == level))
    }), recursive = FALSE))
  design <- as.data.table(columns)
  setnames(design, make.names(names(design), unique = TRUE))
  design
}

# (time in days, event) with follow-up cut at the horizon: an event counts
# only strictly before it, and an unmodelled cause is censoring.
outcome <- function(rows) {
  days <- rows$TIME_TO_EVENT / MINUTES_PER_DAY
  data.table(time = pmin(days, horizon),
             event = fifelse(days < horizon & rows$EVENT_TYPE %in% codes,
                             rows$EVENT_TYPE, 0L))
}

fit_fold <- function(fold) {
  cat(sprintf("\n== fold%d ==\n", fold))
  partitions <- lapply(setNames(SPLITS, SPLITS),
                       function(s) read_partition(fold, s))
  encoding <- fit_encoding(partitions$train)
  designs <- lapply(SPLITS, function(s)
    encode(partitions[[s]], encoding, s))
  names(designs) <- SPLITS

  train <- cbind(outcome(partitions$train), designs$train)
  absent <- causes[!codes %in% train$event]
  if (length(absent) > 0) {
    stop("fold", fold, ": no train events of ", toString(absent))
  }
  formula <- reformulate(names(designs$train),
                         response = "Hist(time, event)")
  fit <- CSC(formula, data = train, ties = cfg$CSC_TIES)
  models <- lapply(codes, function(k)
    fit$models[[match(as.character(k), fit$causes)]])
  coefs <- sapply(models, coef)
  if (anyNA(coefs)) {
    stop("fold", fold, ": coefficient(s) not estimable: ",
         toString(rownames(coefs)[apply(is.na(coefs), 1, any)]))
  }

  out_dir <- file.path(cfg$MODEL_DIR, cfg$EXPERIMENT_NAME,
                       sprintf("fold%d", fold), TASK)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  jump <- sort(unique(train$time[train$event > 0]))
  baseline <- data.table(time_days = exact(jump))
  invisible(lapply(seq_along(causes), function(k) {
    cumhaz <- predictCox(models[[k]], times = jump, centered = FALSE,
                         type = "cumhazard")$cumhazard
    set(baseline, j = paste0("cumhaz_", causes[k]),
        value = exact(as.vector(cumhaz)))
  }))
  fwrite(baseline, file.path(out_dir,
                             sprintf("%s_csc_baseline_hazard.csv", TASK)))

  invisible(lapply(SPLITS, function(s) {
    rows <- partitions[[s]]
    lp <- data.table(idx = rows$idx, time_to_event = rows$TIME_TO_EVENT,
                     event_type = rows$EVENT_TYPE)
    lapply(seq_along(causes), function(k) {
      value <- predict(models[[k]], newdata = designs[[s]], type = "lp",
                       reference = "zero")
      set(lp, j = paste0("lp_", causes[k]), value = exact(as.vector(value)))
    })
    fwrite(lp, file.path(out_dir, sprintf("%s_%s_csc_lp.csv", TASK, s)))
  }))

  counts <- sapply(SPLITS, function(s) {
    event <- outcome(partitions[[s]])$event
    c(n = length(event), censored = sum(event == 0),
      setNames(sapply(codes, function(k) sum(event == k)), causes))
  })
  cat("episodes by partition and outcome, cut at", horizon, "days:\n")
  print(counts)
  cat("train fill medians:",
      paste(filled, encoding$medians, sep = " = ", collapse = ", "), "\n")
  cat("missing values by partition:\n")
  print(sapply(SPLITS, function(s) sapply(c(filled, categorical),
    function(f) sum(is.na(partitions[[s]][[f]])))))
  cat("hazard ratios, exp(beta), by cause:\n")
  colnames(coefs) <- causes
  print(round(exp(coefs), 3))
  cat(sprintf("%d training event times; wrote %s\n", length(jump), out_dir))
}

invisible(lapply(unlist(cfg$FOLDS), fit_fold))
