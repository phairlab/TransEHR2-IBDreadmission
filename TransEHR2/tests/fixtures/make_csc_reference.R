# Reference values for test_csc.py: riskRegression's own CSC predictions on
# a synthetic cohort, beside the parameters TransEHR2.csc rebuilds them from.
#
# Fits riskRegression::CSC to three competing causes, then writes, to
# --out_dir:
#   baseline_hazard.csv  time_days, cumhaz_<cause>: uncentred baseline
#                        cumulative hazards at every event time;
#   lp.csv               idx, time_to_event (minutes), event_type and
#                        lp_<cause>, x beta uncentred;
#   risk.csv             idx, cause, product_limit, time, risk: the
#                        absolute risk predict.CauseSpecificCox returns,
#                        with product.limit TRUE and FALSE.
# The parameters are exported the way fit_csc_baseline.R must export them.
#
# Usage, from the repository root (needs riskRegression):
#   Rscript TransEHR2/tests/fixtures/make_csc_reference.R

suppressPackageStartupMessages({
  library(data.table)
  library(optparse)
  library(prodlim)
  library(riskRegression)
  library(survival)
})

options <- parse_args(OptionParser(option_list = list(
  make_option("--out_dir", type = "character",
              default = "TransEHR2/tests/fixtures/csc_reference",
              help = "Directory to write the reference CSVs into"),
  make_option("--seed", type = "integer", default = 20261016L,
              help = "Seed of the synthetic cohort")
)))

causes <- c("readmission", "death", "out_migration")
set.seed(options$seed)
n <- 400
d <- data.table(x1 = rnorm(n), x2 = rbinom(n, 1, 0.5))
latent <- cbind(
  rexp(n, 0.02 * exp(0.5 * d$x1 + 0.4 * d$x2)),
  rexp(n, 0.008 * exp(-0.3 * d$x1 + 0.7 * d$x2)),
  rexp(n, 0.004 * exp(0.2 * d$x1)),
  runif(n, 0, 150))
# Rounded, so that event times tie with each other and with censoring.
d[, time := round(apply(latent, 1, min), 1) + 0.1]
d[, event := apply(latent, 1, which.min) %% 4L]

fit <- CSC(Hist(time, event) ~ x1 + x2, data = d)
model_of <- function(k) fit$models[[match(as.character(k), fit$causes)]]

jump <- sort(unique(d$time[d$event > 0]))
baseline <- data.table(time_days = jump)
invisible(lapply(seq_along(causes), function(k) {
  cumhaz <- predictCox(model_of(k), times = jump, centered = FALSE,
                       type = "cumhazard")$cumhazard
  set(baseline, j = paste0("cumhaz_", causes[k]), value = as.vector(cumhaz))
}))

patients <- data.table(idx = seq_len(n) - 1L,
                       time_to_event = d$time * 1440, event_type = d$event)
invisible(lapply(seq_along(causes), function(k) {
  lp <- predict(model_of(k), newdata = d, type = "lp", reference = "zero")
  set(patients, j = paste0("lp_", causes[k]), value = as.vector(lp))
}))

# Before, on and between event times, short of the last one, past which
# riskRegression returns NA.
query <- sort(unique(c(jump[c(1, 5, 50)], (jump[20] + jump[21]) / 2,
                       quantile(jump, c(0.1, 0.5, 0.9), names = FALSE),
                       jump[1] / 2)))
risk <- rbindlist(unlist(lapply(c(TRUE, FALSE), function(pl) {
  lapply(seq_along(causes), function(k) {
    absrisk <- predict(fit, newdata = d, times = query, cause = k,
                       product.limit = pl)$absRisk
    data.table(idx = rep(patients$idx, times = length(query)),
               cause = k, product_limit = pl,
               time = rep(query, each = n), risk = as.vector(absrisk))
  })
}), recursive = FALSE))

dir.create(options$out_dir, recursive = TRUE, showWarnings = FALSE)
fwrite(baseline, file.path(options$out_dir, "baseline_hazard.csv"))
fwrite(patients, file.path(options$out_dir, "lp.csv"))
fwrite(risk, file.path(options$out_dir, "risk.csv"))
cat("Wrote", nrow(baseline), "event times,", n, "patients and",
    nrow(risk), "reference risks to", options$out_dir, "\n")
