#!/usr/bin/env Rscript
# lmm_fit.R — Linear Mixed Models with AR(1) for confirmatory analysis
# Usage: Rscript lmm_fit.R <input_csv> <output_csv>

library(nlme)
library(data.table)
library(clubSandwich)

AR1_BOUNDARY <- as.numeric(Sys.getenv("SAFER_AR1_BOUNDARY", "0.999"))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript lmm_fit.R <input_csv> <output_csv>")
}
input_csv  <- args[1]
output_csv <- args[2]

# ---- Read data ----
dat <- fread(input_csv)
message(sprintf("[lmm_fit] Loaded %d rows, %d columns from %s",
                nrow(dat), ncol(dat), input_csv))

features <- sort(unique(dat$feature_name))
message(sprintf("[lmm_fit] %d unique features to fit", length(features)))

# ---- Helpers: fixed-effect inference ----
extract_model_coefs <- function(mod) {
  s <- summary(mod)
  ct <- s$tTable
  out <- data.table(
    term     = rownames(ct),
    estimate = ct[, "Value"],
    se       = ct[, "Std.Error"],
    p_value  = ct[, "p-value"],
    robust_df = NA_real_
  )
  out[, ci_lower := estimate - 1.96 * se]
  out[, ci_upper := estimate + 1.96 * se]
  out[, se := NULL]
  return(out)
}

extract_cr2_coefs <- function(mod, cluster) {
  tests <- clubSandwich::coef_test(
    mod, vcov = "CR2", cluster = cluster, test = "Satterthwaite"
  )
  intervals <- clubSandwich::conf_int(
    mod, vcov = "CR2", cluster = cluster, test = "Satterthwaite"
  )
  if (!identical(as.character(tests$Coef), as.character(intervals$Coef))) {
    stop("clubSandwich coefficient and confidence-interval order disagree")
  }
  data.table(
    term = as.character(tests$Coef),
    estimate = tests$beta,
    ci_lower = intervals$CI_L,
    ci_upper = intervals$CI_U,
    p_value = tests$p_Satt,
    robust_df = tests$df_Satt
  )
}

# ---- Helper: extract AR(1) rho ----
extract_rho <- function(mod) {
  tryCatch({
    rho <- coef(mod$modelStruct$corStruct, unconstrained = FALSE)
    as.numeric(rho)
  }, error = function(e) NA_real_)
}

fit_lme <- function(
  sub,
  random_slope,
  use_ar1,
  method = "REML",
  include_interaction = TRUE
) {
  random_form <- if (random_slope) {
    ~ time_to_event | episode_id
  } else {
    ~ 1 | episode_id
  }
  correlation <- if (use_ar1) corAR1(form = ~ 1 | episode_id) else NULL
  fixed_form <- if (include_interaction) {
    feature_value ~ is_case * time_to_event + age + sex + site
  } else {
    feature_value ~ is_case + time_to_event + age + sex + site
  }
  lme(
    fixed = fixed_form,
    random = random_form,
    correlation = correlation,
    data = sub,
    method = method,
    control = lmeControl(maxIter = 200, msMaxIter = 200, opt = "optim")
  )
}

empty_result <- function(
  feature,
  n_obs,
  n_groups,
  fallback = FALSE,
  ar1_boundary = FALSE,
  robust_fallback = FALSE,
  inference = NA_character_
) {
  data.table(
    feature = feature, term = NA_character_,
    estimate = NA_real_, ci_lower = NA_real_, ci_upper = NA_real_,
    p_value = NA_real_, n_obs = n_obs, n_groups = n_groups,
    converged = FALSE, ar1_rho = NA_real_, fallback = fallback,
    ar1_boundary = ar1_boundary, robust_fallback = robust_fallback,
    robust_df = NA_real_, inference = inference
  )
}

# ---- Main loop ----
results <- list()

for (i in seq_along(features)) {
  feat <- features[i]
  if (i %% 10 == 1 || i == length(features)) {
    message(sprintf("[lmm_fit] (%d/%d) %s", i, length(features), feat))
  }

  sub <- dat[feature_name == feat]
  sub <- sub[complete.cases(sub[, .(feature_value, is_case, time_to_event,
                                     age, sex, site, episode_id, key)])]
  sub$is_case <- as.logical(sub$is_case)
  sub$sex     <- factor(sub$sex)
  sub$site    <- factor(sub$site)

  # Mean-center continuous predictors for interpretable main effects
  sub$age           <- sub$age - mean(sub$age)
  sub$time_to_event <- sub$time_to_event - mean(sub$time_to_event)

  n_obs    <- nrow(sub)
  n_groups <- length(unique(sub$episode_id))

  if (n_obs < 5 || n_groups < 2) {
    # Not enough data — emit NA row
    results[[length(results) + 1]] <- empty_result(feat, n_obs, n_groups)
    next
  }

  # ---- Fit full model: interaction + random slope + AR(1) ----
  mod <- NULL
  mod_converged <- TRUE
  mod_coefs <- NULL
  mod_rho <- NA_real_
  fallback <- FALSE
  ar1_boundary <- FALSE
  robust_fallback <- FALSE
  inference <- "model-based"

  tryCatch({
    mod <- fit_lme(sub, random_slope = TRUE, use_ar1 = TRUE)
    mod_coefs <- extract_model_coefs(mod)
    mod_rho <- extract_rho(mod)
  }, error = function(e) {
    message(sprintf("  [WARN] Full model (random slope) failed for %s: %s", feat, e$message))
    mod <<- NULL
    mod_converged <<- FALSE
  })

  # ---- Convergence fallback: random intercept only ----
  if (is.null(mod)) {
    fallback <- TRUE
    mod_converged <- TRUE
    tryCatch({
      mod <- fit_lme(sub, random_slope = FALSE, use_ar1 = TRUE)
      mod_coefs <- extract_model_coefs(mod)
      mod_rho <- extract_rho(mod)
      message(sprintf("  [INFO] Fallback (random intercept) succeeded for %s", feat))
    }, error = function(e) {
      message(sprintf("  [WARN] Fallback model also failed for %s: %s", feat, e$message))
      mod <<- NULL
      mod_converged <<- FALSE
    })
  }

  # ---- Boundary fallback: no AR(1), CR2 clustered by participant ----
  if (!is.null(mod) && is.finite(mod_rho) && abs(mod_rho) >= AR1_BOUNDARY) {
    ar1_boundary <- TRUE
    robust_fallback <- TRUE
    inference <- "participant-clustered CR2"
    random_slope <- !fallback
    message(sprintf(
      "  [WARN] AR(1) boundary for %s (rho=%.6f); applying no-AR CR2 fallback",
      feat, mod_rho
    ))

    robust_mod <- tryCatch(
      fit_lme(sub, random_slope = random_slope, use_ar1 = FALSE),
      error = function(e) {
        message(sprintf("  [WARN] No-AR model failed for %s: %s", feat, e$message))
        NULL
      }
    )
    if (is.null(robust_mod) && random_slope) {
      fallback <- TRUE
      robust_mod <- tryCatch(
        fit_lme(sub, random_slope = FALSE, use_ar1 = FALSE),
        error = function(e) {
          message(sprintf(
            "  [WARN] No-AR random-intercept fallback failed for %s: %s",
            feat, e$message
          ))
          NULL
        }
      )
    }
    if (is.null(robust_mod)) {
      stop(sprintf("No valid no-AR robust fallback could be fit for %s", feat))
    }

    mod <- robust_mod
    mod_coefs <- extract_cr2_coefs(mod, cluster = sub$key)
    message(sprintf(
      "  [INFO] No-AR participant-clustered CR2 fallback succeeded for %s", feat
    ))
  }

  # Build output
  if (!is.null(mod_coefs)) {
    mod_out <- mod_coefs[term != "(Intercept)"]
    mod_out[, `:=`(
      feature   = feat,
      n_obs     = n_obs,
      n_groups  = n_groups,
      converged = mod_converged,
      ar1_rho   = mod_rho,
      fallback  = fallback,
      ar1_boundary = ar1_boundary,
      robust_fallback = robust_fallback,
      inference = inference
    )]
  } else {
    mod_out <- empty_result(
      feat, n_obs, n_groups,
      fallback = fallback, ar1_boundary = ar1_boundary,
      robust_fallback = robust_fallback, inference = inference
    )
  }

  results[[length(results) + 1]] <- mod_out
}

# ---- Combine and write ----
out <- rbindlist(results, use.names = TRUE, fill = TRUE)

# Reorder columns
col_order <- c("feature", "term", "estimate", "ci_lower", "ci_upper",
               "p_value", "n_obs", "n_groups",
               "converged", "ar1_rho", "fallback", "ar1_boundary",
               "robust_fallback", "robust_df", "inference")
setcolorder(out, col_order)

fwrite(out, output_csv)
message(sprintf("[lmm_fit] Done. Wrote %d rows to %s", nrow(out), output_csv))
