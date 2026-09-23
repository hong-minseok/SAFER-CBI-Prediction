#!/usr/bin/env Rscript

expected_r <- "4.6.1"
expected_packages <- c(
  nlme = "3.1.169",
  data.table = "1.18.4",
  zoo = "1.8.15",
  cli = "3.6.6",
  rlang = "1.3.0",
  sandwich = "3.1.2",
  lifecycle = "1.0.5",
  clubSandwich = "0.7.0"
)
source_urls <- c(
  nlme = "https://cran.r-project.org/src/contrib/nlme_3.1-169.tar.gz",
  data.table = "https://cran.r-project.org/src/contrib/data.table_1.18.4.tar.gz",
  zoo = "https://cran.r-project.org/src/contrib/zoo_1.8-15.tar.gz",
  cli = "https://cran.r-project.org/src/contrib/cli_3.6.6.tar.gz",
  rlang = "https://cran.r-project.org/src/contrib/rlang_1.3.0.tar.gz",
  sandwich = "https://cran.r-project.org/src/contrib/sandwich_3.1-2.tar.gz",
  lifecycle = "https://cran.r-project.org/src/contrib/lifecycle_1.0.5.tar.gz",
  clubSandwich = "https://cran.r-project.org/src/contrib/clubSandwich_0.7.0.tar.gz"
)

if (as.character(getRversion()) != expected_r) {
  stop(
    sprintf("R %s is required; found %s", expected_r, getRversion()),
    call. = FALSE
  )
}

package_version_or_na <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    return(NA_character_)
  }
  as.character(utils::packageVersion(package))
}

for (package in names(expected_packages)) {
  installed <- package_version_or_na(package)
  expected <- unname(expected_packages[[package]])
  if (is.na(installed) || installed != expected) {
    message(sprintf("Installing %s %s", package, expected))
    utils::install.packages(
      unname(source_urls[[package]]),
      repos = NULL,
      type = "source",
      lib = .libPaths()[[1L]]
    )
  }
}

actual <- vapply(names(expected_packages), package_version_or_na, character(1))
mismatch <- is.na(actual) | actual != unname(expected_packages)
if (any(mismatch)) {
  details <- paste(
    sprintf(
      "%s expected=%s actual=%s",
      names(expected_packages)[mismatch],
      unname(expected_packages[mismatch]),
      actual[mismatch]
    ),
    collapse = "; "
  )
  stop(sprintf("R package verification failed: %s", details), call. = FALSE)
}

message("R packages are installed at the expected versions.")
