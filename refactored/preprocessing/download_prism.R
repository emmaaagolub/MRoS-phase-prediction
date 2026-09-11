## Download daily PRISM rasters, clip them to each study area, stack
## the variables into one file per day, then flatten the whole series into a
## table of gridded predictors.
##
## Variables: ppt, tmin, tmean, tmax, tdmean (800 m daily)
##
## Outputs per region:
##   Data/PRISM/<REGION>/prism_daily_stacked/PRISM_<YYYYMMDD>_stacked.tif
##   Data/PRISM/<REGION>/combined_prism_20221001_20260501.parquet
##
## Days already stacked are skipped, so the script can be re-run to fill gaps.

suppressPackageStartupMessages({
  library(sf)
  library(tidyverse)
  library(lubridate)
  library(httr)
  library(terra)
  library(arrow)
})

.script_dir <- local({
  args <- commandArgs(trailingOnly = FALSE)
  hit  <- grep("--file=", args)
  if (length(hit) > 0) {
    dirname(normalizePath(sub("--file=", "", args[hit[1]])))
  } else if (requireNamespace("rstudioapi", quietly = TRUE) &&
             rstudioapi::isAvailable()) {
    dirname(normalizePath(rstudioapi::getSourceEditorContext()$path))
  } else {
    getwd()
  }
})
source(file.path(.script_dir, "regions.R"))

REPO_ROOT  <- repo_root()
DATE_CODES <- format(seq(as.Date(WY_START), as.Date(WY_END), by = "day"), "%Y%m%d")
VARIABLES  <- c("ppt", "tmin", "tmean", "tmax", "tdmean")
PRISM_AREA <- "us"
PRISM_RES  <- "800m"


## Download one variable for one day and clip it to the AOI.
download_prism_var <- function(variable, date_code, aoi) {
  url <- sprintf("https://services.nacse.org/prism/data/get/%s/%s/%s/%s?format=bil",
                 PRISM_AREA, PRISM_RES, variable, date_code)

  work_dir <- file.path(tempdir(), paste0(variable, "_", date_code))
  dir.create(work_dir, showWarnings = FALSE)
  on.exit(unlink(work_dir, recursive = TRUE), add = TRUE)

  zip_path <- file.path(work_dir, paste0(variable, "_", date_code, ".zip"))
  result <- try(GET(url, write_disk(zip_path, overwrite = TRUE), timeout(60)),
                silent = TRUE)

  if (inherits(result, "try-error") || result$status_code != 200) {
    warning("Failed to download ", variable, " ", date_code)
    return(NULL)
  }

  unzip(zip_path, exdir = work_dir)
  bil_file <- list.files(work_dir, pattern = "\\.bil$", full.names = TRUE)
  if (length(bil_file) == 0) return(NULL)

  rast(bil_file) |> crop(aoi) |> mask(aoi)
}


## ---------------------------------------------------------------------------
## Download and stack
## ---------------------------------------------------------------------------

out_dirs <- setNames(
  lapply(REGION_IDS, function(r) ensure_dir(file.path(REPO_ROOT, "Data", "PRISM", r))),
  REGION_IDS
)

for (region_id in REGION_IDS) {
  aoi         <- aoi_polygon(region_id, round_dp = 3, hull = TRUE)
  stacked_dir <- ensure_dir(file.path(out_dirs[[region_id]], "prism_daily_stacked"))

  cat("\n=== ", region_id, " — ", REGION_LABEL[[region_id]], " ===\n", sep = "")

  for (d in DATE_CODES) {
    stacked_path <- file.path(stacked_dir, paste0("PRISM_", d, "_stacked.tif"))
    if (file.exists(stacked_path)) next

    cat("Processing", region_id, d, "\n")
    daily_layers <- list()

    for (v in VARIABLES) {
      r <- download_prism_var(v, d, aoi)
      if (!is.null(r)) {
        names(r) <- v
        daily_layers[[v]] <- r
      }
      Sys.sleep(0.3)  # be polite to the PRISM server
    }

    if (length(daily_layers) > 0) {
      writeRaster(rast(daily_layers), stacked_path, overwrite = TRUE)
    } else {
      cat("  nothing downloaded for", d, "- skipping\n")
    }
  }
}


## ---------------------------------------------------------------------------
## Flatten the daily stacks into one table per region
## ---------------------------------------------------------------------------

for (region_id in REGION_IDS) {
  stacked_dir <- file.path(out_dirs[[region_id]], "prism_daily_stacked")
  parquet_out <- file.path(out_dirs[[region_id]],
                           "combined_prism_20221001_20260501.parquet")

  files <- list.files(stacked_dir, pattern = "_stacked\\.tif$", full.names = TRUE)
  if (length(files) == 0) {
    warning("No stacked rasters for ", region_id)
    next
  }

  tif_dates <- gsub(".*PRISM_(\\d{8})_stacked\\.tif$", "\\1", basename(files))
  cat("\nBuilding parquet for", region_id, "-", length(files), "days\n")

  combined_df <- bind_rows(lapply(seq_along(files), function(i) {
    r  <- rast(files[i])
    df <- as.data.frame(r, xy = TRUE, na.rm = FALSE)
    names(df) <- c("lon", "lat", names(r))
    df$date   <- tif_dates[i]
    df
  }))

  cat("  ", nrow(combined_df), "rows x", ncol(combined_df), "cols\n")
  arrow::write_parquet(combined_df, parquet_out)
}

cat("\nDone.\n")
