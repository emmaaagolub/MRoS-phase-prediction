###############################################################
## download_meteo_oct24_may25.R
##
## Collect HADS, LCD & WCC station data for:
##   2024-10-01 00:00:00 UTC  →  2025-05-31 23:59:59 UTC
## and save to  data_download/  next to this script.
###############################################################


## ---------------- 1.  SET-UP ---------------------------------

## Install/load packages & the local repo  ----------------
pkg_needed <- c("devtools", "tidyverse", "lubridate", "purrr", "progress")
inst <- pkg_needed[!(pkg_needed %in% installed.packages()[,"Package"])]
if (length(inst)) install.packages(inst, repos = "https://cloud.r-project.org")

library(devtools)      # install_local / load_all
library(tidyverse)
library(lubridate)
library(purrr)
library(progress)

## File path setups
# Get script directory robustly (works in Rscript and RStudio)
get_script_dir <- function() {
  # Works when run via `Rscript`
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- "--file="
  path_idx <- grep(file_arg, cmd_args)
  if (length(path_idx) > 0) {
    # Called with Rscript
    return(dirname(normalizePath(sub(file_arg, "", cmd_args[path_idx]))))
  } else if (requireNamespace("rstudioapi", quietly = TRUE) &&
             rstudioapi::isAvailable()) {
    # Interactive in RStudio
    return(dirname(normalizePath(rstudioapi::getSourceEditorContext()$path)))
  } else {
    # Fallback to working directory
    warning("Cannot determine script location; using working directory instead.")
    return(getwd())
  }
}

script_dir <- get_script_dir() # /mros-precipitation-phase-product-prototype/Scripts
mros_path  <- normalizePath(file.path(script_dir, "..", "..", "rainOrSnowTools"), mustWork = TRUE)

## Load MRoS Repo (assumes repo is already cloned and in parent directory to this script)
devtools::load_all(mros_path)

## Set output Dir
out_dir <- normalizePath(file.path(script_dir, "..", "Data", "Stations"), mustWork = TRUE)

## ---------------- 2.  PERIOD & AREA  --------------------------
start_utc <- as_datetime("2024-10-01 00:00:00", tz = "UTC")
end_utc   <- as_datetime("2025-05-31 23:59:59", tz = "UTC")

## Choose reference point & how wide to cast the net ------------
# Lake Tahoe: 39.097291, -120.028572
lon_obs       <- -120.028572  # longitude of map centre
lat_obs       <-  39.097291   # latitude  of map centre
deg_filter    <- 2            # 2° ~ 280 km
dist_thresh_m <- 300000       # 300 km # threshold for capturing stations around


## ---------------- 3.  STATION META ---------------------------
stations_hads <- station_select("HADS", lon_obs, lat_obs, deg_filter, dist_thresh_m)
stations_lcd  <- station_select("LCD",  lon_obs, lat_obs, deg_filter, dist_thresh_m)
stations_wcc  <- station_select("WCC",  lon_obs, lat_obs, deg_filter, dist_thresh_m)


## ---------------- 4.  DOWNLOAD HELPERS -----------------------
## HADS and LCD handle long spans internally — just call once
get_hads <- function() {
  download_meteo_hads(start_utc, end_utc, stations_hads) %>%
    preprocess_meteo("HADS", .)
}

get_lcd <- function() {
  download_meteo_lcd(start_utc, end_utc, stations_lcd) %>%
    preprocess_meteo("LCD", .)
}

## WCC helper — WCC endpoint only survives a single day at a time, so loop day-by-day and bind.
get_wcc <- function() {
  dates <- seq.Date(as_date(start_utc), as_date(end_utc), by = "day")
  pb <- progress_bar$new(
    format = "  WCC downloading [:bar] :percent eta: :eta",
    total  = length(dates),
    width  = 60
  )
  map_dfr(dates, function(d) {
    download_meteo_wcc(as_datetime(d, tz = "UTC"),
                       as_datetime(d, tz = "UTC") + days(1) - seconds(1),
                       stations_wcc)

  }) %>%
    preprocess_meteo("WCC", .)
}


## ---------------- 5.  DOWNLOAD  ----------------------------
cat("Downloading HADS…\n")
hads_df <- get_hads()
write_csv(hads_df, file.path(out_dir, "hads_20241001_20250531.csv"))

cat("Downloading LCD…\n")
lcd_df  <- get_lcd()
write_csv(lcd_df,  file.path(out_dir, "lcd_20241001_20250531.csv"))

cat("Downloading WCC (this will take a while)…\n")
wcc_df  <- get_wcc()
write_csv(wcc_df,  file.path(out_dir, "wcc_20241001_20250531.csv"))


## ---------------- 6.  COMBINE & DEDUP ------------------------
met_all <- bind_rows(hads_df, lcd_df, wcc_df) %>%
  distinct(id, datetime, .keep_all = TRUE)   # in case of overlaps
write_csv(met_all, file.path(out_dir, "meteo_all_20241001_20250531.csv"))

cat("\n Finished downloads. CSVs are in:", out_dir, "\n")
