###############################################################
## download_meteo_oct24_may25.R
##
## Collect HADS, LCD & WCC station data for:
##   2024-10-01 00:00:00 UTC  →  2025-05-31 23:59:59 UTC
## and save to  data_download/  next to this script.
###############################################################


# Collects data based on https://github.com/LynkerIntel/rainOrSnowTools/blob/cicd_pipeline/R/meteo_access.R

## ---------------- 1.  SET-UP ---------------------------------

## Install/load packages & the local repo  ----------------
pkg_needed <- c("devtools", "tidyverse", "lubridate", "purrr", "progress", "readr")
inst <- pkg_needed[!(pkg_needed %in% installed.packages()[,"Package"])]
if (length(inst)) install.packages(inst, repos = "https://cloud.r-project.org")

library(devtools)      # install_local / load_all
library(tidyverse)
library(lubridate)
library(purrr)
library(progress)
library(readr)

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
lon_obs       <- -119.5 # -120.028572    # longitude of map centre
lat_obs       <-  37.75 # 39.097291      # latitude  of map centre
deg_filter    <- 2.25   # 2              # ~350 km radius
dist_thresh_m <- 350000 # 300000         # 350 km radius (just outside rectangle's corners) # threshold for capturing stations around


## ---------------- 3.  STATION META ---------------------------
stations_hads <- station_select("HADS", lon_obs, lat_obs, deg_filter, dist_thresh_m)
stations_lcd  <- station_select("LCD",  lon_obs, lat_obs, deg_filter, dist_thresh_m)
stations_wcc  <- station_select("WCC",  lon_obs, lat_obs, deg_filter, dist_thresh_m)


## ---------------- 4.  DOWNLOAD HELPERS -----------------------
## HADS and LCD handle long spans internally — just call once
# preprocess_meteo() in https://github.com/LynkerIntel/rainOrSnowTools/blob/cicd_pipeline/R/meteo_access.R converts temp data into Celcius
get_hads <- function() {
  download_meteo_hads(start_utc, end_utc, stations_hads) %>%
    preprocess_meteo("HADS", .)
}

get_lcd <- function() {
  download_meteo_lcd(start_utc, end_utc, stations_lcd) %>%
    preprocess_meteo("LCD", .)
}

## WCC helper — WCC endpoint only survives a single day at a time, so loop day-by-day and bind.
get_wcc <- function(start_utc, end_utc, stations_wcc) {
  dates <- seq.Date(as_date(start_utc), as_date(end_utc), by = "day")
  pb   <- progress_bar$new(
    format = "  WCC [:bar] :percent eta: :eta",
    total  = length(dates),
    width = 40
  )

  # set up an empty tibble to collect errors
  error_log <- tibble(date = as.Date(character()), error = character())

  # loop & collect raw results (or NULL on failure)
  raw_list <- vector("list", length(dates))
  for (i in seq_along(dates)) {
    d <- dates[i]
    pb$tick()

    # throttle to avoid overwhelming the server
    Sys.sleep(1)

    dt0 <- as_datetime(d, tz = "UTC")
    dt1 <- dt0 + days(1) - seconds(1)

    raw_list[[i]] <- tryCatch(
      download_meteo_wcc(dt0, dt1, stations_wcc),
      error = function(e) {
        error_log <<- bind_rows(
          error_log,
          tibble(date = d, error = e$message)
        )
        return(NULL)
      }
    )
  }

  # write any errors out
  if (nrow(error_log)) {
    write_csv(error_log, file.path(out_dir, "wcc_error_log.csv"))
    message("Logged ", nrow(error_log), " WCC errors → wcc_error_log.csv")
  }

  # bind only the successful days
  raw_df <- bind_rows(compact(raw_list))  # drops NULLs

  # preprocessing
  df <- preprocess_meteo("WCC", raw_df)

  return(df)
}



## ---------------- 5.  DOWNLOAD  ----------------------------
cat("Downloading HADS…\n")
hads_df <- get_hads()
write_csv(hads_df, file.path(out_dir, "hads_20241001_20250531.csv"))

cat("Downloading LCD…\n")
lcd_df  <- get_lcd()
write_csv(lcd_df,  file.path(out_dir, "lcd_20241001_20250531.csv"))

cat("Downloading WCC (this will take a while)…\n")
wcc_df <- get_wcc(start_utc, end_utc, stations_wcc)
write_csv(wcc_df,  file.path(out_dir, "wcc_20241001_20250531.csv"))


## ---------------- 6.  GRAB METADATA  ----------------------------
# collect all station IDs
all_ids <- c(
  hads_df$id,
  lcd_df$id,
  wcc_df$id
)

# keep only the uniques
unique_ids <- unique(all_ids)

# fetch metadata (lat/lon, elev, etc.) for each ID
stations_meta <- gather_meta(unique_ids)

# inspect
print(stations_meta)
write_csv(stations_meta, file.path(out_dir, "station_metadata_20241001_20250531.csv"))


# temp dew, and temp wet not calculated using rainorsnowtools method model_meteo... later filled in via similar supplementary calculations in "preprocessing.ipynb"