###############################################################
## download_station_data.R
##
## Collect HADS, LCD & WCC station data for two regions:
##   - Sierra Nevada / Lake Tahoe (CA)
##   - Colorado Mountains (CO)
##
## Period: 2022-10-01 00:00:00 UTC  →  2026-05-01 23:59:59 UTC
##
## Outputs saved to:
##   <repo>/Data/Stations/CA/
##   <repo>/Data/Stations/CO/
###############################################################

# Collects data based on:
# https://github.com/LynkerIntel/rainOrSnowTools/blob/cicd_pipeline/R/meteo_access.R


## ---------------- 1.  SET-UP ---------------------------------

pkg_needed <- c("devtools", "tidyverse", "lubridate", "purrr", "progress", "readr")
inst <- pkg_needed[!(pkg_needed %in% installed.packages()[, "Package"])]
if (length(inst)) install.packages(inst, repos = "https://cloud.r-project.org")

library(devtools)
library(tidyverse)
library(lubridate)
library(purrr)
library(progress)
library(readr)


## Resolve paths -----------------------------------------------
# Script lives at: <repo>/Scripts/data_collection/download_station_data.R

get_script_dir <- function() {
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg  <- "--file="
  path_idx  <- grep(file_arg, cmd_args)
  if (length(path_idx) > 0) {
    return(dirname(normalizePath(sub(file_arg, "", cmd_args[path_idx]))))
  } else if (requireNamespace("rstudioapi", quietly = TRUE) &&
             rstudioapi::isAvailable()) {
    return(dirname(normalizePath(rstudioapi::getSourceEditorContext()$path)))
  } else {
    warning("Cannot determine script location; using working directory instead.")
    return(getwd())
  }
}

script_dir <- get_script_dir()
repo_root  <- normalizePath(file.path(script_dir, "..", ".."))
mros_path  <- normalizePath(file.path(repo_root, "..", "rainOrSnowTools"), mustWork = TRUE)

devtools::load_all(mros_path)


## ---------------- 2.  PERIOD ---------------------------------

start_utc <- as_datetime("2022-10-01 00:00:00", tz = "UTC")
end_utc   <- as_datetime("2026-05-01 23:59:59", tz = "UTC")

date_suffix <- paste0(
  format(start_utc, "%Y%m%d"), "_",
  format(end_utc,   "%Y%m%d")
)
message("Date range : ", date_suffix)


## ---------------- 3.  REGION CONFIG --------------------------
# Center point = centroid of AOI polygon vertices (mean lon, mean lat)
# Matches polygon coordinates defined in get_elevation.R

REGIONS <- list(
  CA = list(
    label         = "california",
    lon_obs       = mean(c(-119.45505750721992, -121.27878797084242,
                           -119.11448133630248, -118.49924696303225,
                           -119.46604383531404)),
    lat_obs       = mean(c(39.65343608043361,   39.66189413918429,
                           36.726935737063016,  37.235952484988736,
                           38.37304030164334)),
    deg_filter    = 2.25,
    dist_thresh_m = 350000
  ),
  CO = list(
    label         = "colorado",
    lon_obs       = mean(c(-105.19885928678391, -106.88927700287375,
                           -107.66078646416787, -104.87856373593310)),
    lat_obs       = mean(c(40.62046076499234,   40.555465783967925,
                           38.79540171139857,   38.77382201116306)),
    deg_filter    = 2.25,
    dist_thresh_m = 350000
  )
)


## ---------------- 4.  DOWNLOAD HELPERS -----------------------

# Coerce all columns to character for safe binding across days
normalize_types <- function(df) {
  dplyr::mutate(df, across(everything(), as.character))
}

# Generic batched downloader with per-batch retry + exponential backoff.
# Used for HADS and LCD which both accept a stations tibble.
download_batched <- function(source, start_utc, end_utc, stations,
                             batch_size, max_retries = 3, sleep_base = 5) {
  n         <- nrow(stations)
  n_batches <- ceiling(n / batch_size)
  message("  ", source, ": ", n, " stations across ",
          n_batches, " batches of ", batch_size)
  
  download_fn <- switch(source,
                        HADS = download_meteo_hads,
                        LCD  = download_meteo_lcd,
                        stop("Unknown source: ", source)
  )
  
  error_log <- tibble(
    batch    = integer(),
    stations = character(),
    attempt  = integer(),
    error    = character()
  )
  results <- vector("list", n_batches)
  
  for (i in seq_len(n_batches)) {
    idx       <- ((i - 1) * batch_size + 1):min(i * batch_size, n)
    batch     <- stations[idx, ]
    batch_ids <- paste(batch$id, collapse = ",")
    success   <- FALSE
    
    for (attempt in seq_len(max_retries)) {
      Sys.sleep(sleep_base * attempt)  # 5s, 10s, 15s
      
      results[[i]] <- tryCatch({
        df <- download_fn(start_utc, end_utc, batch)
        preprocess_meteo(source, df)
      }, error = function(e) {
        message("  ", source, " batch ", i, "/", n_batches,
                " attempt ", attempt, "/", max_retries,
                " failed: ", e$message)
        error_log <<- bind_rows(error_log, tibble(
          batch    = i,
          stations = batch_ids,
          attempt  = attempt,
          error    = e$message
        ))
        NULL
      })
      
      if (!is.null(results[[i]])) {
        success <- TRUE
        break
      }
    }
    
    if (!success) {
      message("  ", source, " batch ", i, " failed all ",
              max_retries, " attempts — skipped.")
    }
  }
  
  list(data = bind_rows(compact(results)), errors = error_log)
}

## WCC snotel data
get_wcc_awdb <- function(start_utc, end_utc, stations, out_dir) {
  
  library(httr)
  library(jsonlite)
  
  # ---- Build station triplets ---------------------------------------------
  triplets <- stations %>%
    dplyr::mutate(network_ab = dplyr::case_when(
      network == "snotel"  ~ "SNTL",
      network == "snotelt" ~ "SNTLT",
      network == "scan"    ~ "SCAN",
      TRUE                 ~ toupper(network)
    )) %>%
    dplyr::mutate(triplet = paste(station.id, state, network_ab, sep = ":")) %>%
    dplyr::pull(triplet)
  
  message("  Fetching AWDB data for ", length(triplets), " stations...")
  message("  Period: ", as_date(start_utc), " to ", as_date(end_utc))
  
  # ---- Year chunks x station batches --------------------------------------
  years              <- seq(year(start_utc), year(end_utc))
  station_batch_size <- 10
  station_batches    <- split(triplets,
                              ceiling(seq_along(triplets) / station_batch_size))
  total_calls        <- length(years) * length(station_batches)
  message("  ", length(years), " years x ", length(station_batches),
          " station batches = ", total_calls, " total requests")
  
  # ---- Resume: skip years already checkpointed ---------------------------
  completed_years <- list.files(out_dir,
                                pattern = "wcc_awdb_checkpoint_through_.*\\.csv") %>%
    gsub("wcc_awdb_checkpoint_through_|\\.csv", "", .) %>%
    as.integer()
  
  if (length(completed_years) > 0) {
    message("  Found checkpoints for years: ",
            paste(sort(completed_years), collapse = ", "), " — loading...")
    existing_long <- bind_rows(lapply(
      file.path(out_dir,
                paste0("wcc_awdb_checkpoint_through_",
                       sort(completed_years), ".csv")),
      read_csv, show_col_types = FALSE
    ))
    all_long <- list(existing_long)
    years    <- years[!(years %in% completed_years)]
    message("  Skipping completed years; ", length(years), " remaining.")
  } else {
    all_long <- list()
  }
  
  # ---- Fetch helper -------------------------------------------------------
  fetch_awdb <- function(triplet_batch, begin_date, end_date,
                         max_retries = 3) {
    for (attempt in seq_len(max_retries)) {
      Sys.sleep(c(2, 10, 30)[min(attempt, 3)])
      
      resp <- tryCatch(
        httr::GET(
          "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data",
          query = list(
            stationTriplets = paste(triplet_batch, collapse = ","),
            elements        = "TOBS,RHUM,DPTP",
            beginDate       = begin_date,
            endDate         = end_date,
            duration        = "HOURLY",
            periodRef       = "START"
          ),
          httr::timeout(300)  # 5 minutes
        ),
        error = function(e) { 
          message("    httr error: ", e$message)
          NULL 
        }
      )
      
      # Guard: skip status_code check if request failed
      if (is.null(resp)) {
        message("    Request failed on attempt ", attempt)
        next
      }
      
      if (httr::status_code(resp) == 200) {
        return(jsonlite::fromJSON(
          httr::content(resp, as = "text", encoding = "UTF-8"),
          simplifyVector = TRUE
        ))
      }
      
      message("    HTTP ", httr::status_code(resp),
              " on attempt ", attempt,
              " (", begin_date, " to ", end_date, ")")
    }
    message("    All retries failed")
    return(NULL)
  }
  
  # ---- Parse helper -------------------------------------------------------
  parse_station <- function(station_triplet, data_row) {
    if (is.null(data_row$values) || length(data_row$values) == 0) return(NULL)
    
    elements <- data_row$stationElement$elementCode
    
    dfs <- lapply(seq_along(elements), function(e) {
      vals <- data_row$values[[e]]
      if (is.null(vals) || nrow(vals) == 0) return(NULL)
      data.frame(
        station  = station_triplet,
        datetime = as.POSIXct(vals$date, format = "%Y-%m-%d %H:%M",
                              tz = "UTC"),
        element  = elements[e],
        value    = as.numeric(vals$value),
        stringsAsFactors = FALSE
      )
    })
    
    bind_rows(compact(dfs))
  }
  
  # ---- Main fetch loop: year x station batch ------------------------------
  call_n <- 0
  
  for (yr in years) {
    
    begin_date <- format(max(as_date(start_utc),
                             as_date(paste0(yr, "-01-01"))), "%Y-%m-%d")
    end_date   <- format(min(as_date(end_utc),
                             as_date(paste0(yr, "-12-31"))), "%Y-%m-%d")
    
    for (b in seq_along(station_batches)) {
      call_n <- call_n + 1
      message("  [", call_n, "/", total_calls, "]  ",
              yr, "  batch ", b, "/", length(station_batches),
              " (", length(station_batches[[b]]), " stations)...")
      
      result <- fetch_awdb(station_batches[[b]], begin_date, end_date)
      
      if (is.null(result) || nrow(result) == 0) {
        message("    No data returned — skipping.")
        next
      }
      
      batch_long <- bind_rows(lapply(seq_len(nrow(result)), function(i) {
        parse_station(result$stationTriplet[i], result$data[[i]])
      }))
      
      all_long[[length(all_long) + 1]] <- batch_long
      Sys.sleep(1)
    }
    
    # ---- Checkpoint after each year --------------------------------------
    year_df <- bind_rows(all_long)
    checkpoint_path <- file.path(
      out_dir,
      paste0("wcc_awdb_checkpoint_through_", yr, ".csv")
    )
    write_csv(year_df, checkpoint_path)
    message("  Checkpoint saved through ", yr,
            " -> ", basename(checkpoint_path))
  }
  
  # ---- Combine all results ------------------------------------------------
  long_df <- bind_rows(all_long)
  
  # Save long_df immediately — if anything below fails, raw data is safe
  write_csv(long_df, file.path(out_dir, paste0("wcc_awdb_long_raw_",
                                               format(as_date(start_utc), "%Y%m%d"), "_",
                                               format(as_date(end_utc),   "%Y%m%d"), ".csv")))
  message("  Raw long data saved (", nrow(long_df), " rows)")
  
  if (nrow(long_df) == 0) {
    message("  No data retrieved.")
    return(NULL)
  }
  
  # ---- Pivot wide ---------------------------------------------------------
  wide_df <- long_df %>%
    tidyr::pivot_wider(
      names_from  = element,
      values_from = value
    ) %>%
    dplyr::rename_with(tolower) %>%
    dplyr::rename(
      tair = any_of("tobs"),
      rh   = any_of("rhum"),
      tdew = any_of("dptp")
    ) %>%
    dplyr::mutate(
      .id          = sub(":.*", "", station),
      date         = format(datetime, "%Y-%m-%d %H:%M"),
      datetime_lst = datetime,
      tair         = if ("tair" %in% names(.)) as.numeric(tair) else NA_real_,
      rh           = if ("rh"   %in% names(.)) as.numeric(rh)   else NA_real_,
      tdew         = if ("tdew" %in% names(.)) as.numeric(tdew) else NA_real_
    ) %>%
    dplyr::select(.id, date, tair, rh, tdew, datetime_lst, datetime)
  
  # Save wide_df before preprocessing
  write_csv(wide_df, file.path(out_dir, paste0("wcc_awdb_raw_",
                                               format(as_date(start_utc), "%Y%m%d"), "_",
                                               format(as_date(end_utc),   "%Y%m%d"), ".csv")))
  message("  Retrieved ", nrow(wide_df), " rows for ",
          length(unique(wide_df$.id)), " stations.")
  
  wide_df %>% preprocess_meteo("WCC", .)
}

# ## ---------------- TESTING BLOCK (remove before full run) ----------------
# 
# # Use real stations from the CA region
# test_stations <- station_select("WCC",
#                                 mean(c(-119.455, -121.279, -119.114,
#                                        -118.499, -119.466)),
#                                 mean(c(39.653, 39.662, 36.727,
#                                        37.236, 38.373)),
#                                 deg_filter    = 2.25,
#                                 dist_thresh_m = 350000)
# 
# test_out_dir <- file.path(repo_root, "Data", "Stations", "CA")
# dir.create(test_out_dir, showWarnings = FALSE, recursive = TRUE)
# 
# # Test 1: normal day — should return data
# message("--- Test 1: normal day ---")
# t1 <- download_meteo_wcc(
#   as_datetime("2023-01-15 00:00:00", tz = "UTC"),
#   as_datetime("2023-01-15 23:59:59", tz = "UTC"),
#   test_stations
# )
# message("rows: ", nrow(t1), "  cols: ", paste(names(t1), collapse = ", "))
# message("tair type: ", class(t1$tair))
# 
# # Test 2: simulate a day that returns NULL (bad date far in future)
# message("--- Test 2: expected failure ---")
# t2 <- tryCatch(
#   download_meteo_wcc(
#     as_datetime("2030-01-01 00:00:00", tz = "UTC"),
#     as_datetime("2030-01-01 23:59:59", tz = "UTC"),
#     test_stations
#   ),
#   error = function(e) { message("caught error: ", e$message); NULL }
# )
# message("t2 is NULL: ", is.null(t2))
# 
# # Test 3: simulate what get_wcc does — bind a real day + NULL + real day
# message("--- Test 3: normalize_types + bind_rows across mixed results ---")
# raw_list_test <- list(t1, NULL, t1)
# combined <- bind_rows(lapply(compact(raw_list_test), normalize_types))
# message("combined rows: ", nrow(combined))
# message("all character: ", all(sapply(combined, is.character)))
# 
# # Test 4: simulate resume — write a checkpoint then read it back
# message("--- Test 4: checkpoint write + resume ---")
# write_csv(normalize_types(t1),
#           file.path(test_out_dir, "wcc_checkpoint_through_2023-01-15.csv"))
# existing <- bind_rows(lapply(
#   list.files(test_out_dir, pattern = "wcc_checkpoint_.*\\.csv",
#              full.names = TRUE),
#   read_csv, show_col_types = FALSE
# )) %>% normalize_types()
# message("resumed rows: ", nrow(existing))
# message("datetime sample: ", existing$datetime[1])
# 
# # Clean up test checkpoint
# file.remove(file.path(test_out_dir, "wcc_checkpoint_through_2023-01-15.csv"))
# 
# message("--- All tests passed ---")
# ## ---------------- END TESTING BLOCK -------------------------------------


## ---------------- 5.  MAIN LOOP — one run per region ---------

for (region_id in names(REGIONS)) {
  cfg <- REGIONS[[region_id]]
  
  message("\n", strrep("=", 55))
  message("  Region: ", region_id, " (", cfg$label, ")")
  message("  Centre: ", round(cfg$lon_obs, 3), ", ", round(cfg$lat_obs, 3))
  message(strrep("=", 55))
  
  # Create region output subdirectory
  out_dir <- file.path(repo_root, "Data", "Stations", region_id)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Station selection
  message("Selecting stations...")
  stations_hads <- station_select("HADS", cfg$lon_obs, cfg$lat_obs,
                                  cfg$deg_filter, cfg$dist_thresh_m)
  
  stations_lcd <- lcd_meta %>%
    dplyr::filter(
      LONGITUDE >= cfg$lon_obs - cfg$deg_filter & LONGITUDE <= cfg$lon_obs + cfg$deg_filter,
      LATITUDE  >= cfg$lat_obs - cfg$deg_filter & LATITUDE  <= cfg$lat_obs + cfg$deg_filter
    ) %>%
    dplyr::rowwise() %>%
    dplyr::mutate(dist = geosphere::distHaversine(
      c(cfg$lon_obs, cfg$lat_obs), c(LONGITUDE, LATITUDE))) %>%
    dplyr::ungroup() %>%
    dplyr::filter(dist <= cfg$dist_thresh_m)
  ## DEPRECATED:
  # stations_lcd  <- station_select("LCD",  cfg$lon_obs, cfg$lat_obs,
  #                                 cfg$deg_filter, cfg$dist_thresh_m)
  
  stations_wcc  <- station_select("WCC",  cfg$lon_obs, cfg$lat_obs,
                                  cfg$deg_filter, cfg$dist_thresh_m)
  
  message("  HADS stations : ", nrow(stations_hads))
  message("  LCD stations  : ", nrow(stations_lcd))
  message("  WCC stations  : ", nrow(stations_wcc))
  
  ## ------------ HADS --------------
  # HADS — batched with retry
  message("\nDownloading HADS...")
  hads_out <- download_batched("HADS", start_utc, end_utc, stations_hads,
                               batch_size = 50)
  hads_df  <- hads_out$data
  write_csv(hads_df, file.path(out_dir, paste0("hads_", date_suffix, ".csv")))
  if (nrow(hads_out$errors) > 0)
    write_csv(hads_out$errors, file.path(out_dir, "hads_error_log.csv"))
  # hads_df <- read_csv(file.path(out_dir, paste0("hads_", date_suffix, ".csv")), show_col_types = FALSE)
  
  ## ------------ LCD ASOS --------------
  # LCD — batched with retry (smaller batches due to API timeout sensitivity)
  message("\nDownloading LCD...")
  lcd_out <- download_batched("LCD", start_utc, end_utc, stations_lcd,
                              batch_size = 10)
  lcd_df  <- lcd_out$data
  write_csv(lcd_df, file.path(out_dir, paste0("lcd_", date_suffix, ".csv")))
  if (nrow(lcd_out$errors) > 0)
    write_csv(lcd_out$errors, file.path(out_dir, "lcd_error_log.csv"))
  # lcd_df <- read_csv(file.path(out_dir, paste0("lcd_", date_suffix, ".csv")), show_col_types = FALSE)
  
  ## ------------ WCC SNOTEL --------------
  # WCC — day-by-day loop with retry
  message("\nDownloading WCC (this will take a while)...")
  wcc_df <- get_wcc_awdb(start_utc, end_utc, stations_wcc, out_dir)
  write_csv(wcc_df, file.path(out_dir, paste0("wcc_", date_suffix, ".csv")))
  
  # Station metadata — separate file per region
  message("\nFetching station metadata...")
  all_ids       <- unique(c(hads_df$id, lcd_df$id, wcc_df$id))
  stations_meta <- gather_meta(all_ids)
  write_csv(
    stations_meta,
    file.path(out_dir, paste0("station_metadata_", date_suffix, ".csv"))
  )
  
  message("  Saved ", nrow(stations_meta), " station records -> ", out_dir)
}

message("\nAll regions complete.")

# Note: temp_dew and temp_wet are not calculated here via model_meteo;
# they are filled in downstream in preprocessing.ipynb
