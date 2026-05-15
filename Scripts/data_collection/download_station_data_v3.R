###############################################################
## download_station_data_v3.R
##
## Collect HADS, LCD (pre-deprecation), MADIS (post-deprecation) & WCC
## station data for two regions:
##   - Sierra Nevada / Lake Tahoe (CA)
##   - Colorado Mountains (CO)
##
## Period: 2022-10-01 00:00:00 UTC  →  2026-05-01 23:59:59 UTC
##
## LCD was deprecated on 2025-08-27. The script automatically:
##   - Uses LCD for data up to 2025-08-26 23:59:59 UTC
##   - Uses MADIS for data from 2025-08-27 00:00:00 UTC onward
##   - Skips re-downloading LCD data if an existing file is found
##
## Outputs saved to:
##   <repo>/Data/Stations/CA/
##   <repo>/Data/Stations/CO/
###############################################################

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

# LCD deprecation boundary
lcd_cutoff_utc <- as_datetime("2025-08-27 00:00:00", tz = "UTC")

# Effective LCD window: start → day before cutoff (if start is before cutoff)
lcd_end_utc  <- as_datetime("2025-08-26 23:59:59", tz = "UTC")

# Effective MADIS window: cutoff → end (if end is after cutoff)
madis_start_utc <- lcd_cutoff_utc

date_suffix <- paste0(
  format(start_utc, "%Y%m%d"), "_",
  format(end_utc,   "%Y%m%d")
)
message("Date range : ", date_suffix)
message("LCD window : ", format(start_utc, "%Y-%m-%d"), " → ",
        format(lcd_end_utc,  "%Y-%m-%d"))
message("MADIS window: ", format(madis_start_utc, "%Y-%m-%d"), " → ",
        format(end_utc, "%Y-%m-%d"))


## ---------------- 3.  REGION CONFIG --------------------------

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

normalize_types <- function(df) {
  dplyr::mutate(df, across(everything(), as.character))
}

# Generic batched downloader with per-batch retry + exponential backoff.
# Supports HADS and LCD (both accept a stations tibble).
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
      Sys.sleep(sleep_base * attempt)
      
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


# MADIS downloader: iterates hour-by-hour (MADIS is observation-time based,
# not a bulk range pull like HADS/LCD).
# Returns a preprocessed data frame compatible with the rest of the pipeline.
download_madis_range <- function(start_utc, end_utc,
                                 lon_obs, lat_obs,
                                 deg_filter, dist_thresh_m,
                                 max_retries = 3) {
  
  # Build sequence of hourly observation times to pull
  obs_times <- seq(
    from = ceiling_date(start_utc, unit = "hour"),
    to   = floor_date(end_utc,     unit = "hour"),
    by   = "hour"
  )
  
  n_times <- length(obs_times)
  message("  MADIS: ", n_times, " hourly obs windows to pull (",
          format(min(obs_times), "%Y-%m-%d %H:%M"), " → ",
          format(max(obs_times), "%Y-%m-%d %H:%M"), ")")
  
  results <- vector("list", n_times)
  
  for (i in seq_len(n_times)) {
    obs_t <- obs_times[[i]]
    
    for (attempt in seq_len(max_retries)) {
      Sys.sleep(c(2, 10, 30)[min(attempt, 3)])
      
      results[[i]] <- tryCatch({
        raw <- download_meteo_madis(
          lon_obs       = lon_obs,
          lat_obs       = lat_obs,
          deg_filter    = deg_filter,
          datetime_utc_obs = obs_t
        )
        # download_meteo_madis returns list(observations, stations)
        obs_df <- raw$observations
        if (is.null(obs_df) || nrow(obs_df) == 0) return(NULL)
        preprocess_meteo("MADIS", obs_df)
      }, error = function(e) {
        message("  MADIS [", i, "/", n_times, "] attempt ", attempt,
                " failed: ", e$message)
        NULL
      })
      
      if (!is.null(results[[i]])) break
    }
    
    # Light progress every 24 hours
    if (i %% 24 == 0) {
      message("  MADIS: ", i, "/", n_times, " windows complete (",
              format(obs_t, "%Y-%m-%d"), ")")
    }
  }
  
  bind_rows(compact(results))
}


## WCC AWDB downloader (unchanged from original) ---------------
get_wcc_awdb <- function(start_utc, end_utc, stations, out_dir) {
  
  library(httr)
  library(jsonlite)
  
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
  
  years              <- seq(year(start_utc), year(end_utc))
  station_batch_size <- 10
  station_batches    <- split(triplets,
                              ceiling(seq_along(triplets) / station_batch_size))
  total_calls        <- length(years) * length(station_batches)
  message("  ", length(years), " years x ", length(station_batches),
          " station batches = ", total_calls, " total requests")
  
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
          httr::timeout(300)
        ),
        error = function(e) {
          message("    httr error: ", e$message)
          NULL
        }
      )
      
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
    
    year_df <- bind_rows(all_long)
    checkpoint_path <- file.path(
      out_dir,
      paste0("wcc_awdb_checkpoint_through_", yr, ".csv")
    )
    write_csv(year_df, checkpoint_path)
    message("  Checkpoint saved through ", yr,
            " -> ", basename(checkpoint_path))
  }
  
  long_df <- bind_rows(all_long)
  
  write_csv(long_df, file.path(out_dir, paste0("wcc_awdb_long_raw_",
                                               format(as_date(start_utc), "%Y%m%d"), "_",
                                               format(as_date(end_utc),   "%Y%m%d"), ".csv")))
  message("  Raw long data saved (", nrow(long_df), " rows)")
  
  if (nrow(long_df) == 0) {
    message("  No data retrieved.")
    return(NULL)
  }
  
  wide_df <- long_df %>%
    tidyr::pivot_wider(names_from = element, values_from = value) %>%
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
  
  write_csv(wide_df, file.path(out_dir, paste0("wcc_awdb_raw_",
                                               format(as_date(start_utc), "%Y%m%d"), "_",
                                               format(as_date(end_utc),   "%Y%m%d"), ".csv")))
  message("  Retrieved ", nrow(wide_df), " rows for ",
          length(unique(wide_df$.id)), " stations.")
  
  wide_df %>% preprocess_meteo("WCC", .)
}
# 
# ## ---------------- TESTING BLOCK ------------------------------------------
# 
# # Shared test config — CA region centroid
# test_lon <- mean(c(-119.455, -121.279, -119.114, -118.499, -119.466))
# test_lat <- mean(c(39.653,   39.662,   36.727,   37.236,   38.373))
# test_deg_filter    <- 2.25
# test_dist_thresh_m <- 350000
# 
# test_out_dir <- file.path(repo_root, "Data", "Stations", "CA")
# dir.create(test_out_dir, showWarnings = FALSE, recursive = TRUE)
# 
# # ===========================================================================
# # WCC TESTS 
# # ===========================================================================
# 
# test_stations_wcc <- station_select("WCC", test_lon, test_lat,
#                                     deg_filter    = test_deg_filter,
#                                     dist_thresh_m = test_dist_thresh_m)
# 
# # Test 1: normal day — should return data
# message("--- Test 1 (WCC): normal day ---")
# t1 <- download_meteo_wcc(
#   as_datetime("2023-01-15 00:00:00", tz = "UTC"),
#   as_datetime("2023-01-15 23:59:59", tz = "UTC"),
#   test_stations_wcc
# )
# message("rows: ", nrow(t1), "  cols: ", paste(names(t1), collapse = ", "))
# message("tair type: ", class(t1$tair))
# 
# # Test 2: simulate a day that returns NULL (bad date far in future)
# message("--- Test 2 (WCC): expected failure ---")
# t2 <- tryCatch(
#   download_meteo_wcc(
#     as_datetime("2030-01-01 00:00:00", tz = "UTC"),
#     as_datetime("2030-01-01 23:59:59", tz = "UTC"),
#     test_stations_wcc
#   ),
#   error = function(e) { message("caught error: ", e$message); NULL }
# )
# message("t2 is NULL: ", is.null(t2))
# 
# # Test 3: simulate what get_wcc does — bind a real day + NULL + real day
# message("--- Test 3 (WCC): normalize_types + bind_rows across mixed results ---")
# raw_list_test <- list(t1, NULL, t1)
# combined <- bind_rows(lapply(compact(raw_list_test), normalize_types))
# message("combined rows: ", nrow(combined))
# message("all character: ", all(sapply(combined, is.character)))
# 
# # Test 4: simulate resume — write a checkpoint then read it back
# message("--- Test 4 (WCC): checkpoint write + resume ---")
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
# # ===========================================================================
# # LCD / MADIS SPLIT TESTS (new)
# # ===========================================================================
# 
# # Test 5: LCD file-exists skip — write a dummy LCD file and confirm it is
# #         loaded rather than re-downloaded
# message("--- Test 5 (LCD): file-exists skip ---")
# dummy_lcd_file <- file.path(
#   test_out_dir,
#   paste0("lcd_", format(start_utc, "%Y%m%d"), "_",
#          format(lcd_end_utc, "%Y%m%d"), ".csv")
# )
# # Write a tiny stand-in so the guard fires
# write_csv(tibble(id = "DUMMY", datetime = as_datetime("2023-01-01", tz = "UTC"),
#                  temp_air = 1.0),
#           dummy_lcd_file)
# message("dummy LCD file written: ", file.exists(dummy_lcd_file))
# # Simulate the guard logic from the main loop
# if (file.exists(dummy_lcd_file)) {
#   loaded_lcd <- read_csv(dummy_lcd_file, show_col_types = FALSE)
#   message("LCD skip guard fired correctly — loaded ", nrow(loaded_lcd), " row(s) from disk")
# } else {
#   stop("Test 5 FAILED: guard did not fire")
# }
# file.remove(dummy_lcd_file)
# 
# # Test 6: MADIS single-hour pull — one observation window, should return data
# #         Use a date well within the post-deprecation window
# message("--- Test 6 (MADIS): single-hour pull ---")
# t6 <- tryCatch(
#   download_meteo_madis(
#     lon_obs          = test_lon,
#     lat_obs          = test_lat,
#     deg_filter       = test_deg_filter,
#     datetime_utc_obs = as_datetime("2025-09-01 12:00:00", tz = "UTC")
#   ),
#   error = function(e) { message("caught error: ", e$message); NULL }
# )
# if (!is.null(t6)) {
#   obs_df <- t6$observations
#   message("MADIS obs rows: ", nrow(obs_df),
#           "  cols: ", paste(names(obs_df), collapse = ", "))
#   # Confirm preprocess_meteo works on the result
#   pp6 <- preprocess_meteo("MADIS", obs_df)
#   message("preprocessed rows: ", nrow(pp6),
#           "  cols: ", paste(names(pp6), collapse = ", "))
# } else {
#   message("Test 6: MADIS returned NULL (network issue or no data for window)")
# }
# 
# # Test 7: download_madis_range over a short 3-hour window — exercises the
# #         hourly loop, retry logic, and bind_rows compaction
# message("--- Test 7 (MADIS): 3-hour range via download_madis_range ---")
# t7 <- download_madis_range(
#   start_utc     = as_datetime("2025-09-01 10:00:00", tz = "UTC"),
#   end_utc       = as_datetime("2025-09-01 12:00:00", tz = "UTC"),
#   lon_obs       = test_lon,
#   lat_obs       = test_lat,
#   deg_filter    = test_deg_filter,
#   dist_thresh_m = test_dist_thresh_m
# )
# message("MADIS range rows: ", nrow(t7))
# message("MADIS range cols: ", paste(names(t7), collapse = ", "))
# if (nrow(t7) > 0) {
#   message("id class    : ", class(t7$id))
#   message("temp_air rng: ", round(min(t7$temp_air, na.rm = TRUE), 1),
#           " – ", round(max(t7$temp_air, na.rm = TRUE), 1), " °C")
# }
# 
# # Test 8: MADIS expected failure — date far in future should return empty / NULL
# message("--- Test 8 (MADIS): expected empty result for future date ---")
# t8 <- tryCatch(
#   download_madis_range(
#     start_utc     = as_datetime("2030-01-01 00:00:00", tz = "UTC"),
#     end_utc       = as_datetime("2030-01-01 01:00:00", tz = "UTC"),
#     lon_obs       = test_lon,
#     lat_obs       = test_lat,
#     deg_filter    = test_deg_filter,
#     dist_thresh_m = test_dist_thresh_m
#   ),
#   error = function(e) { message("caught error: ", e$message); NULL }
# )
# message("t8 rows (expect 0 or NULL): ",
#         if (is.null(t8)) "NULL" else nrow(t8))
# 
# # Test 9: lcd_madis_df bind — confirm LCD and MADIS rows combine cleanly
# message("--- Test 9 (combined): lcd_madis_df bind ---")
# dummy_lcd  <- tibble(id = "LCD_STA",   datetime = as_datetime("2024-01-01", tz = "UTC"),
#                      temp_air = 2.0, temp_dew = -1.0, rh = 80.0, ppt = 0.0)
# dummy_madis <- tibble(id = "MADIS_STA", datetime = as_datetime("2025-09-01", tz = "UTC"),
#                       temp_air = 5.0, temp_dew =  1.0, rh = 75.0)
# combined_test <- bind_rows(dummy_lcd, dummy_madis)
# message("combined rows: ", nrow(combined_test), "  (expect 2)")
# message("ids: ", paste(combined_test$id, collapse = ", "))
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
  
  out_dir <- file.path(repo_root, "Data", "Stations", region_id)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  
  # ---- Station selection ------------------------------------------------
  # message("Selecting stations...")
  # stations_hads <- station_select("HADS", cfg$lon_obs, cfg$lat_obs,
  #                                 cfg$deg_filter, cfg$dist_thresh_m)
  # stations_lcd  <- station_select("LCD",  cfg$lon_obs, cfg$lat_obs,
  #                                 cfg$deg_filter, cfg$dist_thresh_m)
  # stations_wcc  <- station_select("WCC",  cfg$lon_obs, cfg$lat_obs,
  #                                 cfg$deg_filter, cfg$dist_thresh_m)
  # 
  # message("  HADS stations : ", nrow(stations_hads))
  # message("  LCD stations  : ", nrow(stations_lcd))
  # message("  WCC stations  : ", nrow(stations_wcc))
  
  
  # ---- HADS (full period, unchanged) ------------------------------------
  # message("\nDownloading HADS...")
  # hads_out <- download_batched("HADS", start_utc, end_utc, stations_hads,
  #                              batch_size = 50)
  # hads_df  <- hads_out$data
  # write_csv(hads_df, file.path(out_dir, paste0("hads_", date_suffix, ".csv")))
  # if (nrow(hads_out$errors) > 0)
  #   write_csv(hads_out$errors, file.path(out_dir, "hads_error_log.csv"))
  hads_df <- read_csv(file.path(out_dir, paste0("hads_", date_suffix, ".csv")), show_col_types = FALSE)
  
  
  # # ---- LCD (pre-deprecation only) ---------------------------------------
  # # The LCD file covers 2022-10-01 → 2025-08-26.
  # # If that file already exists on disk, skip the download entirely.
  # lcd_date_suffix <- paste0(
  #   format(start_utc,    "%Y%m%d"), "_",
  #   format(lcd_end_utc,  "%Y%m%d")
  # )
  # lcd_file <- file.path(out_dir, paste0("lcd_", lcd_date_suffix, ".csv"))
  # 
  # if (file.exists(lcd_file)) {
  #   message("\nLCD file already exists — loading from disk: ", basename(lcd_file))
  #   lcd_df <- read_csv(lcd_file, show_col_types = FALSE)
  # } else {
  #   message("\nDownloading LCD (", format(start_utc, "%Y-%m-%d"),
  #           " → ", format(lcd_end_utc, "%Y-%m-%d"), ")...")
  #   lcd_out <- download_batched("LCD", start_utc, lcd_end_utc, stations_lcd,
  #                               batch_size = 10)
  #   lcd_df  <- lcd_out$data
  #   write_csv(lcd_df, lcd_file)
  #   if (nrow(lcd_out$errors) > 0)
  #     write_csv(lcd_out$errors, file.path(out_dir, "lcd_error_log.csv"))
  # }
  lcd_df <- read_csv(file.path(out_dir, paste0("lcd_", date_suffix, ".csv")), show_col_types = FALSE)
  
  
  
  # ---- MADIS (post-deprecation: 2025-08-27 → end) ----------------------
  # Only run if our overall end date is after the LCD cutoff.
  if (end_utc >= lcd_cutoff_utc) {
    
    madis_date_suffix <- paste0(
      format(madis_start_utc, "%Y%m%d"), "_",
      format(end_utc,         "%Y%m%d")
    )
    madis_file <- file.path(out_dir, paste0("madis_", madis_date_suffix, ".csv"))
    
    if (file.exists(madis_file)) {
      message("\nMADIS file already exists — loading from disk: ", basename(madis_file))
      madis_df <- read_csv(madis_file, show_col_types = FALSE)
    } else {
      message("\nDownloading MADIS (", format(madis_start_utc, "%Y-%m-%d"),
              " → ", format(end_utc, "%Y-%m-%d"), ")...")
      madis_df <- download_madis_range(
        start_utc     = madis_start_utc,
        end_utc       = end_utc,
        lon_obs       = cfg$lon_obs,
        lat_obs       = cfg$lat_obs,
        deg_filter    = cfg$deg_filter,
        dist_thresh_m = cfg$dist_thresh_m
      )
      write_csv(madis_df, madis_file)
    }
    
    # Combined LCD + MADIS for downstream use (mirrors what lcd_df alone did)
    lcd_madis_df <- bind_rows(lcd_df, madis_df)
    
  } else {
    message("\nEnd date is before LCD cutoff — no MADIS download needed.")
    lcd_madis_df <- lcd_df
    madis_df     <- tibble()   # empty; keeps station-metadata id union clean
  }
  
  
  # ---- WCC (full period, unchanged) -------------------------------------
  # message("\nDownloading WCC (this will take a while)...")
  # wcc_df <- get_wcc_awdb(start_utc, end_utc, stations_wcc, out_dir)
  # write_csv(wcc_df, file.path(out_dir, paste0("wcc_", date_suffix, ".csv")))
  wcc_df <- read_csv(file.path(out_dir, paste0("wcc_", date_suffix, ".csv")), show_col_types = FALSE)
  
  
  # ---- Station metadata -------------------------------------------------
  message("\nFetching station metadata...")
  all_ids       <- unique(c(hads_df$id, lcd_madis_df$id, wcc_df$id))
  stations_meta <- gather_meta(all_ids)
  write_csv(
    stations_meta,
    file.path(out_dir, paste0("station_metadata_", date_suffix, "_v3.csv"))
  )
  
  message("  Saved ", nrow(stations_meta), " station records -> ", out_dir)
}

message("\nAll regions complete.")

# Note: temp_dew and temp_wet are not calculated here via model_meteo;
# they are filled in downstream in preprocessing.ipynb