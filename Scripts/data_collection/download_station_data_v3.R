###############################################################
## download_station_data_v5.R
##
## Collect HADS, IEM ASOS (ASOS/AWOS/METAR) & WCC station data
## for two regions:
##   - Sierra Nevada / Lake Tahoe (CA)
##   - Colorado Mountains (CO)
##
## Period: 2022-10-01 00:00:00 UTC  →  2026-05-01 23:59:59 UTC
##
## IEM ASOS replaces both LCD (deprecated 2025-08-27) and MADIS.
## All ASOS data is pulled via IEM in daily batches for the full
## period — the same server and approach used for HADS in
## rainOrSnowTools/R/meteo_access.R.
##
## Resume logic:
##   - If an existing ASOS output file is found, its max datetime
##     is read and only the remaining hours are downloaded.
##   - This applies per-region; HADS and WCC use the same pattern.
##
## Note: temp_wet is not available from IEM ASOS (no wet-bulb
##   element). It is filled downstream in preprocessing.ipynb,
##   identical to prior treatment of missing wet-bulb in LCD.
##
## Outputs saved to:
##   <repo>/Data/Stations/CA/
##   <repo>/Data/Stations/CO/
###############################################################


## ---------------- 1.  SET-UP ---------------------------------

pkg_needed <- c("devtools", "tidyverse", "lubridate", "purrr",
                "progress", "readr", "jsonlite", "geosphere", "riem", "httr")
inst <- pkg_needed[!(pkg_needed %in% installed.packages()[, "Package"])]
if (length(inst)) install.packages(inst, repos = "https://cloud.r-project.org")

library(devtools)
library(tidyverse)
library(lubridate)
library(purrr)
library(progress)
library(readr)
library(jsonlite)
library(geosphere)
library(riem)
library(httr)


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
mros_path  <- normalizePath(file.path(repo_root, "..", "rainOrSnowTools"),
                            mustWork = TRUE)

devtools::load_all(mros_path)


## ---------------- 2.  PERIOD ---------------------------------

start_utc <- as_datetime("2022-10-01 00:00:00", tz = "UTC")
end_utc   <- as_datetime("2026-05-01 23:59:59", tz = "UTC")

date_suffix <- paste0(
  format(start_utc, "%Y%m%d"), "_",
  format(end_utc,   "%Y%m%d")
)

# LCD is only available until this timestamp
lcd_end_utc     <- as_datetime("2025-08-26 23:59:59", tz = "UTC")
lcd_date_suffix <- paste0(format(start_utc, "%Y%m%d"), "_",
                          format(lcd_end_utc, "%Y%m%d"))
asos_start_utc <- as_datetime("2025-08-27 00:00:00", tz = "UTC")
asos_date_suffix <- paste0(format(asos_start_utc, "%Y%m%d"), "_",
                           format(end_utc,         "%Y%m%d"))

message("Date range: ", format(start_utc, "%Y-%m-%d"),
        " → ", format(end_utc, "%Y-%m-%d"))


## ---------------- 3.  REGION CONFIG --------------------------

REGIONS <- list(
  CA = list(
    label         = "california",
    asos_network  = "CA_ASOS",
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
    asos_network  = "CO_ASOS",
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

# ---- IEM station selection -------------------------------------------
# Mirrors station_select("HADS", ...) from meteo_access.R but uses the
# riem network metadata instead of the built-in hads_meta table.
#
# Returns a tibble with at least columns: id, lon, lat
# (lon/lat used downstream for gather_meta; id used for URL building)

station_select_asos <- function(network, lon_obs, lat_obs,
                                deg_filter, dist_thresh_m) {
  message("  Fetching IEM station list for network: ", network)
  meta <- riem::riem_stations(network = network)
  
  if (is.null(meta) || nrow(meta) == 0)
    stop("No stations returned for IEM network: ", network)
  
  meta %>%
    dplyr::select(id, lon, lat, elev = elevation, tzname) %>%
    dplyr::filter(!is.na(lon), !is.na(lat)) %>%
    dplyr::filter(
      lon >= lon_obs - deg_filter & lon <= lon_obs + deg_filter,
      lat >= lat_obs - deg_filter & lat <= lat_obs + deg_filter
    ) %>%
    dplyr::rowwise() %>%
    dplyr::mutate(dist = geosphere::distHaversine(
      c(lon_obs, lat_obs), c(lon, lat))) %>%
    dplyr::ungroup() %>%
    dplyr::filter(dist <= dist_thresh_m)
}


# ---- IEM ASOS bulk downloader ----------------------------------------
# One request per batch covers the requested date range — mirrors the
# download_meteo_hads() pattern in meteo_access.R (bulk range, not hourly).
#
# IEM endpoint : mesonet.agron.iastate.edu/cgi-bin/request/asos.py
# report_type=3: routine hourly observations (FM-15 equivalent)
# Variables    : tmpc (°C), dwpc (°C), relh (%), p01i (precip, inches)
# Note         : temp_wet is absent from IEM ASOS; filled downstream.
# Rate limiting: IEM enforces ~1 req/s; sleep_base=5 s is conservative.
# Batch size   : 50 stations × ~3.6 yr ≈ 180 stn-yrs — well within the
#                IEM 1000 stn-year limit.
#
# Returns list(data = <tibble>, errors = <tibble>)

download_meteo_asos_iem <- function(datetime_utc_start, datetime_utc_end,
                                    stations,
                                    batch_size  = 50,
                                    max_retries = 3,
                                    sleep_base  = 5) {
  n         <- nrow(stations)
  n_batches <- ceiling(n / batch_size)
  message("  IEM ASOS: ", n, " stations across ", n_batches,
          " batches of ", batch_size)
  
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
        sta_params <- paste0("&station=", batch$id, collapse = "")
        url <- paste0(
          "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?",
          "data=tmpc&data=dwpc&data=relh&data=p01i",
          sta_params,
          "&sts=", format(datetime_utc_start, "%Y-%m-%dT%H:%M:%SZ"),
          "&ets=", format(datetime_utc_end,   "%Y-%m-%dT%H:%M:%SZ"),
          "&tz=Etc/UTC&format=onlycomma&latlon=no&elev=no",
          "&missing=M&trace=0.0001&direct=yes&report_type=3"
        )
        
        df <- readr::read_csv(url,
                              col_types = readr::cols(.default = "c"),
                              comment   = "#",
                              show_col_types = FALSE)
        
        if (nrow(df) == 0) return(NULL)
        
        df <- df %>%
          dplyr::rename(id = station, datetime = valid) %>%
          dplyr::mutate(
            datetime = as.POSIXct(datetime,
                                  format = "%Y-%m-%d %H:%M",
                                  tz     = "UTC"),
            temp_air = suppressWarnings(as.numeric(tmpc)),
            temp_dew = suppressWarnings(as.numeric(dwpc)),
            rh       = suppressWarnings(as.numeric(relh)),
            ppt      = suppressWarnings(as.numeric(p01i)),
            ppt      = in_to_mm(ppt)   # inches → mm, matches LCD output
          ) %>%
          dplyr::select(id, datetime, temp_air, temp_dew, rh, ppt) %>%
          dplyr::filter(!is.na(temp_air))
        
        df <- df %>%
          dplyr::mutate(datetime = lubridate::floor_date(datetime, "hour")) %>%
          dplyr::group_by(id, datetime) %>%
          dplyr::slice_min(order_by = is.na(temp_air), n = 1, with_ties = FALSE) %>%
          dplyr::ungroup()
        
        df
        
      }, error = function(e) {
        message("  IEM ASOS batch ", i, "/", n_batches,
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
    
    if (!success)
      message("  IEM ASOS batch ", i, " failed all ", max_retries,
              " attempts — skipped.")
  }
  
  list(data = bind_rows(compact(results)), errors = error_log)
}


# ---- Resume-aware ASOS downloader ------------------------------------
# Wraps download_meteo_asos_iem with file-based resume logic:
#   - If the output file doesn't exist  → download the full period.
#   - If it exists but is incomplete    → append only the missing tail.
#   - If it exists and is complete      → load from disk, skip download.
#
# "Complete" means the file's max(datetime) >= end_utc - 1 hour.
# The append tolerance is 1 hour to account for rounding at boundaries.
#
# Returns the final (possibly merged) data frame.

download_asos_with_resume <- function(out_file, datetime_start, datetime_end,
                                      stations, ...) {
  tolerance <- 3600   # 1 hour, in seconds
  
  if (file.exists(out_file)) {
    existing <- readr::read_csv(out_file, show_col_types = FALSE) %>%
      dplyr::mutate(datetime = as.POSIXct(datetime, tz = "UTC"))
    
    max_dt <- suppressWarnings(max(existing$datetime, na.rm = TRUE))
    
    if (!is.finite(as.numeric(max_dt))) {
      message("  Existing file has no valid datetimes — re-downloading.")
    } else if (as.numeric(datetime_end) - as.numeric(max_dt) <= tolerance) {
      message("  ASOS file complete (max datetime: ",
              format(max_dt, "%Y-%m-%d %H:%M UTC"), ") — loading from disk.")
      return(existing)
    } else {
      # Partial file: resume from the hour after the last good record
      resume_start <- max_dt + 3600
      message("  ASOS file is partial (max datetime: ",
              format(max_dt, "%Y-%m-%d %H:%M UTC"),
              ") — resuming from ", format(resume_start, "%Y-%m-%d %H:%M UTC"))
      
      new_out <- download_meteo_asos_iem(resume_start, datetime_end,
                                         stations, ...)
      if (nrow(new_out$data) > 0) {
        merged <- dplyr::bind_rows(existing, new_out$data) %>%
          dplyr::distinct(id, datetime, .keep_all = TRUE) %>%
          dplyr::arrange(id, datetime)
        readr::write_csv(merged, out_file)
        if (nrow(new_out$errors) > 0)
          readr::write_csv(new_out$errors,
                           sub("\\.csv$", "_error_log.csv", out_file))
        message("  Appended ", nrow(new_out$data),
                " rows; file now has ", nrow(merged), " rows.")
        return(merged)
      } else {
        message("  No new data returned — using existing partial file.")
        return(existing)
      }
    }
  }
  
  # No file (or unreadable) — full download
  message("  Downloading IEM ASOS (",
          format(datetime_start, "%Y-%m-%d"), " → ",
          format(datetime_end,   "%Y-%m-%d"), ")...")
  out <- download_meteo_asos_iem(datetime_start, datetime_end, stations, ...)
  df  <- out$data
  readr::write_csv(df, out_file)
  if (nrow(out$errors) > 0)
    readr::write_csv(out$errors, sub("\\.csv$", "_error_log.csv", out_file))
  message("  Saved ", nrow(df), " rows → ", basename(out_file))
  df
}


# ---- Generic batched downloader (HADS) with resume -------------------

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

# Wraps download_batched from meteo_access.R with the same file-resume
# pattern used for ASOS above.
download_batched_with_resume <- function(source, out_file,
                                         datetime_start, datetime_end,
                                         stations, batch_size,
                                         max_retries = 3, sleep_base = 5) {
  tolerance <- 3600
  
  if (file.exists(out_file)) {
    existing <- readr::read_csv(out_file, show_col_types = FALSE) %>%
      dplyr::mutate(datetime = as.POSIXct(datetime, tz = "UTC"))
    
    max_dt <- suppressWarnings(max(existing$datetime, na.rm = TRUE))
    
    if (is.finite(as.numeric(max_dt)) &&
        as.numeric(datetime_end) - as.numeric(max_dt) <= tolerance) {
      message("  ", source, " file complete (max datetime: ",
              format(max_dt, "%Y-%m-%d %H:%M UTC"), ") — loading from disk.")
      return(existing)
    } else if (is.finite(as.numeric(max_dt))) {
      resume_start <- max_dt + 3600
      message("  ", source, " file is partial — resuming from ",
              format(resume_start, "%Y-%m-%d %H:%M UTC"))
      new_out <- download_batched(source, resume_start, datetime_end,
                                  stations, batch_size,
                                  max_retries, sleep_base)
      if (nrow(new_out$data) > 0) {
        merged <- dplyr::bind_rows(existing, new_out$data) %>%
          dplyr::distinct(id, datetime, .keep_all = TRUE) %>%
          dplyr::arrange(id, datetime)
        readr::write_csv(merged, out_file)
        if (nrow(new_out$errors) > 0)
          readr::write_csv(new_out$errors,
                           file.path(dirname(out_file),
                                     paste0(tolower(source), "_error_log.csv")))
        return(merged)
      } else {
        return(existing)
      }
    }
  }
  
  # Full download
  out <- download_batched(source, datetime_start, datetime_end,
                          stations, batch_size, max_retries, sleep_base)
  df  <- out$data
  readr::write_csv(df, out_file)
  if (nrow(out$errors) > 0)
    readr::write_csv(out$errors,
                     file.path(dirname(out_file),
                               paste0(tolower(source), "_error_log.csv")))
  message("  Saved ", nrow(df), " rows → ", basename(out_file))
  df
}


# ---- WCC AWDB downloader (unchanged) ---------------------------------
get_wcc_awdb <- function(start_utc, end_utc, stations, out_dir) {
  
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
  
  # Resume: skip years already checkpointed
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
        error = function(e) { message("    httr error: ", e$message); NULL }
      )
      
      if (is.null(resp)) {
        message("    Request failed on attempt ", attempt); next
      }
      
      if (httr::status_code(resp) == 200) {
        return(jsonlite::fromJSON(
          httr::content(resp, as = "text", encoding = "UTF-8"),
          simplifyVector = TRUE
        ))
      }
      
      message("    HTTP ", httr::status_code(resp), " on attempt ", attempt,
              " (", begin_date, " to ", end_date, ")")
    }
    message("    All retries failed"); return(NULL)
  }
  
  parse_station <- function(station_triplet, data_row) {
    if (is.null(data_row$values) || length(data_row$values) == 0) return(NULL)
    elements <- data_row$stationElement$elementCode
    dfs <- lapply(seq_along(elements), function(e) {
      vals <- data_row$values[[e]]
      if (is.null(vals) || nrow(vals) == 0) return(NULL)
      data.frame(
        station  = station_triplet,
        datetime = as.POSIXct(vals$date, format = "%Y-%m-%d %H:%M", tz = "UTC"),
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
      message("  [", call_n, "/", total_calls, "]  ", yr,
              "  batch ", b, "/", length(station_batches),
              " (", length(station_batches[[b]]), " stations)...")
      
      result <- fetch_awdb(station_batches[[b]], begin_date, end_date)
      
      if (is.null(result) || nrow(result) == 0) {
        message("    No data returned — skipping."); next
      }
      
      batch_long <- bind_rows(lapply(seq_len(nrow(result)), function(i) {
        parse_station(result$stationTriplet[i], result$data[[i]])
      }))
      
      all_long[[length(all_long) + 1]] <- batch_long
      Sys.sleep(1)
    }
    
    year_df         <- bind_rows(all_long)
    checkpoint_path <- file.path(
      out_dir, paste0("wcc_awdb_checkpoint_through_", yr, ".csv")
    )
    write_csv(year_df, checkpoint_path)
    message("  Checkpoint saved through ", yr,
            " -> ", basename(checkpoint_path))
  }
  
  long_df <- bind_rows(all_long)
  
  write_csv(long_df,
            file.path(out_dir, paste0("wcc_awdb_long_raw_",
                                      format(as_date(start_utc), "%Y%m%d"), "_",
                                      format(as_date(end_utc),   "%Y%m%d"), ".csv")))
  message("  Raw long data saved (", nrow(long_df), " rows)")
  
  if (nrow(long_df) == 0) { message("  No data retrieved."); return(NULL) }
  
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
  
  write_csv(wide_df,
            file.path(out_dir, paste0("wcc_awdb_raw_",
                                      format(as_date(start_utc), "%Y%m%d"), "_",
                                      format(as_date(end_utc),   "%Y%m%d"), ".csv")))
  message("  Retrieved ", nrow(wide_df), " rows for ",
          length(unique(wide_df$.id)), " stations.")
  
  wide_df %>% preprocess_meteo("WCC", .)
}


# ## ---------------- TESTING BLOCK (uncomment to run) -------------------
# 
# # Shared test config — CA region centroid
# test_lon           <- mean(c(-119.455, -121.279, -119.114, -118.499, -119.466))
# test_lat           <- mean(c(39.653,   39.662,   36.727,   37.236,   38.373))
# test_deg_filter    <- 2.25
# test_dist_thresh_m <- 350000
# test_out_dir       <- file.path(repo_root, "Data", "Stations", "CA")
# dir.create(test_out_dir, showWarnings = FALSE, recursive = TRUE)
# 
# # ===========================================================================
# # ASOS STATION SELECTION TEST
# # ===========================================================================
# 
# # Test 1: station_select_asos returns expected columns and sensible row count
# message("--- Test 1 (ASOS): station_select_asos ---")
# t1_sta <- station_select_asos("CA_ASOS", test_lon, test_lat,
#                               test_deg_filter, test_dist_thresh_m)
# message("Stations found: ", nrow(t1_sta))
# message("Cols: ", paste(names(t1_sta), collapse = ", "))
# stopifnot(nrow(t1_sta) > 0)
# stopifnot(all(c("id", "lon", "lat", "dist") %in% names(t1_sta)))
# 
# # ===========================================================================
# # IEM ASOS DOWNLOAD TESTS
# # ===========================================================================
# 
# # Test 2: short-window download — 2 stations, 3-day window
# message("--- Test 2 (ASOS): short range download ---")
# t2 <- download_meteo_asos_iem(
#   datetime_utc_start = as_datetime("2023-01-15 00:00:00", tz = "UTC"),
#   datetime_utc_end   = as_datetime("2023-01-17 23:59:59", tz = "UTC"),
#   stations           = t1_sta[1:2, ],
#   batch_size         = 2
# )
# message("ASOS rows: ", nrow(t2$data))
# message("ASOS cols: ", paste(names(t2$data), collapse = ", "))
# stopifnot(nrow(t2$data) > 0)
# stopifnot(all(c("id", "datetime", "temp_air", "temp_dew", "rh", "ppt")
#               %in% names(t2$data)))
# stopifnot(!"temp_wet" %in% names(t2$data))   # not available from IEM ASOS
# 
# # Verify observations are hourly (no sub-hourly duplicates slipping through)
# message("--- Test 2b (ASOS): verify hourly cadence ---")
# diffs <- t2$data %>%   # already floored at this point
#   dplyr::arrange(id, datetime) %>%
#   dplyr::group_by(id) %>%
#   dplyr::mutate(gap_hrs = as.numeric(difftime(datetime, lag(datetime), units = "hours"))) %>%
#   dplyr::filter(!is.na(gap_hrs))
# 
# # After flooring, all gaps should be exactly 1 hr (or > 1 for missing obs)
# bad_gaps <- diffs %>% dplyr::filter(gap_hrs < 1)
# message("Sub-hourly gaps after flooring: ", nrow(bad_gaps), "  (expect 0)")
# stopifnot(nrow(bad_gaps) == 0)
# 
# dupes <- t2$data %>%
#   dplyr::group_by(id, datetime) %>%
#   dplyr::filter(n() > 1)
# message("Duplicate station-hours after dedup: ", nrow(dupes), "  (expect 0)")
# stopifnot(nrow(dupes) == 0)
# 
# sub_hourly <- diffs %>% dplyr::filter(gap_hrs < 1)
# message("Sub-hourly gaps found: ", nrow(sub_hourly), "  (expect 0)")
# stopifnot(nrow(sub_hourly) == 0)
# 
# # Also confirm no station has more than 1 obs per hour
# dupes <- t2$data %>%
#   dplyr::mutate(hour_floor = lubridate::floor_date(datetime, "hour")) %>%
#   dplyr::group_by(id, hour_floor) %>%
#   dplyr::filter(n() > 1)
# message("Duplicate station-hours: ", nrow(dupes), "  (expect 0)")
# stopifnot(nrow(dupes) == 0)
# 
# # Inspect the sub-hourly gaps
# print(sub_hourly %>% dplyr::select(id, datetime, gap_hrs) %>% dplyr::arrange(id, datetime), n = 20)
# 
# # And see what the raw data looks like around those timestamps
# problem_ids <- unique(sub_hourly$id)
# t2$data %>%
#   dplyr::filter(id %in% problem_ids) %>%
#   dplyr::arrange(id, datetime) %>%
#   print(n = 50)
# 
# # ===========================================================================
# # RESUME LOGIC TESTS
# # ===========================================================================
# 
# # Test 3: complete-file path — download_asos_with_resume loads from disk
# message("--- Test 3 (resume): complete file detected ---")
# test_asos_file <- file.path(test_out_dir, paste0("asos_test_", date_suffix, ".csv"))
# complete_dummy <- tibble(
#   id       = "KTRK",
#   datetime = end_utc,
#   temp_air = 5.0, temp_dew = 1.0, rh = 80.0, ppt = 0.0
# )
# write_csv(complete_dummy, test_asos_file)
# loaded <- download_asos_with_resume(test_asos_file, start_utc, end_utc,
#                                     t1_sta[1:2, ])
# stopifnot(nrow(loaded) == 1)
# file.remove(test_asos_file)
# message("  Resume (complete) guard fired correctly.")
# 
# # Test 4: partial-file path — download_asos_with_resume appends new data
# message("--- Test 4 (resume): partial file detected ---")
# partial_end  <- as_datetime("2023-01-17 23:00:00", tz = "UTC")
# partial_file <- file.path(test_out_dir, paste0("asos_partial_test.csv"))
# partial_dummy <- tibble(
#   id       = t1_sta$id[1],
#   datetime = partial_end,
#   temp_air = 5.0, temp_dew = 1.0, rh = 80.0, ppt = 0.0
# )
# write_csv(partial_dummy, partial_file)
# resumed <- download_asos_with_resume(
#   partial_file,
#   datetime_start = as_datetime("2023-01-15 00:00:00", tz = "UTC"),
#   datetime_end   = as_datetime("2023-01-20 23:59:59", tz = "UTC"),
#   stations       = t1_sta[1:2, ],
#   batch_size     = 2
# )
# message("Resumed rows: ", nrow(resumed), "  (expect > 1)")
# stopifnot(nrow(resumed) > 1)
# file.remove(partial_file)
# message("  Resume (partial) append worked correctly.")
# 
# message("--- All tests passed ---")
# # ---------------- END TESTING BLOCK ---------------------------------


## ---------------- 5.  MAIN LOOP — one run per region ---------

for (region_id in names(REGIONS)) {
  cfg <- REGIONS[[region_id]]
  
  message("\n", strrep("=", 55))
  message("  Region: ", region_id, " (", cfg$label, ")")
  message("  Centre: ", round(cfg$lon_obs, 3), ", ", round(cfg$lat_obs, 3))
  message(strrep("=", 55))
  
  out_dir <- file.path(repo_root, "Data", "Stations", region_id)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  
  
  # ---- Station selection -----------------------------------------------
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
    dplyr::filter(dist <= cfg$dist_thresh_m) %>%
    dplyr::rename(lon = LONGITUDE, lat = LATITUDE)
  ## DEPRECATED:
  # stations_lcd  <- station_select("LCD",  cfg$lon_obs, cfg$lat_obs,
  #                                 cfg$deg_filter, cfg$dist_thresh_m)
  stations_asos <- station_select_asos(cfg$asos_network,
                                       cfg$lon_obs, cfg$lat_obs,
                                       cfg$deg_filter, cfg$dist_thresh_m)
  stations_wcc  <- station_select("WCC",  cfg$lon_obs, cfg$lat_obs,
                                  cfg$deg_filter, cfg$dist_thresh_m)
  
  message("  HADS stations : ", nrow(stations_hads))
  message("  LCD stations  : ", nrow(stations_lcd))
  message("  ASOS stations : ", nrow(stations_asos))
  message("  WCC stations  : ", nrow(stations_wcc))
  
  
  # ---- HADS (full period) ----------------------------------------------
  message("\nDownloading HADS (with resume)...")
  hads_file <- file.path(out_dir, paste0("hads_", date_suffix, ".csv"))
  hads_df   <- download_batched_with_resume(
    source         = "HADS",
    out_file       = hads_file,
    datetime_start = start_utc,
    datetime_end   = end_utc,
    stations       = stations_hads,
    batch_size     = 50
  )

  ## ------------ LCD ASOS --------------
  # Only available up until August 2025
  # LCD — batched with retry (smaller batches due to API timeout sensitivity)
  message("\nDownloading LCD (with resume)...")
  lcd_file <- file.path(out_dir, paste0("lcd_", lcd_date_suffix, ".csv"))
  lcd_df   <- download_batched_with_resume(
    source         = "LCD",
    out_file       = lcd_file,
    datetime_start = start_utc,
    datetime_end   = lcd_end_utc,
    stations       = stations_lcd,
    batch_size     = 10
  )
  
  
  # ---- IEM ASOS (full period, replaces LCD + MADIS) --------------------
  # All ASOS/AWOS/METAR data for the full study period comes from IEM.
  # Resume logic in download_asos_with_resume handles partial files.
  # temp_wet is not available from IEM ASOS; filled downstream.
  message("\nDownloading IEM ASOS (with resume)...")
  asos_file <- file.path(out_dir, paste0("asos_", asos_date_suffix, ".csv"))
  asos_df   <- download_asos_with_resume(
    out_file       = asos_file,
    datetime_start = asos_start_utc,
    datetime_end   = end_utc,
    stations       = stations_asos
  )
  
  
  # ---- WCC (full period) -----------------------------------------------
  message("\nDownloading WCC (this will take a while)...")
  wcc_file <- file.path(out_dir, paste0("wcc_", date_suffix, ".csv"))
  
  if (file.exists(wcc_file)) {
    message("  WCC file found — loading from disk: ", basename(wcc_file))
    wcc_df <- read_csv(wcc_file, show_col_types = FALSE)
  } else {
    wcc_df <- get_wcc_awdb(start_utc, end_utc, stations_wcc, out_dir)
    write_csv(wcc_df, wcc_file)
  }
  
  
  # ---- Station metadata ------------------------------------------------
  message("\nFetching station metadata...")
  
  # HADS and WCC go through gather_meta as before
  meta_hads <- gather_meta(stations_hads, network = "HADS")
  meta_wcc  <- gather_meta(stations_wcc,  network = "WCC")
  
  # LCD and ASOS built directly from selection objects — gather_meta 
  # doesn't recognize their ID formats
  meta_lcd  <- stations_lcd %>%
    dplyr::transmute(
      name         = id,
      id           = id,
      lat          = lat,
      lon          = lon,
      elev         = ELEVATION_.M.,
      timezone_lst = timezone_lst,
      network      = "LCD"
    )
  
  meta_asos <- stations_asos %>%
    dplyr::transmute(
      name         = id,
      id           = id,
      lat          = lat,
      lon          = lon,
      elev         = elev,
      timezone_lst = tzname,
      network      = "ASOS"
    )
  
  stations_meta <- bind_rows(meta_hads, meta_lcd, meta_asos, meta_wcc) %>%
    dplyr::distinct(id, .keep_all = TRUE)
  
  write_csv(
    stations_meta,
    file.path(out_dir, paste0("station_metadata_", date_suffix, "_v3.csv"))
  )
  message("  Saved ", nrow(stations_meta), " station records -> ", out_dir)
}

message("\nAll regions complete.")

# Note: temp_dew and temp_wet are not calculated here via model_meteo;
# they are filled in downstream in preprocessing.ipynb.
# temp_wet is absent from IEM ASOS output (no wet-bulb element available);
# downstream handling is identical to how missing temp_wet was handled
# for LCD observations that failed the wet-bulb QC flag.



## ---------------- 6.  DATA COMPLETENESS DIAGNOSTICS ---------
## Run after the main loop finishes.
## Produces:
##   - Console summary table (n_stations, pct_complete, gaps >6h, max gap)
##   - coverage_heatmap_{region}.png   — station × month fill rate
##   - wcc_gap_timeline_{region}.png   — per-station gap audit for WCC

library(ggplot2)
library(tidyr)
library(dplyr)
library(lubridate)
library(readr)
library(purrr)

## expected obs per hour (1) × hours in each month
expected_hourly <- function(yr, mo) {
  days_in_month(make_date(yr, mo, 1)) * 24L
}

## ---- helper: load one CSV, tag with source name -------------------
load_source <- function(path, source_name) {
  if (!file.exists(path)) {
    message("  [completeness] file not found, skipping: ", basename(path))
    return(NULL)
  }
  df <- read_csv(path, show_col_types = FALSE) %>%
    mutate(datetime = as.POSIXct(datetime, tz = "UTC"),
           source   = source_name)
  message("  [completeness] loaded ", nrow(df), " rows from ", basename(path))
  df
}

## ---- helper: per-station-month fill rate -------------------------
fill_rate <- function(df) {
  df %>%
    filter(!is.na(datetime)) %>%
    mutate(yr = year(datetime), mo = month(datetime)) %>%
    group_by(source, id, yr, mo) %>%
    summarise(n_obs = n(), .groups = "drop") %>%
    mutate(
      ym       = make_date(yr, mo, 1),
      expected = expected_hourly(yr, mo),
      pct      = pmin(n_obs / expected * 100, 100)
    )
}

## ---- helper: gap audit -------------------------------------------
gap_audit <- function(df, threshold_h = 6) {
  df %>%
    filter(!is.na(datetime)) %>%
    arrange(id, datetime) %>%
    group_by(id) %>%
    mutate(gap_h = as.numeric(difftime(datetime, lag(datetime), units = "hours"))) %>%
    filter(!is.na(gap_h), gap_h > threshold_h) %>%
    mutate(gap_start = lag(datetime)) %>%
    select(source, id, gap_start, gap_end = datetime, gap_h) %>%
    ungroup()
}

## ---- helper: summary row for console ----------------------------
summary_row <- function(df, region) {
  gaps <- gap_audit(df)
  tibble(
    region       = region,
    source       = unique(df$source),
    n_stations   = n_distinct(df$id),
    pct_complete = round(mean(!is.na(df$temp_air %||% df$tair), na.rm = TRUE) * 100, 1),
    n_gaps_gt6h  = nrow(gaps),
    max_gap_h    = if (nrow(gaps) > 0) round(max(gaps$gap_h), 1) else 0
  )
}

## ---- operator for purrr-safe NULL coalesce ----------------------
`%||%` <- function(a, b) if (!is.null(a)) a else b

## ---- main diagnostic loop ----------------------------------------
summary_rows <- list()

for (region_id in names(REGIONS)) {
  out_dir <- file.path(repo_root, "Data", "Stations", region_id)
  message("\n[completeness] Region: ", region_id)
  
  ## --- load files ---------------------------------------------------
  hads_df <- load_source(
    file.path(out_dir, paste0("hads_", date_suffix, ".csv")), "HADS")
  lcd_df  <- load_source(
    file.path(out_dir, paste0("lcd_",  lcd_date_suffix, ".csv")), "LCD")
  asos_df <- load_source(
    file.path(out_dir, paste0("asos_", asos_date_suffix, ".csv")), "ASOS")
  wcc_df  <- load_source(
    file.path(out_dir, paste0("wcc_",  date_suffix, ".csv")), "WCC")
  
  ## --- normalise id column across all sources ----------------------
  coerce_id <- function(df) {
    if (is.null(df)) return(NULL)
    if (".id" %in% names(df)) df <- rename(df, id = .id)
    df %>% mutate(id = as.character(id))
  }
  
  hads_df <- coerce_id(hads_df)
  lcd_df  <- coerce_id(lcd_df)
  asos_df <- coerce_id(asos_df)
  wcc_df  <- coerce_id(wcc_df)
  
  ## standardise the WCC column name so gap_audit finds it
  if (!is.null(wcc_df) && "tair" %in% names(wcc_df))
    wcc_df <- rename(wcc_df, temp_air = tair)
  
  all_sources <- compact(list(hads_df, lcd_df, asos_df, wcc_df))
  if (length(all_sources) == 0) next
  all_df <- bind_rows(all_sources)
  
  ## --- console summary ---------------------------------------------
  for (src_df in all_sources) {
    summary_rows[[length(summary_rows) + 1]] <- summary_row(src_df, region_id)
  }
  
  ## --- Plot 1: coverage distribution (histogram) -------------------
  fr_summary <- fr %>%
    group_by(source, id) %>%
    summarise(mean_pct = mean(pct), .groups = "drop")
  
  p_hist <- ggplot(fr_summary, aes(x = mean_pct, fill = source)) +
    geom_histogram(binwidth = 5, colour = "white", linewidth = 0.3) +
    facet_wrap(~ source, ncol = 1, scales = "free_y") +
    scale_x_continuous(limits = c(0, 100), breaks = seq(0, 100, 20),
                       labels = function(x) paste0(x, "%")) +
    scale_fill_manual(values = c(
      HADS = "#2980b9", LCD = "#8e44ad", ASOS = "#27ae60", WCC = "#e67e22"
    ), guide = "none") +
    labs(
      title    = paste0(region_id, " — mean monthly coverage per station"),
      subtitle = "Each bar = number of stations at that average fill rate",
      x        = "Mean % of expected hourly obs present",
      y        = "Station count"
    ) +
    theme_minimal(base_size = 10) +
    theme(
      strip.text       = element_text(face = "bold"),
      panel.grid.minor = element_blank(),
      axis.text.x      = element_text(angle = 0)
    )
  
  hist_path <- file.path(out_dir, paste0("coverage_hist_", region_id, ".png"))
  ggsave(hist_path, p_hist, width = 8, height = 7, dpi = 150)
  message("  [completeness] saved: ", basename(hist_path))
  
  ## --- Plot 2: WCC gap length distribution -------------------------
  if (!is.null(wcc_df)) {
    wcc_gaps <- gap_audit(wcc_df, threshold_h = 6)
    
    if (nrow(wcc_gaps) > 0) {
      p_gap <- ggplot(wcc_gaps, aes(x = gap_h)) +
        geom_histogram(binwidth = 12, fill = "#c0392b", colour = "white",
                       linewidth = 0.3) +
        geom_vline(xintercept = 24,  linetype = "dashed",
                   colour = "#7f8c8d", linewidth = 0.5) +
        geom_vline(xintercept = 168, linetype = "dashed",
                   colour = "#2c3e50", linewidth = 0.5) +
        annotate("text", x = 25,  y = Inf, label = "24 h",
                 hjust = 0, vjust = 1.4, size = 3, colour = "#7f8c8d") +
        annotate("text", x = 169, y = Inf, label = "1 week",
                 hjust = 0, vjust = 1.4, size = 3, colour = "#2c3e50") +
        scale_x_continuous(breaks = seq(0, max(wcc_gaps$gap_h) + 24, by = 48)) +
        labs(
          title    = paste0(region_id, " — WCC gap lengths (gaps > 6 h)"),
          subtitle = paste0(nrow(wcc_gaps), " gaps across ",
                            n_distinct(wcc_gaps$id), " stations"),
          x        = "Gap length (hours)",
          y        = "Count"
        ) +
        theme_minimal(base_size = 10) +
        theme(panel.grid.minor = element_blank())
    } else {
      p_gap <- ggplot() +
        annotate("text", x = 0.5, y = 0.5,
                 label = paste0(region_id, " WCC: no gaps > 6 h"),
                 size = 5) +
        theme_void()
    }
    
    gap_path <- file.path(out_dir, paste0("wcc_gap_hist_", region_id, ".png"))
    ggsave(gap_path, p_gap, width = 8, height = 4, dpi = 150)
    message("  [completeness] saved: ", basename(gap_path))
  }
 
}
summary_tbl <- bind_rows(summary_rows)
print(as.data.frame(summary_tbl), row.names = FALSE)
