## Collect HADS, LCD and WCC (SNOTEL) station observations for both
## study areas over the full study period.
##
## Station selection and post-processing come from the rainOrSnowTools package,
## which must be checked out next to this repo:
##   https://github.com/LynkerIntel/rainOrSnowTools
##
## Output: Data/Stations/<REGION>/{hads,lcd,wcc,station_metadata}_<dates>.csv
##
## temp_dew and temp_wet are not computed here; compile_observations.py
## derives them from the hourly record.

suppressPackageStartupMessages({
  library(devtools)
  library(tidyverse)
  library(lubridate)
  library(purrr)
  library(readr)
  library(httr)
  library(jsonlite)
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

REPO_ROOT <- repo_root()
devtools::load_all(normalizePath(file.path(REPO_ROOT, "..", "rainOrSnowTools"),
                                 mustWork = TRUE))

DATE_SUFFIX <- paste0(format(WY_START, "%Y%m%d"), "_", format(WY_END, "%Y%m%d"))

## Search radius around each region centre.
DEG_FILTER    <- 2.25
DIST_THRESH_M <- 350000


## ---------------------------------------------------------------------------
## Download helpers
## ---------------------------------------------------------------------------

## Batched downloader with per-batch retry and increasing backoff.
## Used for HADS and LCD, which both take a stations tibble.
download_batched <- function(source, start_utc, end_utc, stations,
                             batch_size, max_retries = 3, sleep_base = 5) {
  n         <- nrow(stations)
  n_batches <- ceiling(n / batch_size)
  message("  ", source, ": ", n, " stations in ", n_batches, " batches")

  download_fn <- switch(source,
    HADS = download_meteo_hads,
    LCD  = download_meteo_lcd,
    stop("Unknown source: ", source)
  )

  error_log <- tibble(batch = integer(), stations = character(),
                      attempt = integer(), error = character())
  results <- vector("list", n_batches)

  for (i in seq_len(n_batches)) {
    idx   <- ((i - 1) * batch_size + 1):min(i * batch_size, n)
    batch <- stations[idx, ]

    for (attempt in seq_len(max_retries)) {
      Sys.sleep(sleep_base * attempt)

      results[[i]] <- tryCatch({
        preprocess_meteo(source, download_fn(start_utc, end_utc, batch))
      }, error = function(e) {
        message("  ", source, " batch ", i, "/", n_batches,
                " attempt ", attempt, " failed: ", e$message)
        error_log <<- bind_rows(error_log, tibble(
          batch = i, stations = paste(batch$id, collapse = ","),
          attempt = attempt, error = e$message
        ))
        NULL
      })

      if (!is.null(results[[i]])) break
    }

    if (is.null(results[[i]])) {
      message("  ", source, " batch ", i, " gave up after ", max_retries,
              " attempts.")
    }
  }

  list(data = bind_rows(compact(results)), errors = error_log)
}


## WCC/SNOTEL is fetched from the AWDB REST API rather than through
## rainOrSnowTools, with a checkpoint written after each year.
get_wcc_awdb <- function(start_utc, end_utc, stations, out_dir) {

  triplets <- stations %>%
    mutate(network_ab = case_when(
      network == "snotel"  ~ "SNTL",
      network == "snotelt" ~ "SNTLT",
      network == "scan"    ~ "SCAN",
      TRUE                 ~ toupper(network)
    )) %>%
    mutate(triplet = paste(station.id, state, network_ab, sep = ":")) %>%
    pull(triplet)

  message("  AWDB: ", length(triplets), " stations, ",
          as_date(start_utc), " to ", as_date(end_utc))

  years           <- seq(year(start_utc), year(end_utc))
  station_batches <- split(triplets, ceiling(seq_along(triplets) / 10))
  total_calls     <- length(years) * length(station_batches)

  ## Resume from any year already checkpointed on disk.
  completed_years <- list.files(
    out_dir, pattern = "wcc_awdb_checkpoint_through_.*\\.csv"
  ) %>%
    gsub("wcc_awdb_checkpoint_through_|\\.csv", "", .) %>%
    as.integer()

  if (length(completed_years) > 0) {
    message("  Resuming; already have ", paste(sort(completed_years), collapse = ", "))
    all_long <- list(bind_rows(lapply(
      file.path(out_dir, paste0("wcc_awdb_checkpoint_through_",
                                sort(completed_years), ".csv")),
      read_csv, show_col_types = FALSE
    )))
    years <- years[!(years %in% completed_years)]
  } else {
    all_long <- list()
  }

  fetch_awdb <- function(triplet_batch, begin_date, end_date, max_retries = 3) {
    for (attempt in seq_len(max_retries)) {
      Sys.sleep(c(2, 10, 30)[min(attempt, 3)])

      resp <- tryCatch(
        GET("https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data",
            query = list(
              stationTriplets = paste(triplet_batch, collapse = ","),
              elements  = "TOBS,RHUM,DPTP",
              beginDate = begin_date,
              endDate   = end_date,
              duration  = "HOURLY",
              periodRef = "START"
            ),
            timeout(300)),
        error = function(e) NULL
      )

      if (is.null(resp)) next
      if (status_code(resp) == 200) {
        return(fromJSON(content(resp, as = "text", encoding = "UTF-8"),
                        simplifyVector = TRUE))
      }
      message("    HTTP ", status_code(resp), " on attempt ", attempt)
    }
    NULL
  }

  ## Turn one station's nested API response into long rows.
  parse_station <- function(station_triplet, data_row) {
    if (is.null(data_row$values) || length(data_row$values) == 0) return(NULL)
    elements <- data_row$stationElement$elementCode

    bind_rows(compact(lapply(seq_along(elements), function(e) {
      vals <- data_row$values[[e]]
      if (is.null(vals) || nrow(vals) == 0) return(NULL)
      data.frame(
        station  = station_triplet,
        datetime = as.POSIXct(vals$date, format = "%Y-%m-%d %H:%M", tz = "UTC"),
        element  = elements[e],
        value    = as.numeric(vals$value),
        stringsAsFactors = FALSE
      )
    })))
  }

  call_n <- 0
  for (yr in years) {
    begin_date <- format(max(as_date(start_utc), as_date(paste0(yr, "-01-01"))), "%Y-%m-%d")
    end_date   <- format(min(as_date(end_utc),   as_date(paste0(yr, "-12-31"))), "%Y-%m-%d")

    for (b in seq_along(station_batches)) {
      call_n <- call_n + 1
      message("  [", call_n, "/", total_calls, "] ", yr, " batch ", b)

      result <- fetch_awdb(station_batches[[b]], begin_date, end_date)
      if (is.null(result) || nrow(result) == 0) next

      all_long[[length(all_long) + 1]] <- bind_rows(lapply(
        seq_len(nrow(result)),
        function(i) parse_station(result$stationTriplet[i], result$data[[i]])
      ))
      Sys.sleep(1)
    }

    write_csv(bind_rows(all_long),
              file.path(out_dir, paste0("wcc_awdb_checkpoint_through_", yr, ".csv")))
    message("  Checkpointed through ", yr)
  }

  long_df <- bind_rows(all_long)
  write_csv(long_df, file.path(out_dir, paste0("wcc_awdb_long_raw_", DATE_SUFFIX, ".csv")))

  if (nrow(long_df) == 0) {
    message("  No WCC data retrieved.")
    return(NULL)
  }

  wide_df <- long_df %>%
    tidyr::pivot_wider(names_from = element, values_from = value) %>%
    rename_with(tolower) %>%
    rename(tair = any_of("tobs"), rh = any_of("rhum"), tdew = any_of("dptp")) %>%
    mutate(
      .id          = sub(":.*", "", station),
      date         = format(datetime, "%Y-%m-%d %H:%M"),
      datetime_lst = datetime,
      tair = if ("tair" %in% names(.)) as.numeric(tair) else NA_real_,
      rh   = if ("rh"   %in% names(.)) as.numeric(rh)   else NA_real_,
      tdew = if ("tdew" %in% names(.)) as.numeric(tdew) else NA_real_
    ) %>%
    select(.id, date, tair, rh, tdew, datetime_lst, datetime)

  write_csv(wide_df, file.path(out_dir, paste0("wcc_awdb_raw_", DATE_SUFFIX, ".csv")))
  message("  ", nrow(wide_df), " rows for ", length(unique(wide_df$.id)), " stations.")

  preprocess_meteo("WCC", wide_df)
}


## ---------------------------------------------------------------------------
## Run both regions
## ---------------------------------------------------------------------------

for (region_id in REGION_IDS) {

  ## Search centre is the centroid of the AOI vertices.
  centre  <- colMeans(REGION_AOI[[region_id]])
  lon_obs <- centre[1]
  lat_obs <- centre[2]

  message("\n", strrep("=", 55))
  message("  ", region_id, " — ", REGION_LABEL[[region_id]])
  message("  Centre: ", round(lon_obs, 3), ", ", round(lat_obs, 3))
  message(strrep("=", 55))

  out_dir <- ensure_dir(file.path(REPO_ROOT, "Data", "Stations", region_id))

  stations_hads <- station_select("HADS", lon_obs, lat_obs, DEG_FILTER, DIST_THRESH_M)
  stations_wcc  <- station_select("WCC",  lon_obs, lat_obs, DEG_FILTER, DIST_THRESH_M)

  ## LCD stations are filtered from the metadata table directly rather than
  ## through station_select.
  stations_lcd <- lcd_meta %>%
    filter(
      LONGITUDE >= lon_obs - DEG_FILTER, LONGITUDE <= lon_obs + DEG_FILTER,
      LATITUDE  >= lat_obs - DEG_FILTER, LATITUDE  <= lat_obs + DEG_FILTER
    ) %>%
    rowwise() %>%
    mutate(dist = geosphere::distHaversine(c(lon_obs, lat_obs),
                                           c(LONGITUDE, LATITUDE))) %>%
    ungroup() %>%
    filter(dist <= DIST_THRESH_M)

  message("  HADS: ", nrow(stations_hads),
          " | LCD: ", nrow(stations_lcd),
          " | WCC: ", nrow(stations_wcc))

  message("\nDownloading HADS...")
  hads_out <- download_batched("HADS", WY_START, WY_END, stations_hads, batch_size = 50)
  write_csv(hads_out$data, file.path(out_dir, paste0("hads_", DATE_SUFFIX, ".csv")))
  if (nrow(hads_out$errors) > 0) {
    write_csv(hads_out$errors, file.path(out_dir, "hads_error_log.csv"))
  }

  ## Smaller LCD batches; the API times out on large requests.
  message("\nDownloading LCD...")
  lcd_out <- download_batched("LCD", WY_START, WY_END, stations_lcd, batch_size = 10)
  write_csv(lcd_out$data, file.path(out_dir, paste0("lcd_", DATE_SUFFIX, ".csv")))
  if (nrow(lcd_out$errors) > 0) {
    write_csv(lcd_out$errors, file.path(out_dir, "lcd_error_log.csv"))
  }

  message("\nDownloading WCC (slow)...")
  wcc_df <- get_wcc_awdb(WY_START, WY_END, stations_wcc, out_dir)
  write_csv(wcc_df, file.path(out_dir, paste0("wcc_", DATE_SUFFIX, ".csv")))

  message("\nFetching station metadata...")
  all_ids       <- unique(c(hads_out$data$id, lcd_out$data$id, wcc_df$id))
  stations_meta <- gather_meta(all_ids)
  write_csv(stations_meta,
            file.path(out_dir, paste0("station_metadata_", DATE_SUFFIX, ".csv")))

  message("  ", nrow(stations_meta), " stations -> ", out_dir)
}

message("\nAll regions complete.")
