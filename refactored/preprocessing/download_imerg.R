## Download GPM IMERG half-hourly Probability of Liquid Precipitation
## for both study areas and save one daily table per region.
##
## Output: Data/IMERG/<REGION>/gpm_imerg_<date>.parquet
##         (48 half-hourly columns plp_01..plp_48 per grid cell)
##
## Days whose parquet already exists are skipped, so the script can be re-run
## to fill gaps. A retry pass at the end re-fetches any day that came back
## empty or partial.
##
## One-time setup:
##   1. Create a NASA Earthdata account at https://urs.earthdata.nasa.gov/
##   2. Approve "NASA GESDISC DATA ARCHIVE" under Profile -> Applications
##   3. Add NASA_DATA_USER and NASA_DATA_PASSWORD to your .Renviron
##      (usethis::edit_r_environ()), then restart R.
##
## Requests go to the OPeNDAP .ascii endpoint via curl rather than through
## climateR/RNetCDF: the NetCDF-C library has its own HTTP stack that ignores
## R's auth settings, whereas curl reads .netrc. The request is subset
## server-side.

suppressPackageStartupMessages({
  library(sf)
  library(glue)
  library(lubridate)
  library(furrr)
  library(purrr)
  library(arrow)
  library(curl)
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
ALL_DAYS  <- seq(as.Date(WY_START), as.Date(WY_END), by = "day")

## IMERG Final (V07) lags real time by about 3.5 months and the V07 record
## stops here; use the Late run for anything more recent.
FINAL_RUN_END  <- as.Date("2025-09-30")
LATENCY_CUTOFF <- Sys.Date() - 120


## ---------------------------------------------------------------------------
## Earthdata authentication
## ---------------------------------------------------------------------------

nasa_user <- Sys.getenv("NASA_DATA_USER")
nasa_pass <- Sys.getenv("NASA_DATA_PASSWORD")

if (nchar(nasa_user) == 0 || nchar(nasa_pass) == 0) {
  stop("Set NASA_DATA_USER and NASA_DATA_PASSWORD in your .Renviron, then ",
       "restart R.")
}

HOME_DIR     <- path.expand("~")
NETRC_PATH   <- file.path(HOME_DIR, ".netrc")
COOKIES_PATH <- file.path(HOME_DIR, ".urs_cookies")

if (!file.exists(NETRC_PATH) || file.size(NETRC_PATH) == 0) {
  writeLines(c("machine urs.earthdata.nasa.gov",
               paste("login", nasa_user),
               paste("password", nasa_pass)),
             con = NETRC_PATH)
}
if (!file.exists(COOKIES_PATH)) file.create(COOKIES_PATH)

make_curl_handle <- function() {
  h <- curl::new_handle()
  curl::handle_setopt(h, netrc = 1L, netrc_file = NETRC_PATH,
                      followlocation = 1L, unrestricted_auth = 1L,
                      cookiefile = COOKIES_PATH, cookiejar = COOKIES_PATH)
  h
}


## ---------------------------------------------------------------------------
## URL construction
## ---------------------------------------------------------------------------

## The 48 half-hourly file URLs for one day.
opendap_urls <- function(date, run = "final", version_str = NULL) {
  if (run == "final") {
    product_id <- "3B-HHR.MS.MRG.3IMERG"
    product    <- "GPM_3IMERGHH.07"
  } else if (run == "late") {
    product_id <- "3B-HHR-L.MS.MRG.3IMERG"
    product    <- "GPM_3IMERGHHL.07"
  } else {
    stop("run must be 'final' or 'late'")
  }

  date <- as.Date(date)
  if (is.null(version_str)) {
    version_str <- if (run == "late" && date >= as.Date("2026-03-01")) "V07C" else "V07B"
  }

  base_url <- glue(
    "https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/{product}/",
    "{format(date, '%Y')}/{format(date, '%j')}/"
  )
  origin_time <- as.POSIXct(paste0(date, " 00:00:00"), tz = "UTC")
  times <- seq(ymd_hms(glue("{date} 00:00:00"), tz = "UTC"),
               by = "30 mins", length.out = 48)

  sapply(times, function(t) {
    filename <- glue(
      "{product_id}.{format(t, '%Y%m%d')}",
      "-S{format(t, '%H%M%S')}",
      "-E{format(t + minutes(29) + seconds(59), '%H%M%S')}",
      ".{sprintf('%04d', as.integer(difftime(t, origin_time, units = 'mins')))}",
      ".{version_str}.HDF5"
    )
    glue("{base_url}{filename}")
  })
}

## Late-run version strings changed mid-record; probe with a tiny request to
## find which one is live for a given day.
resolve_version <- function(date, run, candidates = c("V07C", "V07B")) {
  for (v in candidates) {
    probe_url <- paste0(opendap_urls(date, run, version_str = v)[1],
                        ".ascii?lat[0:1:0]")
    resp <- tryCatch(curl::curl_fetch_memory(probe_url, handle = make_curl_handle()),
                     error = function(e) list(status_code = 0))
    if (resp$status_code == 200) return(v)
  }
  warning("Could not resolve IMERG version for ", date, " run='", run, "'")
  candidates[1]
}


## ---------------------------------------------------------------------------
## Fetching
## ---------------------------------------------------------------------------

## Fetch one half-hourly timestep, subset to the AOI bounding box server-side.
## The .ascii response has one line per longitude slice, then the lat and lon
## coordinate vectors.
get_gpm_ascii <- function(url, aoi) {
  tryCatch({
    bbox        <- sf::st_bbox(aoi)
    lat_min_idx <- max(0,    floor((bbox$ymin + 90)  / 0.1))
    lat_max_idx <- min(1799, floor((bbox$ymax + 90)  / 0.1))
    lon_min_idx <- max(0,    floor((bbox$xmin + 180) / 0.1))
    lon_max_idx <- min(3599, floor((bbox$xmax + 180) / 0.1))

    ascii_url <- paste0(
      url, ".ascii",
      "?probabilityLiquidPrecipitation[0:1:0][",
      lon_min_idx, ":1:", lon_max_idx, "][", lat_min_idx, ":1:", lat_max_idx, "]",
      ",lon[", lon_min_idx, ":1:", lon_max_idx, "]",
      ",lat[", lat_min_idx, ":1:", lat_max_idx, "]"
    )

    resp <- curl::curl_fetch_memory(ascii_url, handle = make_curl_handle())
    if (resp$status_code != 200) stop("HTTP ", resp$status_code)

    lines <- trimws(strsplit(rawToChar(resp$content), "\n")[[1]])

    lat_vals <- as.numeric(strsplit(
      sub("^lat,\\s*", "", lines[grep("^lat,", lines)]), ",\\s*")[[1]])
    lon_vals <- as.numeric(strsplit(
      sub("^lon,\\s*", "", lines[grep("^lon,", lines)]), ",\\s*")[[1]])

    plp_lines <- grep("^probabilityLiquidPrecipitation\\[0\\]\\[", lines, value = TRUE)
    plp_mat   <- do.call(rbind, lapply(plp_lines, function(l) {
      as.numeric(strsplit(sub("^[^,]+,\\s*", "", l), ",\\s*")[[1]])
    }))

    ## plp_mat is (n_lon x n_lat); expand.grid varies lon fastest to match.
    expand.grid(x = lon_vals, y = lat_vals) |>
      transform(probabilityLiquidPrecipitation = as.vector(plp_mat))

  }, error = function(e) {
    message("  [WARN] ", basename(url), ": ", conditionMessage(e))
    NA
  })
}

## Fetch all 48 timesteps for a day in small parallel batches and merge them
## into one wide table keyed on grid cell.
get_gpm_day <- function(day, aoi, run, batch_size = 5, pause_seconds = 5,
                        workers = 3) {

  version_str <- if (run == "late" && as.Date(day) >= as.Date("2026-03-01")) {
    resolve_version(day, run)
  } else {
    "V07B"
  }

  urls        <- opendap_urls(day, run, version_str)
  num_batches <- ceiling(length(urls) / batch_size)
  all_dfs     <- vector("list", length(urls))

  plan(multisession, workers = workers)
  on.exit(plan(sequential), add = TRUE)

  for (batch_num in seq_len(num_batches)) {
    idx_start <- (batch_num - 1) * batch_size + 1
    idx_end   <- min(batch_num * batch_size, length(urls))

    all_dfs[idx_start:idx_end] <- future_map(
      urls[idx_start:idx_end],
      ~ get_gpm_ascii(.x, aoi),
      .options = furrr_options(
        seed = TRUE, packages = c("sf", "curl"),
        globals = list(get_gpm_ascii = get_gpm_ascii,
                       make_curl_handle = make_curl_handle,
                       NETRC_PATH = NETRC_PATH,
                       COOKIES_PATH = COOKIES_PATH,
                       aoi = aoi)
      )
    )

    if (batch_num < num_batches) Sys.sleep(pause_seconds)
  }

  valid <- all_dfs[!sapply(all_dfs, function(x) length(x) == 1 && is.na(x))]
  if (length(valid) == 0) {
    message("  [WARN] no data for ", day)
    return(NA)
  }

  ## Name each timestep's column before merging so they don't collide.
  valid <- lapply(seq_along(valid), function(i) {
    df <- valid[[i]]
    names(df)[names(df) == "probabilityLiquidPrecipitation"] <-
      paste0("plp_", sprintf("%02d", i))
    df
  })

  message(sprintf("  %s: %d/48 timesteps", day, length(valid)))
  purrr::reduce(valid, ~ merge(.x, .y, by = c("x", "y"), all = TRUE))
}


## Which run to use for a given day.
run_for_day <- function(day) {
  if (as.Date(day) <= LATENCY_CUTOFF && as.Date(day) <= FINAL_RUN_END) "final" else "late"
}

## A day's file is incomplete if it is missing, unreadable, or has < 48 slots.
is_incomplete <- function(f) {
  tryCatch({
    df <- arrow::read_parquet(f)
    if (is.logical(df) || all(is.na(df)) || nrow(df) == 0) return(TRUE)
    length(grep("^plp_\\d{2}$", names(df))) < 48
  }, error = function(e) TRUE)
}


## ---------------------------------------------------------------------------
## Main loop, then a retry pass over anything that failed
## ---------------------------------------------------------------------------

out_dirs <- setNames(
  lapply(REGION_IDS, function(r) ensure_dir(file.path(REPO_ROOT, "Data", "IMERG", r))),
  REGION_IDS
)

for (region_id in REGION_IDS) {
  aoi     <- aoi_polygon(region_id, round_dp = 3, hull = TRUE)
  out_dir <- out_dirs[[region_id]]

  message("\n", strrep("=", 60))
  message("REGION: ", region_id, " — ", REGION_LABEL[[region_id]])
  message(strrep("=", 60))

  for (i in seq_along(ALL_DAYS)) {
    day      <- ALL_DAYS[i]
    out_file <- file.path(out_dir, glue("gpm_imerg_{day}.parquet"))

    if (file.exists(out_file)) next

    run <- run_for_day(day)
    message(sprintf("[%s] %s (%d/%d) run='%s'",
                    region_id, day, i, length(ALL_DAYS), run))

    ## Periodic garbage collection and worker restart.
    if (i %% 20 == 0) {
      gc()
      plan(sequential)
    }

    day_df <- tryCatch(get_gpm_day(day, aoi, run),
                       error = function(e) {
                         message("[ERROR] ", conditionMessage(e))
                         NA
                       })
    arrow::write_parquet(day_df, out_file)
  }
}

message("\n", strrep("=", 60))
message("RETRY PASS")
message(strrep("=", 60))

for (region_id in REGION_IDS) {
  aoi     <- aoi_polygon(region_id, round_dp = 3, hull = TRUE)
  out_dir <- out_dirs[[region_id]]

  failed_files <- Filter(is_incomplete,
                         list.files(out_dir, pattern = "\\.parquet$", full.names = TRUE))

  if (length(failed_files) == 0) {
    message("[", region_id, "] nothing to retry.")
    next
  }

  failed_dates <- as.Date(sub("gpm_imerg_(.+)\\.parquet", "\\1", basename(failed_files)))
  message(sprintf("[%s] retrying %d days", region_id, length(failed_dates)))

  for (j in seq_along(failed_dates)) {
    day <- failed_dates[j]
    Sys.sleep(10)

    day_df <- tryCatch(get_gpm_day(day, aoi, run_for_day(day), pause_seconds = 10),
                       error = function(e) {
                         message("[ERROR] retry: ", conditionMessage(e))
                         NA
                       })
    arrow::write_parquet(day_df, file.path(out_dir, glue("gpm_imerg_{day}.parquet")))
  }
}

message("\n--- Final status ---")
for (region_id in REGION_IDS) {
  all_parquets <- list.files(out_dirs[[region_id]], pattern = "\\.parquet$",
                             full.names = TRUE)
  still_failed <- Filter(is_incomplete, all_parquets)
  message(sprintf("[%s] %d / %d still incomplete",
                  region_id, length(still_failed), length(all_parquets)))
  if (length(still_failed) > 0) {
    cat(paste0("    ", sub("gpm_imerg_(.+)\\.parquet", "\\1", basename(still_failed))),
        sep = "\n")
  }
}
