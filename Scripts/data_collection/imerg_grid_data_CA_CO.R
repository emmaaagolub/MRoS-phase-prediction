library(sf)
library(glue)
library(lubridate)
library(furrr)
library(purrr)
library(arrow)
library(curl)

# =============================================================================
# PURPOSE:
#   Downloads half-hourly GPM IMERG Probability of Liquid Precipitation (PLP)
#   data from NASA's OPeNDAP servers for two regions (CA Sierra Nevada and CO
#   Mountains), clips each to its respective AOI, merges all 48 half-hourly
#   timesteps into a daily gridded table, and saves the result as a daily
#   parquet file.
#
#   Regions:
#     CA – Sierra Nevada / Lake Tahoe  (EPSG:26911)
#     CO – Colorado Mountains          (EPSG:32613)
#
#   Temporal window: 2022-10-01 to 2025-05-01
#     NOTE: The loop skips days whose output parquet already exists.
#
#   Output: Data/IMERG/{CA|CO}/gpm_imerg_{date}.parquet
#           (one parquet per day per region)
#
# AUTHENTICATION:
#   climateR/RNetCDF use the NetCDF-C C library which has its own HTTP stack
#   and cannot use R's curl/httr config. After much debugging, we bypass all
#   of that and use R's curl package directly against the .ascii OPeNDAP
#   endpoint, which handles auth via .netrc and returns plain text we parse
#   ourselves. This is simpler, more reliable, and server-side subsetted.
#
#   One-time setup:
#   1. Create a NASA Earthdata account: https://urs.earthdata.nasa.gov/
#   2. Approve "NASA GESDISC DATA ARCHIVE":
#      Profile -> Applications -> Authorized Apps
#   3. Add to your .Renviron (run usethis::edit_r_environ()):
#        NASA_DATA_USER=your_username
#        NASA_DATA_PASSWORD=your_password
#      Then restart R.
# =============================================================================


# -----------------------------------------------------------------------------
# 0.  Authentication — write .netrc for curl
# -----------------------------------------------------------------------------

nasa_user <- Sys.getenv("NASA_DATA_USER")
nasa_pass <- Sys.getenv("NASA_DATA_PASSWORD")

if (nchar(nasa_user) == 0 || nchar(nasa_pass) == 0) {
  stop(
    "NASA Earthdata credentials not found.\n",
    "Add NASA_DATA_USER and NASA_DATA_PASSWORD to your .Renviron:\n",
    "  usethis::edit_r_environ()\n",
    "Then restart R."
  )
}

home_dir     <- "C:/Users/EmmaGolub"
netrc_path   <- file.path(home_dir, ".netrc")
cookies_path <- file.path(home_dir, ".urs_cookies")

if (!file.exists(netrc_path) || file.size(netrc_path) == 0) {
  message("Writing .netrc to ", netrc_path)
  writeLines(
    c(
      "machine urs.earthdata.nasa.gov",
      paste("login",    nasa_user),
      paste("password", nasa_pass)
    ),
    con = netrc_path
  )
}

if (!file.exists(cookies_path)) file.create(cookies_path)

message("Auth setup:")
message("  .netrc  : ", netrc_path,   "  exists=", file.exists(netrc_path))
message("  cookies : ", cookies_path, "  exists=", file.exists(cookies_path))


# -----------------------------------------------------------------------------
# 1.  Region configuration
# -----------------------------------------------------------------------------

make_aoi <- function(lonlat_list) {
  pts  <- do.call(rbind, lonlat_list)
  hull <- pts[chull(pts), ]
  hull <- rbind(hull, hull[1, ])
  poly <- st_polygon(list(hull))
  st_sfc(poly, crs = 4326) |> st_sf()
}

regions <- list(
  
  CA = list(
    label = "California: Sierra Nevada / Lake Tahoe",
    aoi   = make_aoi(list(
      c(-119.455, 39.653),
      c(-121.279, 39.662),
      c(-119.114, 36.727),
      c(-118.499, 37.236),
      c(-119.466, 38.373)
    ))
  ),
  
  CO = list(
    label = "Colorado Mountains",
    aoi   = make_aoi(list(
      c(-105.199, 40.620),
      c(-106.889, 40.555),
      c(-107.661, 38.795),
      c(-104.879, 38.774)
    ))
  )
  
)


# -----------------------------------------------------------------------------
# 2.  Paths
# -----------------------------------------------------------------------------

BASE_DIR <- file.path(
  "C:/Users/EmmaGolub/Desktop/MRoS_local",
  "mros-precipitation-phase-product-prototype"
)

out_dirs <- list(
  CA = file.path(BASE_DIR, "Data", "IMERG", "CA"),
  CO = file.path(BASE_DIR, "Data", "IMERG", "CO")
)

for (d in out_dirs) dir.create(d, showWarnings = FALSE, recursive = TRUE)


# -----------------------------------------------------------------------------
# 3.  Date window
# -----------------------------------------------------------------------------

all_days <- seq(as.Date("2022-10-01"), as.Date("2025-05-01"), by = "day")

# IMERG Final product (V07) has ~3.5-month latency:
#   days older than 120 days  -> "final"  (higher quality)
#   more recent days          -> "late"   (near-real-time)
latency_cutoff <- Sys.Date() - 120


# -----------------------------------------------------------------------------
# 4.  OPeNDAP URL builder
# -----------------------------------------------------------------------------

opendap_urls <- function(date, run = "final") {
  
  if (run == "final") {
    product_id <- "3B-HHR.MS.MRG.3IMERG"
    product    <- "GPM_3IMERGHH.07"
  } else if (run == "late") {
    product_id <- "3B-HHR-L.MS.MRG.3IMERG"
    product    <- "GPM_3IMERGHHL.07"
  } else {
    stop("Supply either 'final' or 'late' for run=")
  }
  
  date        <- as.Date(date)
  year        <- format(date, "%Y")
  julian      <- format(date, "%j")
  origin_time <- as.POSIXct(paste0(date, " 00:00:00"), tz = "UTC")
  base_url    <- glue(
    "https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/{product}/{year}/{julian}/"
  )
  
  times <- seq(ymd_hms(glue("{date} 00:00:00"), tz = "UTC"),
               by = "30 mins", length.out = 48)
  
  sapply(times, function(t) {
    start_str    <- format(t, "%H%M%S")
    end_str      <- format(t + minutes(29) + seconds(59), "%H%M%S")
    file_time    <- format(t, "%Y%m%d")
    minutes_diff <- sprintf("%04d",
                            as.integer(difftime(t, origin_time, units = "mins")))
    filename <- glue(
      "{product_id}.{file_time}-S{start_str}-E{end_str}.{minutes_diff}.V07B.HDF5"
    )
    glue("{base_url}{filename}")
  })
}


# -----------------------------------------------------------------------------
# 5.  Single-URL fetch via .ascii endpoint
#     Uses curl directly — bypasses RNetCDF/NetCDF-C auth issues entirely.
#     Server-side spatial subset keeps response small (~10-30KB per timestep).
#
#     ASCII response format:
#       Dataset: <filename>
#       probabilityLiquidPrecipitation[0][0], v1, v2, ...  <- one lon per line
#       probabilityLiquidPrecipitation[0][1], v1, v2, ...
#       ...
#       lat, lat1, lat2, ...
#       lon, lon1, lon2, ...
# -----------------------------------------------------------------------------

make_curl_handle <- function() {
  h <- curl::new_handle()
  curl::handle_setopt(h, netrc             = 1L)
  curl::handle_setopt(h, netrc_file        = "C:/Users/EmmaGolub/.netrc")
  curl::handle_setopt(h, followlocation    = 1L)
  curl::handle_setopt(h, unrestricted_auth = 1L)
  curl::handle_setopt(h, cookiefile        = "C:/Users/EmmaGolub/.urs_cookies")
  curl::handle_setopt(h, cookiejar         = "C:/Users/EmmaGolub/.urs_cookies")
  h
}

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
      lon_min_idx, ":1:", lon_max_idx, "][",
      lat_min_idx, ":1:", lat_max_idx, "]",
      ",lon[", lon_min_idx, ":1:", lon_max_idx, "]",
      ",lat[", lat_min_idx, ":1:", lat_max_idx, "]"
    )
    
    resp <- curl::curl_fetch_memory(ascii_url, handle = make_curl_handle())
    
    if (resp$status_code != 200) stop("HTTP ", resp$status_code)
    
    lines <- trimws(strsplit(rawToChar(resp$content), "\n")[[1]])
    
    # Parse lat and lon vectors
    lat_vals <- as.numeric(
      strsplit(sub("^lat,\\s*", "", lines[grep("^lat,", lines)]), ",\\s*")[[1]]
    )
    lon_vals <- as.numeric(
      strsplit(sub("^lon,\\s*", "", lines[grep("^lon,", lines)]), ",\\s*")[[1]]
    )
    
    # Parse PLP — each line is one lon slice across all lats
    plp_lines <- grep("^probabilityLiquidPrecipitation\\[0\\]\\[", lines, value = TRUE)
    plp_mat   <- do.call(rbind, lapply(plp_lines, function(l) {
      as.numeric(strsplit(sub("^[^,]+,\\s*", "", l), ",\\s*")[[1]])
    }))
    
    # plp_mat is (n_lon x n_lat); expand.grid gives x=lon, y=lat
    expand.grid(x = lon_vals, y = lat_vals) |>
      transform(probabilityLiquidPrecipitation = as.vector(plp_mat))
    
  }, error = function(e) {
    message("  [WARN] Failed: ", basename(url), "\n         ", conditionMessage(e))
    return(NA)
  })
}


# -----------------------------------------------------------------------------
# 6.  Full-day fetch with batching + parallel workers
# -----------------------------------------------------------------------------

get_gpm_day <- function(day, aoi, run,
                        batch_size    = 5,
                        pause_seconds = 5,
                        workers       = 3) {
  
  urls        <- opendap_urls(date = day, run = run)
  total_urls  <- length(urls)       # always 48
  num_batches <- ceiling(total_urls / batch_size)
  all_dfs     <- vector("list", total_urls)
  
  plan(multisession, workers = workers)
  on.exit(plan(sequential), add = TRUE)
  
  t_start <- Sys.time()
  
  for (batch_num in seq_len(num_batches)) {
    
    idx_start  <- (batch_num - 1) * batch_size + 1
    idx_end    <- min(batch_num * batch_size, total_urls)
    batch_urls <- urls[idx_start:idx_end]
    
    message(sprintf("  Batch %d/%d  [URLs %d-%d]",
                    batch_num, num_batches, idx_start, idx_end))
    
    batch_results <- future_map(
      batch_urls,
      ~ {
        library(sf)
        library(curl)
        get_gpm_ascii(.x, aoi)
      },
      .options = furrr_options(
        seed     = TRUE,
        packages = c("sf", "curl"),
        globals  = list(
          get_gpm_ascii    = get_gpm_ascii,
          make_curl_handle = make_curl_handle,
          aoi              = aoi
        )
      )
    )
    
    all_dfs[idx_start:idx_end] <- batch_results
    
    if (batch_num < num_batches) Sys.sleep(pause_seconds)
  }
  
  valid <- all_dfs[!sapply(all_dfs, function(x) length(x) == 1 && is.na(x))]
  
  if (length(valid) == 0) {
    message("  [WARN] No data retrieved for ", day)
    return(NA)
  }
  
  # Each df has columns (x, y, probabilityLiquidPrecipitation)
  # Rename PLP column to timestep index before merging so columns don't collide
  valid <- lapply(seq_along(valid), function(i) {
    df <- valid[[i]]
    names(df)[names(df) == "probabilityLiquidPrecipitation"] <- paste0("plp_", sprintf("%02d", i))
    df
  })
  
  final_df <- purrr::reduce(valid, ~ merge(.x, .y, by = c("x", "y"), all = TRUE))
  
  elapsed <- round(difftime(Sys.time(), t_start, units = "secs"), 1)
  message(sprintf("  Done: %s  (%s s,  %d/48 timesteps)", day, elapsed, length(valid)))
  
  return(final_df)
}


# -----------------------------------------------------------------------------
# 7.  Main loop — regions then days
# -----------------------------------------------------------------------------

for (region_id in names(regions)) {
  
  reg     <- regions[[region_id]]
  out_dir <- out_dirs[[region_id]]
  
  message("\n", strrep("=", 60))
  message("REGION: ", region_id, " — ", reg$label)
  message(strrep("=", 60))
  
  for (i in seq_along(all_days)) {
    
    day      <- all_days[i]
    out_file <- file.path(out_dir, glue("gpm_imerg_{day}.parquet"))
    
    if (file.exists(out_file)) {
      message(sprintf("[%s] %s (%d/%d) — already exists, skipping",
                      region_id, day, i, length(all_days)))
      next
    }
    
    run <- if (as.Date(day) <= latency_cutoff) "final" else "late"
    
    message(sprintf("\n[%s] Processing %s (%d/%d)  run='%s'",
                    region_id, day, i, length(all_days), run))
    
    # Periodic GC to avoid memory creep over long runs
    if (i %% 20 == 0) {
      gc()
      plan(sequential)
    }
    
    day_df <- tryCatch(
      get_gpm_day(day = day, aoi = reg$aoi, run = run),
      error = function(e) {
        message("[ERROR] get_gpm_day failed: ", conditionMessage(e))
        NA
      }
    )
    
    arrow::write_parquet(day_df, out_file)
    message(sprintf("[%s] Saved -> %s", region_id, basename(out_file)))
    
  }
}

message("\nAll done!")


# =============================================================================
# QUICK TEST — run this before the main loop to verify auth + parsing
# =============================================================================
# test_url <- opendap_urls(as.Date("2023-01-15"), run = "final")[5]
# result   <- get_gpm_ascii(test_url, regions$CA$aoi)
# head(result)
# nrow(result)  # expect 870 (29 lon x 30 lat)



# -----------------------------------------------------------------------------
# 8.  Retry pass — re-fetch any days that previously failed (wrote NA)
# -----------------------------------------------------------------------------

message("\n", strrep("=", 60))
message("RETRY PASS — scanning for failed parquets")
message(strrep("=", 60))

for (region_id in names(regions)) {
  
  reg     <- regions[[region_id]]
  out_dir <- out_dirs[[region_id]]
  
  # Find parquets that contain NA (failed fetches)
  all_parquets <- list.files(out_dir, pattern = "\\.parquet$", full.names = TRUE)
  
  failed_files <- Filter(function(f) {
    tryCatch({
      df <- arrow::read_parquet(f)
      # A failed day was written as NA, which arrow stores as a 1-row logical NA
      is.logical(df) || all(is.na(df)) || nrow(df) == 0
    }, error = function(e) TRUE)   # unreadable = also failed
  }, all_parquets)
  
  if (length(failed_files) == 0) {
    message("[", region_id, "] No failed files found — all good!")
    next
  }
  
  # Extract dates from filenames: gpm_imerg_YYYY-MM-DD.parquet
  failed_dates <- as.Date(
    sub("gpm_imerg_(.+)\\.parquet", "\\1", basename(failed_files))
  )
  
  message(sprintf("[%s] Retrying %d failed days...", region_id, length(failed_dates)))
  
  for (j in seq_along(failed_dates)) {
    
    day      <- failed_dates[j]
    out_file <- file.path(out_dir, glue("gpm_imerg_{day}.parquet"))
    run      <- if (as.Date(day) <= latency_cutoff) "final" else "late"
    
    message(sprintf("\n[%s] RETRY %s (%d/%d)  run='%s'",
                    region_id, day, j, length(failed_dates), run))
    
    # Wait a bit before retrying — transient server issues often clear
    Sys.sleep(10)
    
    day_df <- tryCatch(
      get_gpm_day(day = day, aoi = reg$aoi, run = run,
                  pause_seconds = 10),   # slower on retry
      error = function(e) {
        message("[ERROR] Retry failed: ", conditionMessage(e))
        NA
      }
    )
    
    arrow::write_parquet(day_df, out_file)
    status <- if (is.data.frame(day_df)) "OK" else "STILL FAILED"
    message(sprintf("[%s] %s -> %s", region_id, basename(out_file), status))
  }
}

# Summary of anything still failing after retry
message("\n--- Final status ---")
for (region_id in names(regions)) {
  out_dir      <- out_dirs[[region_id]]
  all_parquets <- list.files(out_dir, pattern = "\\.parquet$", full.names = TRUE)
  still_failed <- Filter(function(f) {
    tryCatch({
      df <- arrow::read_parquet(f)
      is.logical(df) || all(is.na(df)) || nrow(df) == 0
    }, error = function(e) TRUE)
  }, all_parquets)
  message(sprintf("[%s] %d / %d files still failed",
                  region_id, length(still_failed), length(all_parquets)))
  if (length(still_failed) > 0) {
    message("  Still failed dates:")
    cat(paste0("    ",
               sub("gpm_imerg_(.+)\\.parquet", "\\1", basename(still_failed))),
        sep = "\n")
  }
}