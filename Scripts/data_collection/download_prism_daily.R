library(sf)
library(tidyverse)
library(lubridate)
library(httr)
library(terra)

# PURPOSE:
# This script downloads daily PRISM climate rasters, clips them to region AOIs,
# stacks multiple meteorological variables into daily multi-band GeoTIFFs, and
# converts the full time series into per-region parquet tables of gridded predictors.
#
# Downloads five daily PRISM variables:
#   - ppt     – precipitation
#   - tmin    – minimum temperature
#   - tmean   – mean temperature
#   - tmax    – maximum temperature
#   - tdmean  – mean dewpoint temperature
#
# Outputs (one set per region):
#   Data/PRISM/<REGION>/prism_daily_stacked/PRISM_<DATE>_stacked.tif
#   Data/PRISM/<REGION>/combined_prism_20241001_20250930.parquet

# =============================================================================
# 1.  AOI helper + region definitions
# =============================================================================

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

# =============================================================================
# 2.  Paths
# =============================================================================

BASE_DIR <- file.path(
  "C:/Users/EmmaGolub/Desktop/MRoS_local",
  "mros-precipitation-phase-product-prototype"
)

out_dirs <- setNames(
  lapply(names(regions), function(r) file.path(BASE_DIR, "Data", "PRISM", r)),
  names(regions)
)
for (d in out_dirs) dir.create(d, showWarnings = FALSE, recursive = TRUE)

# =============================================================================
# 3.  Parameters
# =============================================================================

dates        <- seq(ymd("2024-10-01"), ymd("2025-09-30"), by = "day")
# dates        <- seq(ymd("2022-10-01"), ymd("2024-09-30"), by = "day")
# dates        <- seq(ymd("2025-10-01"), ymd("2026-05-01"), by = "day")

date_codes   <- format(dates, "%Y%m%d")
variables    <- c("ppt", "tmin", "tmean", "tmax", "tdmean")
prism_region <- "us"
prism_res    <- "800m"

# =============================================================================
# 4.  Download function
# =============================================================================

download_prism_var <- function(variable, date_code, aoi,
                               prism_region = "us", prism_res = "800m") {
  base_url <- "https://services.nacse.org/prism/data/get"
  url <- sprintf("%s/%s/%s/%s/%s?format=bil",
                 base_url, prism_region, prism_res, variable, date_code)
  
  tempdir_day <- file.path(tempdir(), paste0(variable, "_", date_code))
  dir.create(tempdir_day, showWarnings = FALSE)
  
  zip_path <- file.path(tempdir_day, paste0(variable, "_", date_code, ".zip"))
  
  result <- try(
    GET(url, write_disk(zip_path, overwrite = TRUE), timeout(60)),
    silent = TRUE
  )
  
  if (inherits(result, "try-error") || result$status_code != 200) {
    warning(paste("Failed to download:", variable, date_code))
    unlink(tempdir_day, recursive = TRUE)
    return(NULL)
  }
  
  unzip(zip_path, exdir = tempdir_day)
  bil_file <- list.files(tempdir_day, pattern = "\\.bil$", full.names = TRUE)
  
  if (length(bil_file) == 0) {
    unlink(tempdir_day, recursive = TRUE)
    return(NULL)
  }
  
  r <- rast(bil_file)
  r <- crop(r, aoi) |> mask(aoi)
  
  unlink(tempdir_day, recursive = TRUE)
  return(r)
}

# # =============================================================================
# # TEST CHUNK — run this before the full loop
# # Downloads 1 variable x 1 day x 1 region to verify paths, AOI, and
# # PRISM connectivity are all working before committing to the full run.
# # =============================================================================
# 
# test_region   <- "CA"           # which region to test
# test_date     <- "20241001"     # single date (first day of WY)
# test_variable <- "ppt"          # single variable
# 
# cat("--- PRISM download test ---\n")
# cat("Region  :", test_region, "-", regions[[test_region]]$label, "\n")
# cat("Date    :", test_date, "\n")
# cat("Variable:", test_variable, "\n\n")
# 
# # --- 1. Check output directories were created --------------------------------
# test_stacked_dir <- file.path(out_dirs[[test_region]], "prism_daily_stacked")
# dir.create(test_stacked_dir, showWarnings = FALSE, recursive = TRUE)
# 
# cat("Output dir:", test_stacked_dir, "\n")
# cat("Exists    :", dir.exists(test_stacked_dir), "\n\n")
# 
# # --- 2. Check AOI looks sane -------------------------------------------------
# test_aoi <- regions[[test_region]]$aoi
# cat("AOI CRS   :", st_crs(test_aoi)$epsg, "\n")
# cat("AOI bbox  :\n"); print(st_bbox(test_aoi)); cat("\n")
# 
# # --- 3. Try downloading one raster -------------------------------------------
# cat("Attempting download...\n")
# test_rast <- download_prism_var(test_variable, test_date, test_aoi,
#                                 prism_region, prism_res)
# 
# if (is.null(test_rast)) {
#   cat("FAILED: download returned NULL. Check internet access and PRISM URL.\n")
# } else {
#   cat("SUCCESS: raster downloaded and clipped.\n")
#   cat("  Dimensions :", nrow(test_rast), "rows x", ncol(test_rast), "cols\n")
#   cat("  Resolution :", paste(res(test_rast), collapse = " x "), "\n")
#   cat("  CRS        :", crs(test_rast, describe = TRUE)$code, "\n")
#   cat("  Value range:", round(minmax(test_rast)[1], 3), "to",
#       round(minmax(test_rast)[2], 3), "\n\n")
#   
#   # --- 4. Write a test TIF and confirm it lands in the right place -----------
#   test_tif <- file.path(test_stacked_dir,
#                         paste0("TEST_PRISM_", test_date, "_", test_variable, ".tif"))
#   writeRaster(test_rast, test_tif, overwrite = TRUE)
#   cat("Test TIF written :", file.exists(test_tif), "\n")
#   cat("Path             :", test_tif, "\n\n")
#   
#   # --- 5. Quick plot ----------------------------------------------------------
#   plot(test_rast, main = paste("TEST:", test_region, test_variable, test_date))
#   
#   # --- 6. Clean up test file (comment out to keep it) ------------------------
#   file.remove(test_tif)
#   cat("Test TIF removed.\n")
# }
# 
# cat("\n--- Test complete. If all checks passed, run the full loop. ---\n")


# =============================================================================
# 5.  Per-region download loop
# =============================================================================

for (region_id in names(regions)) {
  region_info <- regions[[region_id]]
  aoi         <- region_info$aoi
  stacked_dir <- file.path(out_dirs[[region_id]], "prism_daily_stacked")
  dir.create(stacked_dir, showWarnings = FALSE, recursive = TRUE)
  
  cat("\n=================================================\n")
  cat("Region:", region_id, "-", region_info$label, "\n")
  cat("=================================================\n")
  
  for (d in date_codes) {
    stacked_path <- file.path(stacked_dir, paste0("PRISM_", d, "_stacked.tif"))
    
    if (file.exists(stacked_path)) {
      cat("Already processed:", region_id, d, "\n")
      next
    }
    
    cat("Processing:", region_id, d, "\n")
    daily_layers <- list()
    
    for (v in variables) {
      cat("  - downloading", v, "\n")
      r <- download_prism_var(v, d, aoi, prism_region, prism_res)
      if (!is.null(r)) {
        names(r) <- v
        daily_layers[[v]] <- r
      }
      Sys.sleep(0.3)  # be polite to the server
    }
    
    if (length(daily_layers) > 0) {
      r_stack <- rast(daily_layers)
      writeRaster(r_stack, stacked_path, overwrite = TRUE)
      cat("  Saved:", basename(stacked_path), "\n")
    } else {
      cat("  No layers downloaded for", region_id, d, "- skipping.\n")
    }
  }
}

# =============================================================================
# 6.  Per-region: combine stacked TIFFs -> parquet
# =============================================================================

for (region_id in names(regions)) {
  stacked_dir <- file.path(out_dirs[[region_id]], "prism_daily_stacked")
  parquet_out <- file.path(out_dirs[[region_id]],
                           "combined_prism_20221001_20260501.parquet")
  
  files <- list.files(stacked_dir, pattern = "_stacked\\.tif$", full.names = TRUE)
  
  if (length(files) == 0) {
    warning("No stacked .tif files found for region ", region_id, " in: ", stacked_dir)
    next
  }
  
  tif_dates <- gsub(".*PRISM_(\\d{8})_stacked\\.tif$", "\\1", basename(files))
  
  cat("\nBuilding parquet for", region_id, "-", length(files), "days...\n")
  
  daily_list <- lapply(seq_along(files), function(i) {
    r  <- rast(files[i])
    df <- as.data.frame(r, xy = TRUE, na.rm = FALSE)
    names(df) <- c("lon", "lat", names(r))
    df$date   <- tif_dates[i]
    df
  })
  
  combined_df <- bind_rows(daily_list)
  
  cat("  Dimensions:", nrow(combined_df), "rows x", ncol(combined_df), "cols\n")
  
  arrow::write_parquet(combined_df, parquet_out)
  cat("  Parquet written to:", parquet_out, "\n")
}

cat("\nDone.\n")