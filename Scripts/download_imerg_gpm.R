#!/usr/bin/env Rscript
# -------------------------------------------------------------
#  citsci_imerg_download.R
#  Download GPM IMERG v07 half-hour PLP for a custom date range
# -------------------------------------------------------------
## ---------------- 1.  SET-UP ---------------------------------


## Install/load packages & the local repo  ----------------
pkgs <- c("devtools", "tidyverse", "lubridate", "readr", "sf", "climateR", "terra",
                "furrr", "progressr", "glue")
for (p in pkgs) if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
lapply(pkgs, library, character.only = TRUE)


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

## ---------------- 2.  USER SETTINGS ---------------------------------

# date window
start_date <- as.POSIXct("2024-10-01 00:00:00", tz = "UTC")
end_date   <- as.POSIXct("2025-05-31 23:30:00", tz = "UTC")

# Center point as sf
pt <- st_point(c(lon_obs, lat_obs)) %>%
  st_sfc(crs = 4326) %>%
  st_transform(5070)  # project to meters (NAD83 / Conus Albers)

# Buffer and convert back to lat/lon bounding box
aoi_bbox <- st_buffer(pt, dist = dist_thresh_m) %>%
  st_transform(4326) %>%
  st_bbox()

aoi <- list(
  xmin = aoi_bbox["xmin"],
  xmax = aoi_bbox["xmax"],
  ymin = aoi_bbox["ymin"],
  ymax = aoi_bbox["ymax"]
)

# where to save the GeoTIFFs
out_dir <- normalizePath(file.path(script_dir, "..", "Data", "gpm_imerg"), mustWork = TRUE)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

varname <- "probabilityLiquidPrecipitation"   # IMERG PLP

## ---------------- 3.  RUNTIME SET-UP ---------------------------------

# Make sure ~/.netrc exists and is 0600:
#   machine urs.earthdata.nasa.gov
#   login   <your_Earthdata_user>
#   password <your_password>
Sys.setenv("CURL_CA_BUNDLE" = "")
Sys.getenv("HOME")     # should point to directory containing .netrc
Sys.setenv("NETRC" = normalizePath("~/.netrc"))
# TEST THIS BEFORE RUNNING <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# BUILD THE HALF-HOURLY TIMESTAMP VECTOR ----
timestamps <- seq(start_date, end_date, by = "30 min")

# HELPER FUNCTION TO DOWNLOAD ONE SLICE ----
download_imerg_slice <- function(ts, aoi, dest_dir, product_version = "GPM_3IMERGHHL.07") {

  base_url   <- construct_gpm_base_url(ts, product_version)
  prod_stub  <- construct_gpm_product(ts,    product_version)

  # pull the catalog for that Julian day and find the exact file
  urls       <- get_final_urls(paste0(base_url, "catalog.xml"))
  slice_url  <- get_closest_url(urls, prod_stub)

  ## read the slice via OPeNDAP + climateR
  s <- climateR::dap(URL     = slice_url,
                     varname = varname,
                     AOI     = aoi,
                     verbose = FALSE)

  ## write to disk
  ts_label  <- format(as.POSIXct(ts, tz = "UTC"), "%Y%m%dT%H%M")
  out_file  <- file.path(dest_dir, glue("IMERG_PLP_{ts_label}.tif"))

  if (!file.exists(out_file)) terra::writeRaster(s, out_file, overwrite = TRUE)
  out_file
}

## ---------------- 4.  PARELLEL DOWNLOAD ---------------------------------
plan(multisession, workers = max(1, parallel::detectCores() - 1))
progressr::handlers(global = TRUE)

with_progress({
  future_walk(timestamps,
              download_imerg_slice,
              aoi       = aoi,
              dest_dir  = out_dir,
              .progress = TRUE)
})

message(" All requested IMERG slices are now in: ", normalizePath(out_dir))
