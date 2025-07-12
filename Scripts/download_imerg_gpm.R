###############################################################
## citsci_imerg_download.R
##
## Download GPM IMERG half-hour PLP for a custom date range
## IMERG lives as HDF5/netCDF on NASA servers
## OPeNDAP streaming: get_imerg() uses netCDF streaming and gives back an R data frame of values at specified locations...
## Option to download as gridded tiffs included in the end.
###############################################################


## ---------------- 1.  SET-UP ---------------------------------
pkgs <- c("devtools","tidyverse","lubridate","readr","sf","climateR",
          "terra","furrr","progressr","glue", "httr")
for(p in pkgs) if(!requireNamespace(p, quietly=TRUE)) install.packages(p)
lapply(pkgs, library, character.only=TRUE)

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
out_dir <- normalizePath(file.path(script_dir, "..", "Data", "gpm_imerg"), mustWork = TRUE)

## Earthdata auth
Sys.getenv("HOME")
Sys.getenv("NETRC")
file.exists(Sys.getenv("NETRC"))
readLines(Sys.getenv("NETRC"))
set_config( config(netrc = 1L, netrc_file = Sys.getenv("NETRC")) )
## In windows, need to adjust security permissions in terminal to replicate in Linux system:
# > cd $Env:USERPROFILE
# > icacls ".\.netrc" /inheritance:r
# > icacls ".\.netrc" /grant:r "EmmaGolub:F"
# > icacls ".\.netrc" /grant:r "SYSTEM:R"
# > icacls ".\.netrc"


## ---------------- 2.  USER SETTINGS ---------------------------------
# 4-hour test window on Mar 1 2025, half-hourly steps
datetimes <- seq(
  as.POSIXct("2025-03-01 00:00:00", tz="UTC"),
  as.POSIXct("2025-03-01 04:00:00", tz="UTC"),
  by = "30 min"
)


# Bring in MRoS observation records (NULL for now until access to parquet)
points_file <- NULL
# points_file <- "/path/to/your/observations.parquet"

if(!is.null(points_file) && file.exists(points_file)) {
  points <- arrow::read_parquet(points_file) %>%
    rename(
      placeholder_obs_id = id_column, # adjust to what's actually in parquet
      lon        = lon_column,     # adjust to what's actually in parquet
      lat        = lat_column      # adjust to what's actually in parquet
    ) %>%
    select(placeholder_obs_id, lon, lat)
} else {
  points <- tibble(
    placeholder_obs_id = "TEST1",
    lon        = -105.237502,
    lat        =  39.094364
  )
}

# Expand to every combination of point × datetime
queries <- crossing(points, datetime_utc = datetimes)

## ---------------- 3.  RUN GET_IMERG() ---------------------------------
results <- queries %>%
  mutate(
    plp = pmap_dbl(
      list(datetime_utc, lon, lat),
      function(dt, lo, la) {
        get_imerg(
          datetime_utc    = dt,
          lon_obs         = lo,
          lat_obs         = la,
          product_version = "GPM_3IMERGHHL.07",
          verbose         = TRUE
        )
      }
    )
  )

# save the point‐wise PLP table
write_csv(
  results,
  file.path(out_dir, "imerg_PLP_20250301_4h.csv") # change output name accordingly depending on time frame run
)

## ---------------- 4. DOWNLOAD TIFFS ---------------------------------
# Set this to TRUE to grab a small GeoTIFF around each obs point:
download_tiffs <- TRUE

if(download_tiffs) {
  # helper: download & crop one half-hour slice around one point
  download_imerg_tif <- function(datetime, lon, lat, buffer_deg = 0.2) {
    # build the catalog URL
    base_url  <- construct_gpm_base_url(format(datetime, "%Y-%m-%dT%H:%M:%OSZ"))
    catalog   <- paste0(base_url, "catalog.xml")
    urls      <- get_final_urls(catalog)
    stub      <- construct_gpm_product(format(datetime, "%Y-%m-%dT%H:%M:%OSZ"))
    dap_url   <- if(any(idx <- grepl(stub, urls))) urls[idx][1] else get_closest_url(urls, stub)

    # read the full IMERG PLP grid via OPeNDAP
    r_full <- terra::rast(dap_url, subds="probabilityLiquidPrecipitation")

    # crop to a small box around (lon, lat)
    ext <- terra::ext(lon - buffer_deg, lon + buffer_deg,
                      lat - buffer_deg, lat + buffer_deg)
    r_crop <- terra::crop(r_full, ext)

    # write out
    ts_lbl <- format(datetime, "%Y%m%dT%H%M")
    out_f  <- file.path(out_dir,
                        glue("IMERG_PLP_{ts_lbl}_{lon}_{lat}.tif"))
    terra::writeRaster(r_crop, out_f, overwrite=TRUE)
    invisible(out_f)
  }

  # loop over results (this will be slow if many rows!)
  # If only datetime, lon, lat, then drop placeholder_obs_id here
  walk3(
    results$datetime_utc,
    results$lon,
    results$lat,
    ~ download_imerg_tif(.x, .y, .z)
  )
}
