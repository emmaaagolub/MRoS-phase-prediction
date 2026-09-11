## Shared region definitions and paths for the R download scripts.
## Sourced by get_elevation.R, download_station_data.R, download_imerg.R and
## download_prism.R.

suppressPackageStartupMessages({
  library(sf)
})

## Study period, matching refactored/config.py
WY_START <- as.POSIXct("2022-10-01 00:00:00", tz = "UTC")
WY_END   <- as.POSIXct("2026-05-01 23:59:59", tz = "UTC")

## Area-of-interest vertices, (lon, lat) in WGS84.
REGION_AOI <- list(
  CA = rbind(
    c(-119.45505750721992, 39.65343608043361),
    c(-121.27878797084242, 39.66189413918429),
    c(-119.11448133630248, 36.726935737063016),
    c(-118.49924696303225, 37.235952484988736),
    c(-119.46604383531404, 38.37304030164334)
  ),
  CO = rbind(
    c(-105.19885928678391, 40.62046076499234),
    c(-106.88927700287375, 40.555465783967925),
    c(-107.66078646416787, 38.79540171139857),
    c(-104.87856373593310, 38.77382201116306)
  )
)

REGION_LABEL <- list(
  CA = "California: Sierra Nevada / Lake Tahoe",
  CO = "Colorado Mountains"
)

## Filename stem used by get_elevation.R for the raw 10 m DEMs.
REGION_DEM_STEM <- list(CA = "california", CO = "colorado")

REGION_IDS <- names(REGION_AOI)


## Repo root, found from the location of the running script so paths do not
## depend on the working directory.
repo_root <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  hit  <- grep("--file=", args)
  if (length(hit) > 0) {
    script_dir <- dirname(normalizePath(sub("--file=", "", args[hit[1]])))
  } else if (requireNamespace("rstudioapi", quietly = TRUE) &&
             rstudioapi::isAvailable()) {
    script_dir <- dirname(normalizePath(rstudioapi::getSourceEditorContext()$path))
  } else {
    stop("Cannot locate script; run with Rscript or from RStudio.")
  }
  # script lives at <repo>/refactored/preprocessing/
  normalizePath(file.path(script_dir, "..", ".."))
}


## Build an sf polygon from AOI vertices.
##   round_dp: the IMERG and PRISM downloads were originally run with the
##   vertices rounded to 3 decimals. Kept so re-runs match existing files.
##   hull: order the vertices by convex hull rather than as listed.
aoi_polygon <- function(region_id, round_dp = NULL, hull = FALSE) {
  pts <- REGION_AOI[[region_id]]
  if (!is.null(round_dp)) pts <- round(pts, round_dp)
  if (hull) pts <- pts[chull(pts), ]
  ring <- rbind(pts, pts[1, ])
  st_sf(st_sfc(st_polygon(list(ring)), crs = 4326))
}


## Ensure a directory exists and return it.
ensure_dir <- function(path) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
  path
}
