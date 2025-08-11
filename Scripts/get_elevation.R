###############################################################
#  DEM download & clip for the Lake Tahoe / Sierra AOI
#  1/3-arc-second (≈10 m) DEM from USGS TNM
###############################################################

## ---------------- 1.  SET-UP ---------------------------------

## Install/load packages & the local repo  ----------------
pkg_needed <- c("devtools", "tidyverse", "lubridate", "purrr", "progress", "readr", "terra")
inst <- pkg_needed[!(pkg_needed %in% installed.packages()[,"Package"])]
if (length(inst)) install.packages(inst, repos = "https://cloud.r-project.org")

library(devtools)
library(tidyverse)
library(lubridate)
library(purrr)
library(progress)
library(readr)
library(terra)

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

## Set output Dir
out_dir <- normalizePath(file.path(script_dir, "..", "Data", "Elevation"), mustWork = TRUE)




## ---------------- 2.  GET ELEVATION  ---------------------------------

# AOI extent
aoi <- ext(-121, -118, 35.5, 40)

# Seamless 1/3″ VRT (≈10 m)
vrt <- rast("/vsicurl/https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/USGS_Seamless_DEM_13.vrt")

dem <- crop(vrt, aoi)      # terra pulls only intersecting tiles

writeRaster(dem,
            filename = "Data/Stations/DEM_AOI_TNM_10m.tif",
            datatype  = "INT16",
            gdal      = c("COMPRESS=LZW"),
            overwrite = TRUE)
