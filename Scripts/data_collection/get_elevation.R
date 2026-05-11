###############################################################
#  DEM download & clip
#  1/3-arc-second (≈10 m) DEMs from USGS TNM for two AOIs:
#    - California Sierra Nevada / Lake Tahoe
#    - Colorado Mountains
#
#  Outputs saved to: <repo>/Data/Elevation/
###############################################################


## ---------------- 1.  SET-UP ---------------------------------

pkg_needed <- c("terra", "sf")
inst <- pkg_needed[!(pkg_needed %in% installed.packages()[, "Package"])]
if (length(inst)) install.packages(inst, repos = "https://cloud.r-project.org")

library(terra)
library(sf)


## Resolve paths -----------------------------------------------
# Script lives at: <repo>/Scripts/data_collection/get_elevation.R
# Repo root is two levels up from script_dir

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
out_dir    <- file.path(repo_root, "Data", "Elevation")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

message("Repo root : ", repo_root)
message("Output dir: ", out_dir)


## ---------------- 2.  DEFINE AOI POLYGONS --------------------
# Coordinates supplied as (lat, lon) pairs — stored here as (lon, lat) for sf

# --- California Sierra Nevada / Lake Tahoe ---
california_pts <- rbind(
  c(-119.45505750721992, 39.65343608043361),
  c(-121.27878797084242, 39.66189413918429),
  c(-119.11448133630248, 36.726935737063016),
  c(-118.49924696303225, 37.235952484988736),
  c(-119.46604383531404, 38.37304030164334)
)

# --- Colorado Mountains ---
colorado_pts <- rbind(
  c(-105.19885928678391, 40.62046076499234),
  c(-106.88927700287375, 40.555465783967925),
  c(-107.66078646416787, 38.79540171139857),
  c(-104.87856373593310, 38.77382201116306)
)

make_aoi_ext <- function(pts) {
  # Close the ring (sf requires first == last point)
  ring   <- rbind(pts, pts[1, ])
  poly   <- st_polygon(list(ring))
  sf_obj <- st_sfc(poly, crs = 4326)
  ext(vect(sf_obj))   # returns terra SpatExtent
}

california_extent    <- make_aoi_ext(california_pts)
colorado_ext <- make_aoi_ext(colorado_pts)

message("California extent   : ", paste(as.vector(california_extent),    collapse = ", "))
message("Colorado extent: ", paste(as.vector(colorado_ext), collapse = ", "))


## ---------------- 3.  DOWNLOAD HELPER ------------------------

vrt_url <- "/vsicurl/https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/USGS_Seamless_DEM_13.vrt"

download_dem <- function(region_name, aoi_ext, out_dir, vrt_url) {
  out_file <- file.path(out_dir, paste0(region_name, "_DEM_AOI_TNM_10m.tif"))
  
  if (file.exists(out_file)) {
    message(region_name, ": file already exists, skipping — ", out_file)
    return(invisible(out_file))
  }
  
  message(region_name, ": opening USGS VRT...")
  vrt <- rast(vrt_url)
  
  message(region_name, ": cropping to AOI extent...")
  dem <- crop(vrt, aoi_ext)
  
  message(region_name, ": writing -> ", out_file)
  writeRaster(
    dem,
    filename  = out_file,
    datatype  = "INT16",
    gdal      = c("COMPRESS=LZW"),
    overwrite = TRUE
  )
  
  message(region_name, ": done.")
  invisible(out_file)
}


## ---------------- 4.  DOWNLOAD BOTH REGIONS ------------------

download_dem("california", california_extent,    out_dir, vrt_url)
download_dem("colorado",     colorado_ext, out_dir, vrt_url)

message("\nAll DEMs saved to: ", out_dir)