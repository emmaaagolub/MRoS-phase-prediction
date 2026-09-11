## Download 1/3 arc-second (~10 m) DEMs from the USGS National Map
## and clip them to each study area.
##
## Output: Data/Elevation/<region>_DEM_AOI_TNM_10m.tif

suppressPackageStartupMessages({
  library(terra)
  library(sf)
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

OUT_DIR <- ensure_dir(file.path(repo_root(), "Data", "Elevation"))
VRT_URL <- paste0(
  "/vsicurl/https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/",
  "TIFF/USGS_Seamless_DEM_13.vrt"
)

for (region_id in REGION_IDS) {
  stem     <- REGION_DEM_STEM[[region_id]]
  out_file <- file.path(OUT_DIR, paste0(stem, "_DEM_AOI_TNM_10m.tif"))

  if (file.exists(out_file)) {
    message(region_id, ": already downloaded, skipping.")
    next
  }

  aoi_ext <- ext(vect(aoi_polygon(region_id)))
  message(region_id, ": cropping USGS mosaic to ",
          paste(round(as.vector(aoi_ext), 3), collapse = ", "))

  dem <- crop(rast(VRT_URL), aoi_ext)
  writeRaster(dem, out_file, datatype = "INT16",
              gdal = c("COMPRESS=LZW"), overwrite = TRUE)

  message(region_id, ": wrote ", basename(out_file))
}

message("DEMs saved to ", OUT_DIR)
