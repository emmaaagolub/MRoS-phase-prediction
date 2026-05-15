library(climateR)
library(sf)
library(tidyverse)
library(lubridate)
library(httr)
library(terra)


# PURPOSE:
# This script downloads daily PRISM climate rasters, clips them to a Sierra Nevada AOI, 
# stacks multiple meteorological variables into daily multi-band GeoTIFFs, and then converts 
# the full time series into a single parquet table of gridded predictors.

# Downloads five daily PRISM variables:
# - ppt – precipitation
# - tmin – minimum temperature
# - tmean – mean temperature
# - tmax – maximum temperature
# - tdmean – mean dewpoint temperature

# Final output:
# Daily stacked GeoTIFFs (one per day)
# One parquet file containing:
# - Spatial coordinates
# - Daily PRISM climate predictors
# - Full WY-scale time series


# read in grid file
grid <- st_read('sierras_aoi_prismGrid.gpkg')
aoi <- st_read('sierras_aoi.shp')

# split this into 500 locations

# split the prism data into chunks
# 
# chunk_size <- 500
# num_rows <- nrow(grid)
# num_chunks <- ceiling(num_rows / chunk_size)
# 
# for (i in 1:num_chunks) {
#   start_row <- (i - 1) * chunk_size + 1
#   end_row <- min(i * chunk_size, num_rows)
# 
#   chunk <- grid[start_row:end_row, ]
#   
#   chunk <- chunk %>% 
#     st_drop_geometry() %>% 
#     mutate(Name = paste0(col, '-', row)) %>% 
#     select(Latitude = lat,
#            Longitude = lon,  
#            Name)
#     
#   filename <- paste0('clipped/prism_800m_sierras_', i,'.csv')
# 
#   # write the chunk to a CSV file
#   write.table(chunk, file = filename, row.names = FALSE, col.names = FALSE, sep=",")
# 
#   rm(chunk)
# }

aoi <- st_transform(aoi, 4326)

dates <- seq(ymd("2024-10-01"), ymd("2025-09-30"), by = "day")
date_codes <- format(dates, "%Y%m%d")

variables <- c("ppt", "tmin", "tmean", "tmax", "tdmean")
region <- "us"
res <- "800m"
outdir <- "prism_daily_stacked"

dir.create(outdir, showWarnings = FALSE)

download_prism_var <- function(variable, date_code, aoi, outdir, region = "us", res = "800m") {
  base_url <- "https://services.nacse.org/prism/data/get"
  url <- sprintf("%s/%s/%s/%s/%s?format=bil", base_url, region, res, variable, date_code)
  
  tempdir_day <- file.path(tempdir(), paste0(variable, "_", date_code))
  dir.create(tempdir_day, showWarnings = FALSE)
  
  zip_path <- file.path(tempdir_day, paste0(variable, "_", date_code, ".zip"))
  res <- try(GET(url, write_disk(zip_path, overwrite = TRUE), timeout(60)), silent = TRUE)
  if (inherits(res, "try-error") || res$status_code != 200) {
    warning(paste("Failed to download:", variable, date_code))
    return(NULL)
  }
  
  unzip(zip_path, exdir = tempdir_day)
  bil_file <- list.files(tempdir_day, pattern = "\\.bil$", full.names = TRUE)
  if (length(bil_file) == 0) return(NULL)
  
  r <- rast(bil_file)
  r <- crop(r, aoi) |> mask(aoi)
  
  unlink(tempdir_day, recursive = TRUE)
  return(r)
}

# loop through days and stack variables

for (d in date_codes) {
  stacked_path <- file.path(outdir, paste0("PRISM_", d, "_stacked.tif"))
  if (file.exists(stacked_path)) {
    cat("Already processed:", d, "\n")
    next
  }
  
  cat("Processing date:", d, "\n")
  daily_layers <- list()
  
  for (v in variables) {
    cat("  - downloading", v, "\n")
    r <- download_prism_var(v, d, aoi, outdir, region, res)
    if (!is.null(r)) {
      names(r) <- v
      daily_layers[[v]] <- r
    }
    Sys.sleep(0.3)
  }
  
  if (length(daily_layers) > 0) {
    r_stack <- rast(daily_layers)
    writeRaster(r_stack, stacked_path, overwrite = TRUE)
  }
}

# read the data

outdir <- "prism_daily_stacked"
files <- list.files(outdir, pattern = "_stacked\\.tif$", full.names = TRUE)

# extract the date code from filenames
dates <- gsub(".*PRISM_(\\d{8})_stacked\\.tif$", "\\1", files)

daily_list <- lapply(seq_along(files), function(i) {
  r <- rast(files[i])
  # extract cell coordinates and values
  df <- as.data.frame(r, xy = TRUE, na.rm = FALSE)
  names(df) <- c("lon", "lat", names(r))
  return(df)
})

names(daily_list) <- dates

# combine the data
combined_df <- bind_rows(
  lapply(seq_along(daily_list), function(i) {
    df <- daily_list[[i]]
    df$date <- dates[i]
    df
  })
)

head(combined_df)

# write output as parquet
arrow::write_parquet(combined_df, "combined_prism_20241001_20250601.parquet")

# look at the rast stack!
r_stack <- rast('/Users/nhur/Documents/Projects/MRoS/wy25/prism_daily_stacked/PRISM_20250214_stacked.tif')
plot(r_stack)

