library(climateR)
library(sf)
library(glue)
library(lubridate)
library(purrr)
library(curl)
library(terra)

### https://urs.earthdata.nasa.gov/ ####
# if no earth data login, then need to define these
# NASA_DATA_USER = "XX"
# NASA_DATA_PASSWORD = "XX"
########################################

# make sure a .netrc file exists, if not, create one
if(!climateR::checkNetrc()){
  message("Writting a '.netrc' file...")
  climateR::writeNetrc(
    login     = NASA_DATA_USER,
    password  = NASA_DATA_PASSWORD,
    netrcFile = getNetrcPath())
}

# Should be TRUE
climateR::checkNetrc(machine = "urs.earthdata.nasa.gov")

path <- getNetrcPath()
cat("Netrc path:", path, "\n")
cat(readLines(path), sep = "\n")

# # # test to see if EarthData access is working:
test_url <- "https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGHH.07/2024/002/3B-HHR.MS.MRG.3IMERG.20240102-S000000-E002959.0000.V07B.HDF5.dmr.html"
test_url <- "https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGHH.07/2024/002/3B-HHR.MS.MRG.3IMERG.20240102-S000000-E002959.0000.V07B.HDF5.dap.nc4"
test_url <- "https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGHH.07/2024/002/3B-HHR.MS.MRG.3IMERG.20240102-S000000-E002959.0000.V07B.HDF5.dap.nc4"
system(sprintf('curl --netrc-file "%s" -L "%s"',"C:/Users/EmmaGolub/.netrc",test_url))


# define area of interest -----------------------------------------------------

aoi = st_as_sfc(st_bbox(c(xmin = -121,
                          xmax = -118,
                          ymax = 40,
                          ymin = 35.5),
                        crs = 4326))

# define functions -------------------------------------------------------------
# need to iteratively create links for each time and grid area for opendap calls

# product version is the FINAL version!
# only uses the v7 outputs!
opendap_urls <- function(date,
                         run = "final") {

  if (run == "final") {
    product_id = "3B-HHR.MS.MRG.3IMERG"
    product = "GPM_3IMERGHH.07"
  } else if(run == "late") {
    product_id = "3B-HHR-L.MS.MRG.3IMERG"
    product = "GPM_3IMERGHHL.07"
  } else {
    stop("suppply either 'final' or 'late'")
  }

  # get the date/times configured correctly
  date <- as.Date(date)
  year    <- format(date, "%Y")
  julian  <- format(date, "%j")
  origin_time <- as.POSIXct(paste0(date, "00:00"), tz = "UTC")

  base_url <- glue("https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/{product}/{year}/{julian}/")


  # 30-minute interval urls
  times <- seq(ymd_hms(glue("{date} 00:00:00")), by = "30 mins", length.out = 48)

  # create urls based on date (should be 48 for each day)
  urls <- sapply(times, function(t) {
    start_str <- format(t, "%H%M%S")
    end_str <- format(t + minutes(29) + seconds(59), "%H%M%S")
    file_time <- format(t, "%Y%m%d")
    minutes_diff <- sprintf("%04d", as.numeric(difftime(t, as.POSIXct(date), units = "mins")))
    filename <- glue("{product_id}.{file_time}-S{start_str}-E{end_str}.{minutes_diff}.V07B.HDF5.dmr.html")

    # finalize the url!
    glue("{base_url}{filename}")
  })

  return(urls)
}

# # get the GPM data using climateR::dap()
# get_gpm <- function(url, aoi) {
#
#   # assign GPM variable
#   var = 'probabilityLiquidPrecipitation'
#
#   tryCatch({
#
#     # Get Data
#     gpm_obs =
#       climateR::dap(
#         URL     = url,
#         varname = var,
#         AOI     = aoi,
#         verbose = FALSE
#       )
#
#     message("Succesfully retrieved GPM data using climateR::dap()")
#     message("GPM data column names: ", paste0(names(gpm_obs), collapse = ", "))
#
#     # drop geometry column to make it dataframe
#     gpm_obs <- sf::st_drop_geometry(gpm_obs)
#
#     df <- as.data.frame(gpm_obs[["probabilityLiquidPrecipitation"]], xy = TRUE)
#
#     return(df)
#
#   }, error = function(er) {
#     message("Error FAILED to get GPM data using climateR::dap() returning NA value...")
#     message("Original error message:")
#     message(conditionMessage(er))
#
#     # return NA if error
#     return(NA)
#
#   })
#
#
# }

# Windows System Alternative:---------
earthdata_handle <- curl::new_handle(
  netrc      = 1,
  netrc_file = normalizePath("C:/Users/EmmaGolub/.netrc"),
  followlocation = TRUE
)

download_slice <- function(hdf_url,
                           varname  = "probabilityLiquidPrecipitation",
                           aoi      = NULL,
                           dest_dir = tempdir()) {

  dest <- file.path(dest_dir, basename(hdf_url))
  if (!file.exists(dest)) {
    curl::curl_download(
      hdf_url,
      destfile = dest,
      handle   = earthdata_handle
    )
  }

  sds  <- terra::sds(dest)$subdatasets
  subd <- grep(varname, sds, value = TRUE)
  if (length(subd) != 1) stop("variable not found: ", varname)

  r <- terra::rast(subd)
  if (!is.null(aoi)) r <- terra::crop(r, aoi)
  r
}


# replace get_gpm() with local‑file version
get_gpm <- function(url, aoi) {
  tryCatch({
    r  <- download_slice(url, aoi = aoi)            # download individual SpatRaster
    df <- as.data.frame(r, xy = TRUE, na.rm = FALSE)
    names(df)[3] <- "probabilityLiquidPrecipitation"
    df
  }, error = function(e) {
    message("Download/read failed for: ", url)
    NA
  })
}

# now for the whole day
get_gpm_day <- function(day, aoi, run) {

  # day in the format of "YYYY-MM-DD"
  urls <- opendap_urls(date = day,
                       run)

  start_all <- Sys.time()

  all_dfs <- list()

  for (i in seq_along(urls)) {

    start <- Sys.time()

    url_1 <- urls[i]

    gpm_data <- get_gpm(url_1, aoi)

    all_dfs[[i]] <- gpm_data

    end <- Sys.time()

    message(paste("Done with collecting url", i, "of", length(urls),
                   "took:", round(difftime(end, start, units = "secs"), 2), "seconds\n"))

    }

  # remove NULL or empty dfs
  all_dfs_clean <- all_dfs[!sapply(all_dfs, is.null)]

  if (length(all_dfs_clean) == 0) {
    message("No data retrieved, returning NA")
    return(NA)
  }

  # merge all dfs by x and y
  final_df <- purrr::reduce(all_dfs_clean, ~ merge(.x, .y, by = c("x", "y"), all = TRUE))

  names(final_df) <- gsub("_", "", names(final_df))

  end_all <- Sys.time()

  message(paste("Data collection for", day, "took:", round(difftime(end_all, start_all, units = "secs"), 2), "seconds\n"))
  return(final_df)

}


# test -------------------------------------------------------------------------
# all_days <- seq(as.Date("2024-10-01"), as.Date("2025-05-30"), by = "day")

day = "2025-02-01"
# final run is only available at this moment (2025-07-24) to 2025-02-28
day_test <- get_gpm_day(day = day,
                        aoi = aoi,
                        run = "final" # final or late
                        )

arrow::write_parquet(day_test,
                     '/Data/gpm_imerg/gpm_imerg_20241001.parquet')

# running these processes in parallel causes timeout issues... :()
# not sure if this can be done in batches?
# test that using purrr?




# see the raw gpm gridded output ?
url_test <- opendap_urls(day)[5]
gpm_obs_raw =
  climateR::dap(
    URL     = url_test,
    varname = 'probabilityLiquidPrecipitation',
    AOI     = aoi,
    verbose = FALSE
  )

plot(gpm_obs_raw[["probabilityLiquidPrecipitation"]])
# cool :)
