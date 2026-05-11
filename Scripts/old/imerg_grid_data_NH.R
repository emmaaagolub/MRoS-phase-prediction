library(climateR)
library(sf)
library(glue)
library(lubridate)
library(furrr)
library(purrr)

# PURPOSE:
# This script downloads half-hourly GPM IMERG Probability of Liquid Precipitation (PLP) data from NASA’s OPeNDAP servers, 
# clips it to a fixed Sierra Nevada AOI, merges all 48 half-hourly timesteps into a daily gridded table, 
# and saves the result as a daily parquet file.

# Final output:
# One parquet file per day
# Each file contains:
#  - Grid coordinates
#  - PLP values for all half-hourly timesteps
# Spatially consistent, temporally dense precipitation-phase proxy


### https://urs.earthdata.nasa.gov/ ####
# if no earth data login, then need to define these
# NASA_DATA_USER = "XX"
# NASA_DATA_PASSWORD = "XX"
########################################

# make sure a .netrc file exists, if not, create one
if(!climateR::checkNetrc()){
  message("Writting a '.netrc' file...")
  climateR::writeNetrc(login = NASA_DATA_USER, 
                       password = NASA_DATA_PASSWORD)
}

# define area of interest -----------------------------------------------------

aoi = st_as_sfc(st_bbox(c(xmin = -121, 
                          xmax = -118, 
                          ymax = 40, 
                          ymin = 35.5), 
                        crs = 4326)) %>% 
  st_sf()

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
    filename <- glue("{product_id}.{file_time}-S{start_str}-E{end_str}.{minutes_diff}.V07B.HDF5")
    
    # finalize the url!
    glue("{base_url}{filename}")
  })
  
  return(urls)
}

# get the GPM data using climateR::dap()
get_gpm <- function(url, aoi) {
  
  # assign GPM variable
  var = 'probabilityLiquidPrecipitation'
  
  tryCatch({
    
    # Get Data
    gpm_obs =
      climateR::dap(
        URL     = url,
        varname = var,
        AOI     = aoi,
        verbose = FALSE
      )
    
    message("Succesfully retrieved GPM data using climateR::dap()")
    message("GPM data column names: ", paste0(names(gpm_obs), collapse = ", "))
    
    # drop geometry column to make it dataframe
    gpm_obs <- sf::st_drop_geometry(gpm_obs)
    
    df <- as.data.frame(gpm_obs[["probabilityLiquidPrecipitation"]], xy = TRUE)
    
    return(df)
    
  }, error = function(er) {
    message("Error FAILED to get GPM data using climateR::dap() returning NA value...")
    message("Original error message:")
    message(conditionMessage(er))
    
    # return NA if error
    return(NA)
    
  })
  
  
}



get_gpm_day <- function(day, aoi, run) {
  
  # batch processing
  batch_size <- 5
  pause_seconds <- 5  # pause between batches 
  
  # url list
  urls <- opendap_urls(date = day, run)
  total_urls <- length(urls)
  num_batches <- ceiling(total_urls / batch_size)
  
  all_dfs <- vector("list", total_urls)
  
  # set up the p-processing
  plan(multisession, workers = 3)
  
  start_all <- Sys.time()
  
  for (batch_num in seq_len(num_batches)) {
    batch_start <- (batch_num - 1) * batch_size + 1
    batch_end <- min(batch_num * batch_size, total_urls)
    current_batch <- urls[batch_start:batch_end]
    
    message(sprintf("Starting batch %d/%d: URLs %d to %d", 
                    batch_num, num_batches, batch_start, batch_end))
    
    # parallel proc. of get_gpm with necessary packages loaded
    batch_results <- future_map(
      current_batch,
      ~ {
        library(sf)
        library(terra)
        library(climateR)
        
        start <- Sys.time()
        result <- get_gpm(.x, aoi)
        end <- Sys.time()
        
        message(sprintf("  Done with URL %d of %d (%.2f secs)", 
                        which(urls == .x), total_urls, as.numeric(difftime(end, start, units = "secs"))))
        result
      },
      .options = furrr_options(
        seed = TRUE,
        packages = c("sf", "terra", "climateR")
      )
    )
    
    all_dfs[batch_start:batch_end] <- batch_results
    
    if (batch_num < num_batches) {
      message(sprintf("Pausing for %d seconds before next batch...", pause_seconds))
      Sys.sleep(pause_seconds)
    }
  }
  
  # clean results
  all_dfs_clean <- all_dfs[!sapply(all_dfs, is.null)]
  
  if (length(all_dfs_clean) == 0) {
    message("No data retrieved, returning NA")
    return(NA)
  }
  
  # merge by lat lon
  final_df <- purrr::reduce(all_dfs_clean, ~ merge(.x, .y, by = c("x", "y"), all = TRUE))
  names(final_df) <- gsub("_", "", names(final_df))
  
  end_all <- Sys.time()
  message(paste("Data collection for", day, "took:", round(difftime(end_all, start_all, units = "secs"), 2), "seconds\n"))
  
  return(final_df)
}



# run -------------------------------------------------------------------------
all_days <- seq(as.Date("2024-10-01"), as.Date("2025-05-30"), by = "day")

max = length(all_days)
gdrive = '/Users/nhur/Library/CloudStorage/GoogleDrive-nhur@lynker.com/My Drive/mountain_rain_or_snow/Precipitation phase product prototype/gpm_imerg/imerg_data/'
local = '/Users/nhur/Documents/Projects/MRoS/wy25/imerg_data/'



for (i in 204:max) {

  if (i %% 20 == 0) {
    gc()
    future::plan(sequential)
  }
  
  day = all_days[i]
  
  # final run is only available at this moment (2025-07-24) to 2025-02-28
  # onyl get final values all days before feb
  
  # if (as.Date(day) < as.Date("2025-02-01")) {
  #   
  #   day_test <- get_gpm_day(day = day, 
  #                           aoi = aoi,
  #                           run = "final" 
  #   )
  #   
  # } else {
    
    day_test <- get_gpm_day(day = day, 
                            aoi = aoi,
                            run = "late"
    )
    
  # }
  
  arrow::write_parquet(day_test, 
                       paste0(local, glue('/gpm_imerg_{day}.parquet')))
  
  print(paste("Done with processing", day, ":", i, "of", max))
  
}





# # see the raw gpm gridded output ?
# url_test <- opendap_urls(day)[5]
# gpm_obs_raw =
#   climateR::dap(
#     URL     = url_test,
#     varname = 'probabilityLiquidPrecipitation',
#     AOI     = aoi,
#     verbose = FALSE
#   )
# 
# plot(gpm_obs_raw[["probabilityLiquidPrecipitation"]])
# # cool :)
