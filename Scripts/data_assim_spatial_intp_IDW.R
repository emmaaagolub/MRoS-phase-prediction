################################################################################
## IDW Spatial Interpolation


### Part A: Build off of rainOrSnowTools
# Inputs
# MRoS: lon, lat, datetime_utc, phase
# Stations: raw time series + metadata (lat/lon, elevation, tz)
# IMERG: PLP (half-hourly)
#
# Time syncing
# Convert station timestamps to UTC using each station’s timezone
# Select station readings closest in time within a ±1h window
# Sample IMERG at the obs time/point (or nearest half-hour; average to the hour when appropriate). IMERG is natively half-hourly
#
# Spatialization to the obs point
# Use the same IDW (+ lapse) routine as in the MRoS tools to downscale station predictors (Tair, Td, Twb, RH) to the MRoS point (carry both DEM elevation and station meta-elevation for QA)
# Attach IMERG PLP to each row
#
# Output: one tidy row per obs with predictors (station-derived + IMERG + elevation) and target (MRoS phase)

### Part B: make surfaces that correspond to the same modeling assumptions as the per-obs pipeline
# Pick an hour t (UTC).
# Stations → grid: run the same IDW+lapse logic to produce predictor grids (Tair/Td/Twb/RH) for hour t
# IMERG → grid: regrid/resample IMERG PLP to your map grid (don’t re-interpolate PLP with IDW—it’s already gridded)
# Interpolate MRoS phase to “p_snow”/“p_mix”
# Visuals: side-by-side predictor grids and PLP; overlay MRoS points for that hour

### Part C: Add other gridded data
# MERRA-2
# NLDAS-2
################################################################################


## 1.  SET-UP ------------------------------------------------------------------

# Install/load packages & the local repo  ----------------
packages <- c("devtools", "tidyverse", "lubridate", "arrow", "sf", "terra", "gt", "rstudioapi", "gstat")

installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}
invisible(lapply(packages, library, character.only = TRUE))

devtools::install_github("LynkerIntel/rainOrSnowTools")
library(rainOrSnowTools)

# Folder setups
source.code <- getSourceEditorContext()$path
script_folder <- dirname(source.code)
temp_folder <- dirname(script_folder)
parent_folder <- dirname(temp_folder)



## 2. LOAD DATA ----------------------------------------------------------------
mros_parquet      = "./Data/observations/wy25_mros_obs.parquet"
stations_dir      = "./Data/Stations/"
stations_meta_csv = "./Data/Stations/station_metadata_20241001_20250531.csv"
imerg_dir         = "./Data/IMERG/imerg_data-20250731T220028Z-1-001/imerg_data"
dem_path          = "C:/Users/EmmaGolub/Desktop/MRoS_local/local_data/DEM_AOI_TNM_10m.tif" # sourcing from personal drive because too large to push to repo
out_path          <- "./outputs/"
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)


# Read & prep MRoS
mros <- arrow::read_parquet(mros_parquet) %>%
  as_tibble() %>%
  mutate(
    latitude   = as.numeric(latitude),
    longitude  = as.numeric(longitude),
    datetime_utc = mdy_hms(paste(date_submitted_utc, time_submitted_utc), tz = "UTC")
  ) %>%
  arrange(datetime_utc)

# Elevation for each obs from DEM (meters)
dem <- terra::rast(dem_path)
mros_sf <- st_as_sf(mros, coords = c("longitude","latitude"), crs = 4326, remove = FALSE)
mros$elev_m <- terra::extract(dem, terra::vect(mros_sf))[,2]
mros_filter <- mros %>%
  select(phase, latitude, longitude, datetime_utc, elev_m) %>%
  # add row_id to MRoS
  mutate(row_id = row_number())

# Read & prep station MET data into the EXACT schema select_meteo expects
# select_meteo(df, datetime_obs) expects:
#   df columns: id, datetime (POSIXct), and any of temp_air, temp_wet, temp_dew, rh
meta_df <- readr::read_csv(stations_meta_csv, show_col_types = FALSE) %>%
  transmute(id = as.character(id), lat, lon, elev)

station_files <- list.files(stations_dir, pattern = "\\.csv$", full.names = TRUE)
station_files <- station_files[!grepl("meta", station_files, ignore.case = TRUE)]

stations_list <- lapply(station_files, function(f) {
  df <- readr::read_csv(f, show_col_types = FALSE)
  if (!"id" %in% names(df))
    df$id <- tools::file_path_sans_ext(basename(f))

  df$id <- as.character(df$id)

  # Make sure datetime column is parsed (format like 2024-10-01T00:12:00Z)
  if ("datetime" %in% names(df)) {
    df$datetime <- ymd_hms(df$datetime, tz = "UTC")
  }

  # Add any missing columns with NA
  for (nm in c("temp_air","temp_wet","temp_dew","rh")) {
    if (!nm %in% names(df)) df[[nm]] <- NA_real_
  }

  df %>% select(id, datetime, temp_air, temp_wet, temp_dew, rh)
})

stations_long <- bind_rows(stations_list) %>%
  arrange(id, datetime)



## 3. BUILD PER-OBS PREDICTORS -------------------------------------------------
# PART A - DATA ASSIMILATIONS

# select_meteo() returns the NEAREST-in-time per station/variable, but it does not enforce a max time gap window. To enforce ±1h, add a filter on time_gap afterwards.
max_gap <- lubridate::hours(1)

# Function to process a single obs row
process_obs <- function(obs_row, stations_long, meta_df) {
  lon_obs  <- obs_row$longitude
  lat_obs  <- obs_row$latitude
  t_obs    <- obs_row$datetime_utc
  elev_obs <- obs_row$elev_m
  row_id   <- obs_row$row_id

  # nearest-in-time per station/variable
  sel <- rainOrSnowTools:::select_meteo(df = stations_long, datetime_obs = t_obs)

  # enforce ±1h window
  sel <- sel %>%
    left_join(
      stations_long %>%
        group_by(id) %>%
        slice_min(abs(datetime - t_obs), n = 1, with_ties = FALSE) %>%
        ungroup() %>%
        mutate(time_gap = abs(datetime - t_obs)) %>%
        select(id, time_gap),
      by = "id"
    )

  # Only filter if time_gap exists
  if ("time_gap" %in% names(sel)) {
    sel <- sel %>%
      filter(is.na(time_gap) | time_gap <= max_gap) %>%
      select(-time_gap)
  }

  # QC
  sel_qc <- rainOrSnowTools:::qc_meteo(sel)

  # model
  modeled <- rainOrSnowTools:::model_meteo(
    # Model climate data for an ID, at a location/elevation/time
    #
    #   Assigns the constant lapse rate of -0.005 K/m from Girotto et al. (2014)
    #   Joins the metadata to the meteo data
    #   Calcs distance between obs and stations
    #   MODELS AIR TEMPERATURE
    #      Computes the IDW weights
    #      Estimates temp with IDW and constant/variable lapse rates
    #      Puts everything into a single-row data frame
    #   Does the same for MODEL WET BULB and DEW POINT TEMPERATURE and RELATIVE HUMIDITY
    #     Computes temp_dew when rh and temp_air exist

    id            = row_id,
    lon_obs       = lon_obs,
    lat_obs       = lat_obs,
    elevation     = elev_obs,
    datetime_utc  = t_obs,
    meteo_df      = sel_qc,
    meta_df       = meta_df,
    n_station_thresh = 5
  )

  modeled %>%
    mutate(lon = lon_obs, lat = lat_obs, datetime_utc = t_obs, elev_m = elev_obs)
}



# TESTING ON 20 rows ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
test_mros_rows <- mros_filter %>% dplyr::slice(20:40)

# run process_obs over the 10 rows
modeled_met_list <- lapply(seq_len(nrow(test_mros_rows)), function(i) {
  process_obs(test_mros_rows[i, ], stations_long, meta_df)
})
modeled_met <- dplyr::bind_rows(modeled_met_list)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# FULL DATASET RUN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# if (!requireNamespace("progress", quietly = TRUE)) install.packages("progress")
# library(progress)
# pb <- progress_bar$new(
#   total = nrow(mros_filter),
#   format = "  building per_obs [:bar] :percent eta: :eta",
#   clear = FALSE, width = 72
# )
# per_obs_list <- vector("list", nrow(mros_filter))
# for (i in seq_len(nrow(mros_filter))) {
#   per_obs_list[[i]] <- try(process_obs(mros_filter[i, ], stations_long, meta_df), silent = TRUE)
#   pb$tick()
# }
# arrow::write_parquet(per_obs_final,  "outputs/per_obs_WY2025_withIMERG.parquet")
# message("Phase A table written → ",  "outputs/per_obs_WY2025_withIMERG.parquet")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



## 4. IMERG (WIDE → LONG → HOURLY) ---------------------------------------------

# Pick one daily parquet to inspect
one_imerg <- list.files(imerg_dir, pattern = "\\.parquet$", full.names = TRUE)[1]
imerg_sample <- arrow::read_parquet(one_imerg) %>% tibble::as_tibble()
cat("\n--- ONE IMERG PARQUET (WIDE) ---\n")
glimpse(imerg_sample)
print(head(imerg_sample, 3))

# identify columns that look like dates or datetime strings
.is_time_col <- function(nm) {
  stringr::str_detect(nm, "^\\d{4}-\\d{2}-\\d{2}(?:\\s+\\d{2}:\\d{2}:\\d{2})?$")
}

# Convert a single wide parquet → long tidy: (time_utc, lat, lon, plp)
read_imerg_wide_to_long <- function(path) {
  df <- arrow::read_parquet(path) %>% tibble::as_tibble()

  # rename x/y → lon/lat
  if ("x" %in% names(df)) df <- dplyr::rename(df, lon = x)
  if ("y" %in% names(df)) df <- dplyr::rename(df, lat = y)

  # collect time-like columns
  time_cols <- names(df)[.is_time_col(names(df))]
  if (length(time_cols) == 0) {
    stop("No time-like columns detected in IMERG parquet: ", path)
  }

  long <- df %>%
    tidyr::pivot_longer(
      cols = dplyr::all_of(time_cols),
      names_to = "time_str",
      values_to = "plp_raw"
    ) %>%
    dplyr::mutate(
      time_utc = dplyr::if_else(
        stringr::str_detect(time_str, "^\\d{4}-\\d{2}-\\d{2}$"),
        paste0(time_str, " 00:00:00"),
        time_str
      ),
      time_utc = lubridate::ymd_hms(time_utc, tz = "UTC"),
      plp = as.numeric(plp_raw)
    ) %>%
    dplyr::select(time_utc, lat, lon, plp)

  # scale to [0,1] if 0..100
  if (max(long$plp, na.rm = TRUE) > 1) {
    long <- dplyr::mutate(long, plp = plp / 100)
  }
  long
}


# # TESTING ON 10 rows ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # Extract dates from filenames directly
imerg_dates <- str_extract(basename(imerg_files), "\\d{4}-\\d{2}-\\d{2}") %>%
  lubridate::as_date()

# Get overlap with test_mros_rows dates
test_dates <- unique(lubridate::as_date(test_mros_rows$datetime_utc))

overlap_dates <- as_date(intersect(test_dates, imerg_dates))
cat("Overlap dates:", paste(overlap_dates, collapse=", "), "\n")

# Keep only IMERG files with those dates
imerg_files_subset <- imerg_files[imerg_dates %in% overlap_dates]

cat("Selected", length(imerg_files_subset), "IMERG parquet(s) overlapping with test_mros_rows\n")
print(basename(imerg_files_subset))

# now read only those
imerg_long_list <- lapply(imerg_files_subset, read_imerg_wide_to_long)
imerg_long_all  <- dplyr::bind_rows(imerg_long_list)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# # FULL DATASET RUN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # Read ALL IMERG parquets using lapply, then bind
# imerg_files <- list.files(imerg_dir, pattern = "\\.parquet$", full.names = TRUE)
# imerg_long_list <- lapply(imerg_files, read_imerg_wide_to_long)
# imerg_long_all  <- dplyr::bind_rows(imerg_long_list)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# Half-hourly → hourly mean
imerg_all <- imerg_long_all %>%
  dplyr::mutate(hour_utc = lubridate::floor_date(time_utc, "hour")) %>%
  dplyr::group_by(hour_utc, lat, lon) %>%
  dplyr::summarise(plp = mean(plp, na.rm = TRUE), .groups = "drop")
glimpse(imerg_all)

# ATTACH IMERG TO per_obs_test
attach_imerg_nn <- function(per_obs_tbl, imerg_hourly_tbl) {
  per_obs_tbl <- per_obs_tbl %>%
    dplyr::mutate(hour_utc = lubridate::floor_date(datetime_utc, "hour"))

  # split by hour for efficient nearest-neighbor per hour
  obs_split   <- split(per_obs_tbl, per_obs_tbl$hour_utc)
  imerg_split <- split(imerg_hourly_tbl, imerg_hourly_tbl$hour_utc)

  out_list <- vector("list", length(obs_split))
  nm       <- names(obs_split)

  for (k in seq_along(obs_split)) {
    h   <- nm[k]
    obs <- obs_split[[k]]
    sl  <- imerg_split[[h]]

    if (is.null(sl) || nrow(sl) == 0) {
      obs$plp <- NA_real_
      out_list[[k]] <- obs
      next
    }

    # use coordinate names from per_obs_tbl
    obs_sf <- sf::st_as_sf(obs, coords = c("lon","lat"), crs = 4326)
    sl_sf  <- sf::st_as_sf(sl,  coords = c("lon","lat"), crs = 4326)
    nn_idx <- sf::st_nearest_feature(obs_sf, sl_sf)
    obs$plp <- sl$plp[nn_idx]

    out_list[[k]] <- obs
  }

  dplyr::bind_rows(out_list) %>%
    dplyr::arrange(datetime_utc)
}

# Run on per-obs test set
per_obs_with_imerg <- attach_imerg_nn(modeled_met, imerg_all) %>%
  relocate(lat, .after = id) %>%
  relocate(lon, .after = lat) %>%
  relocate(datetime_utc, .after = lon) %>%
  relocate(elev_m, .after = datetime_utc)
arrow::write_parquet(per_obs_with_imerg, "outputs/per_obs_TABLE_TEST_withIMERG.parquet")
message("Wrote per-obs TEST with IMERG → outputs/per_obs_TABLE_TEST_withIMERG.parquet")



# 4. CREATE GRID ---------------------------------------------------------------
# PHASE B — Gridded surfaces at a chosen hour
#   - Predictors (Tair/Twb/RH): IDW from per_obs (which already used station IDW+lapse)
#   - IMERG PLP: regrid hourly IMERG to DEM (no IDW)
#   - MRoS phase → labels (p_snow, p_mix) and IDW those


# Pull unique datetimes from test set
map_times <- sort(unique(per_obs_with_imerg$datetime_utc))

# To keep runtime manageable, you can slice first:
# map_times <- head(map_times, 5)

# Function to build grids for a single datetime
build_grids_for_time <- function(t0,
                                 per_obs_tbl,
                                 imerg_tbl,
                                 dem,
                                 out_dir,
                                 idp = 2) {

  pts_sel <- per_obs_tbl %>%
    filter(datetime_utc == t0)

  if (nrow(pts_sel) == 0) return(NULL)

  # set up DEM grid
  crs_dem <- crs(dem)
  grid_pts <- as.data.frame(terra::xyFromCell(dem, 1:terra::ncell(dem)))
  names(grid_pts) <- c("x","y")
  grid_sf <- st_as_sf(grid_pts, coords = c("x","y"), crs = crs_dem)

  # source pts
  src_sf <- st_as_sf(pts_sel, coords = c("lon","lat"), crs = 4326, remove = FALSE) %>%
    st_transform(crs_dem)

  # predictors (adjust col names if needed)
  src_sf$tair <- pts_sel$temp_air_idw_const %||% pts_sel$temp_air_idw %||% NA
  src_sf$twb  <- pts_sel$temp_wet_idw_const %||% pts_sel$temp_wet_idw %||% NA
  src_sf$rh   <- pts_sel$rh_idw_const %||% pts_sel$rh_idw %||% NA

  idw_to_raster <- function(src_sf, var, grid_sf, dem, idp = 2) {
    if (!var %in% names(src_sf)) return(NULL)
    src_ok <- src_sf[!is.na(src_sf[[var]]), ]
    if (nrow(src_ok) < 3) return(NULL)
    g <- gstat::gstat(formula = as.formula(paste0(var,"~1")),
                      data = src_ok, set = list(idp = idp))
    pred <- predict(g, newdata = grid_sf, debug.level = 0)$var1.pred
    r <- dem; values(r) <- as.numeric(pred); r
  }

  r_tair <- idw_to_raster(src_sf,"tair",grid_sf,dem,idp)
  r_twb  <- idw_to_raster(src_sf,"twb", grid_sf,dem,idp)
  r_rh   <- idw_to_raster(src_sf,"rh",  grid_sf,dem,idp)

  # IMERG slice
  t_hr <- lubridate::floor_date(t0,"hour")
  imerg_slice <- imerg_tbl %>% filter(hour_utc == t_hr)
  if (nrow(imerg_slice)==0) return(NULL)

  bbox <- st_bbox(st_as_sf(as.polygons(ext(dem), crs=crs_dem), crs=crs_dem) |> st_transform(4326))
  tmpl_ll <- terra::rast(xmin=bbox$xmin,xmax=bbox$xmax,ymin=bbox$ymin,ymax=bbox$ymax,
                         resolution=0.1,crs="EPSG:4326")
  pts_ll <- st_as_sf(imerg_slice,coords=c("lon","lat"),crs=4326)
  r_imerg_ll <- terra::rasterize(terra::vect(pts_ll), tmpl_ll,
                                 field=imerg_slice$plp, fun=mean)
  r_plp <- terra::project(r_imerg_ll, dem, method="near")

  # outputs
  stamp <- format(t0,"%Y%m%dT%H%MZ")
  if (!is.null(r_tair)) terra::writeRaster(r_tair, file.path(out_path,paste0("tair_",stamp,".tif")), overwrite=TRUE)
  if (!is.null(r_twb))  terra::writeRaster(r_twb,  file.path(out_path,paste0("twb_", stamp,".tif")), overwrite=TRUE)
  if (!is.null(r_rh))   terra::writeRaster(r_rh,   file.path(out_path,paste0("rh_",  stamp,".tif")), overwrite=TRUE)
  terra::writeRaster(r_plp, file.path(out_path,paste0("imergPLP_",stamp,".tif")), overwrite=TRUE)

  invisible(list(t=t0, tair=r_tair, twb=r_twb, rh=r_rh, plp=r_plp))
}

# Run across datetimes
dem <- terra::rast(dem_path)
results <- lapply(map_times, build_grids_for_time,
                  per_obs_tbl=per_obs_with_imerg,
                  imerg_tbl=imerg_all,
                  dem=dem,
                  out_dir=out_path)
