# IDW Spatial Interpolation

### Part A: Apply rainOrSnowTools to create tabular IDW'd predictors (lat, lon, datetime_utc, [predictors])

### Part B: Make spatially interpolated surfaces on DEM grid

### Part C: Add other gridded data (TBD)
# MERRA-2
# NLDAS-2



## SET-UP ----------------------------------------------------------------------

# Install/load packages & the local repo
packages <- c("devtools", "tidyverse", "lubridate", "arrow", "sf", "terra", "gt", "rstudioapi", "gstat", "progress",
              "FNN", "data.table", "patchwork", "ggplot2")

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


## LOAD DATA -------------------------------------------------------------------
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


# Read & prep station data
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

  # Ensure format like 2024-10-01T00:12:00Z)
  if ("datetime" %in% names(df)) {
    df$datetime <- suppressWarnings(lubridate::ymd_hms(df$datetime, tz = "UTC"))
  }
    for (nm in c("temp_air","temp_wet","temp_dew","rh")) {
    if (!nm %in% names(df)) df[[nm]] <- NA_real_
  }
  df <- df %>%
    select(id, datetime, temp_air, temp_wet, temp_dew, rh)
  return(df)
})

stations_long <- bind_rows(stations_list) %>%
  arrange(id, datetime)


# ==============================================================================
## PART A - DATA ASSIMILATIONS
# ==============================================================================
# Inputs: all tabular
#  - MRoS: lon, lat, datetime_utc, phase
#  - Stations: raw time series + metadata (lat/lon, elevation, tz)
#  - IMERG: PLP (half-hourly)
# Time syncing
#  - Convert station timestamps to UTC using each station’s timezone
#  - Select station readings closest in time within a ±1h window
#  - Sample IMERG at the obs time/point (average to the hour)
# Spatialization to the obs point
#  - Use the same IDW (+ lapse) routine as in the MRoS tools to downscale station predictors (Tair, Td, Twb, RH) to the MRoS point
#  - (carry both DEM elevation and station meta-elevation for QA)
#  - Attach IMERG PLP to each row
# Output
#  - Modeled_meteo_with_imerg = assimilated df with predictors (station-derived + IMERG + elevation) and target (MRoS phase)


## BUILD PER-OBS PREDICTORS ----------------------------------------------------

# select_meteo() returns the nearest-in-time per station/variable, but it does not enforce a max time gap window.
max_gap <- lubridate::hours(1)


# ------------------------------------------------------------------------------
# process_obs()
# Purpose:
#   Build station-derived predictors at one MRoS observation using
#   MRoS tooling (select_meteo -> qc_meteo -> model_meteo).
#
# Inputs:
#   obs_row        data.frame (single row) with columns:
#                  longitude, latitude, datetime_utc, elev_m, row_id
#   stations_long  data.frame with columns:
#                  id, datetime (POSIXct, UTC), temp_air, temp_wet, temp_dew, rh
#   meta_df        data.frame with station metadata: id, lat, lon, elev
#
# Returns:
#   tibble (1 row) of modeled predictors with columns produced by model_meteo(),
#   plus lon, lat, datetime_utc, and elev_m carried through.
#   (NULL if select_meteo returns no usable data)
# ------------------------------------------------------------------------------
process_obs <- function(obs_row, stations_long, meta_df) {
  lon_obs  <- obs_row$longitude
  lat_obs  <- obs_row$latitude
  t_obs    <- obs_row$datetime_utc
  elev_obs <- obs_row$elev_m
  row_id   <- obs_row$row_id

  # Nearest-in-time per station/variable
  sel <- rainOrSnowTools:::select_meteo(df = stations_long, datetime_obs = t_obs)

  # # enforce ±1h window
  # sel <- sel %>%
  #   left_join(
  #     stations_long %>%
  #       group_by(id) %>%
  #       slice_min(abs(datetime - t_obs), n = 1, with_ties = FALSE) %>%
  #       ungroup() %>%
  #       mutate(time_gap = abs(datetime - t_obs)) %>%
  #       select(id, time_gap),
  #     by = "id"
  #   )
  #
  # # Only filter if time_gap exists
  # if ("time_gap" %in% names(sel)) {
  #   sel <- sel %>%
  #     filter(is.na(time_gap) | time_gap <= max_gap) %>%
  #     select(-time_gap)
  # }

  # QC
  sel_qc <- rainOrSnowTools:::qc_meteo(sel)

  # Aplly MRoS model_meteo()
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
test_mros_rows <- mros_filter %>% dplyr::slice(20:80)
pb <- progress_bar$new(
  total = nrow(test_mros_rows),
  format = "  building modeled_met [:bar] :percent eta: :eta",
  clear = FALSE, width = 72
)
modeled_met_list <- lapply(seq_len(nrow(test_mros_rows)), function(i) {
  res <- try(process_obs(test_mros_rows[i, ], stations_long, meta_df), silent = TRUE)
  pb$tick()
  res
})
modeled_met <- dplyr::bind_rows(Filter(Negate(is.null), modeled_met_list))
arrow::write_parquet(modeled_met,  "outputs/modeled_met_TEST.parquet")

# add phase back
phase_key <- mros_filter %>%
  select(row_id, phase) %>%
  mutate(phase = tolower(phase)) %>%
  mutate(mros_plp_proxy = case_when(
    phase == "snow" ~ 0,
    phase == "mix"  ~ 50,
    phase == "rain" ~ 100,
    TRUE            ~ NA_real_
  ))
modeled_met <- modeled_met %>%
  left_join(phase_key %>% rename(id = row_id), by = "id")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# FULL DATASET RUN (~ 3 days to run) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# pb <- progress_bar$new(
#   total = nrow(mros_filter),
#   format = "  building modeled_met [:bar] :percent eta: :eta",
#   clear = FALSE, width = 72
# )
# modeled_met_list <- lapply(seq_len(nrow(mros_filter)), function(i) {
#   res <- try(process_obs(mros_filter[i, ], stations_long, meta_df), silent = TRUE)
#   pb$tick()
#   res
# })
# modeled_met <- dplyr::bind_rows(Filter(Negate(is.null), modeled_met_list))
# # add phase back
# phase_key <- mros_filter %>%
#   select(row_id, phase) %>%
#   mutate(phase = tolower(phase)) %>%
#   mutate(mros_plp_proxy = case_when(
#     phase == "snow" ~ 0,
#     phase == "mix"  ~ 50,
#     phase == "rain" ~ 100,
#     TRUE            ~ NA_real_
#   ))
# modeled_met <- modeled_met %>%
#   left_join(phase_key %>% rename(id = row_id), by = "id")
# arrow::write_parquet(modeled_met,  "outputs/modeled_met_WY2025.parquet")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



## IMERG (WIDE → LONG → HOURLY) ------------------------------------------------

# Inspect one of the parquets
one_imerg <- list.files(imerg_dir, pattern = "\\.parquet$", full.names = TRUE)[1]
imerg_sample <- arrow::read_parquet(one_imerg) %>% tibble::as_tibble()
cat("\n--- ONE IMERG PARQUET (WIDE) ---\n")
glimpse(imerg_sample)
print(head(imerg_sample, 3))

# Identify columns that look like dates or datetime strings
.is_time_col <- function(nm) {
  stringr::str_detect(nm, "^\\d{4}-\\d{2}-\\d{2}(?:\\s+\\d{2}:\\d{2}:\\d{2})?$")
}


# ------------------------------------------------------------------------------
# read_imerg_wide_to_long()
# Purpose:
#   Convert an IMERG parquet (wide by timestamps) to tidy long (time_utc, lat, lon, plp).
#
# Inputs:
#   path   character, path to a single parquet file with columns x/y or lon/lat,
#          and many columns that are ISO dates or datetimes.
#
# Returns:
#   tibble with columns: time_utc (POSIXct, UTC), lat, lon, plp (numeric, 0..100)
#   Note: keeps the original PLP scale
# ------------------------------------------------------------------------------
read_imerg_wide_to_long <- function(path) {
  df <- arrow::read_parquet(path) %>% tibble::as_tibble()

  # Rename x/y → lon/lat
  if ("x" %in% names(df)) df <- dplyr::rename(df, lon = x)
  if ("y" %in% names(df)) df <- dplyr::rename(df, lat = y)

  # Collect time-like columns
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

  # Make sure PLP is 0..100
  if (max(long$plp, na.rm = TRUE) <= 1) {
    long <- dplyr::mutate(long, plp = 100 * plp)
  }
  long
}


# # TESTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# # Extract dates from filenames directly
imerg_files <- list.files(imerg_dir, pattern = "\\.parquet$", full.names = TRUE)
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

# Now read only those
imerg_long_list <- lapply(imerg_files_subset, read_imerg_wide_to_long)
imerg_long_all  <- dplyr::bind_rows(imerg_long_list)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# # FULL DATASET RUN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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


# ------------------------------------------------------------------------------
# attach_imerg_nn()
# Purpose:
#   For each modeled_met row, attach the nearest IMERG PLP at the same hour
#   using nearest-neighbor in space.
#
# Inputs:
#   modeled_met_tbl   tibble with lon, lat, datetime_utc (POSIXct)
#   imerg_hourly_tbl  tibble with hour_utc (POSIXct), lat, lon, plp (0..100)
#
# Returns:
#   same rows as modeled_met_tbl with an added numeric column `plp` (0..100)
# ------------------------------------------------------------------------------
attach_imerg_nn <- function(modeled_met_tbl, imerg_hourly_tbl) {
  modeled_met_tbl <- modeled_met_tbl %>%
    dplyr::mutate(hour_utc = lubridate::floor_date(datetime_utc, "hour"))

  # Split by hour for nearest-neighbor per hour
  obs_split   <- split(modeled_met_tbl, modeled_met_tbl$hour_utc)
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

    # Use coordinate names from modeled_met_tbl
    obs_sf <- sf::st_as_sf(obs, coords = c("lon","lat"), crs = 4326)
    sl_sf  <- sf::st_as_sf(sl,  coords = c("lon","lat"), crs = 4326)
    nn_idx <- sf::st_nearest_feature(obs_sf, sl_sf)
    obs$plp <- sl$plp[nn_idx]   # 0..100

    out_list[[k]] <- obs
  }

  dplyr::bind_rows(out_list) %>%
    dplyr::arrange(datetime_utc)
}

# Run modeled_met
modeled_met_with_imerg <- attach_imerg_nn(modeled_met, imerg_all) %>%
  relocate(lat, .after = id) %>%
  relocate(lon, .after = lat) %>%
  relocate(datetime_utc, .after = lon) %>%
  relocate(elev_m, .after = datetime_utc)
arrow::write_parquet(modeled_met_with_imerg, "outputs/modeled_met_TEST_withIMERG.parquet")
# arrow::write_parquet(modeled_met_with_imerg, "outputs/modeled_met_WY2025_withIMERG.parquet")





# ==============================================================================
## PHASE B — CREATING SURFACES
# ==============================================================================
#   - Inspect DEM, re-project to meters (EPSG:3310)
#   - Build a 1-km grid (aligned to DEM)
#   - Clip all data points to DEM AOI (Sierra)
#   - Perform IDW on tabular predictors to the 1 km grid
#   - Output one GeoTIFF per var per hour
#   - Visualizing


# Parameters -------------------------------------------------------------------
coarse_res_m <- 1000          # 1 km output grid
idw_power    <- 2             # IDW power
min_points   <- 2             # at least this many points to run IDW
time_tol_min <- 30            # use points within ± this many minutes of each hour
out_dir_tif  <- "./outputs/phaseB_tifs"
dir.create(out_dir_tif, recursive = TRUE, showWarnings = FALSE)

# Predictors to IDW from points
predictors_to_grid <- c(
  "plp",
  "temp_wet", "rh", "temp_dew_avg_obs", "temp_air_avg_obs",
  "mros_plp_proxy", "temp_air_idw_lapse_const", "temp_dew_idw_lapse_const"
)


# DEM set up -------------------------------------------------------------------
dem <- terra::rast(dem_path)

cat("\nDEM info\n--------\n")
print(dem)
cat("CRS (WKT):\n", terra::crs(dem), "\n")
cat("Is lon/lat? ", terra::is.lonlat(dem), "\n")
cat("Resolution: ", paste(terra::res(dem), collapse=", "), "\n")
cat("Extent:     ", paste(terra::ext(dem)), "\n\n")

# Aggregate to ~1 km (10 m * 100 = 1000 m)
#   - reduces raster size ~100×
#   - much faster to reproject
dem_1km_native <- terra::aggregate(dem, fact = 100, fun = mean, na.rm = TRUE)

# Reproject the coarsened DEM to match modeled grid
target_crs <- terra::crs(mros_sf)
dem_1km_proj <- terra::project(dem_1km_native, target_crs, method = "bilinear")
terra::writeRaster(dem_1km_proj, file.path(out_path, "DEM_1km_projected.tif"), overwrite = TRUE)

# Use this DEM for elevation extraction
dem <- dem_1km_proj

if (terra::is.lonlat(dem)) {
  # Work in meters for 1-km gridding
  dem_m <- terra::project(dem, "EPSG:3310")    # (NAD83 / California Albers) for Sierras
  cat("Reprojected DEM to EPSG:3310 (meters).\n")
} else {
  dem_m <- dem
  cat("DEM already in projected (metric) CRS.\n")
}


# Build coarse grid ------------------------------------------------------------
coarse <- terra::rast(terra::ext(dem_m), crs = terra::crs(dem_m), resolution = coarse_res_m)
elev_coarse <- terra::resample(dem_m, coarse, method = "average")
names(elev_coarse) <- "elev_coarse"

# Create grid centers as sf (these are the IDW targets)
grid_xy <- as.data.frame(terra::xyFromCell(elev_coarse, 1:terra::ncell(elev_coarse)))
names(grid_xy) <- c("x", "y")
grid_sf <- sf::st_as_sf(grid_xy, coords = c("x","y"), crs = terra::crs(elev_coarse))


# Filter points to DEM AOI (Sierra only) ---------------------------------------
stopifnot(all(c("lon","lat","datetime_utc") %in% names(modeled_met_with_imerg)))

aoi_m <- sf::st_as_sf(terra::as.polygons(terra::ext(dem_m), crs = terra::crs(dem_m)))
pts_ll <- sf::st_as_sf(modeled_met_with_imerg, coords = c("lon","lat"), crs = 4326, remove = FALSE)
pts_m  <- sf::st_transform(pts_ll, sf::st_crs(aoi_m))

inside <- as.logical(sf::st_within(pts_m, aoi_m, sparse = FALSE)[,1])
pts_m_sierra <- pts_m[inside, ]
cat("Points kept in Sierra AOI: ", nrow(pts_m_sierra), " / ", nrow(pts_m), "\n")


# ------------------------------------------------------------------------------
# idw_to_grid()
# Purpose:
#   Interpolate a point variable to the target 1-km grid via IDW.
#
# Inputs:
#   src_sf     sf point data already in the grid CRS; must contain `value_col`
#   value_col  character, name of the numeric column to interpolate
#   grid_sf    sf points = centers of grid cells (prediction locations)
#   template   SpatRaster (single-layer) whose cells align with `grid_sf`
#   idp        numeric, inverse distance power
#   nmin       integer, minimum number of valid points required
#
# Returns:
#   SpatRaster (single layer) with predictions for `value_col`, aligned to `template`,
#   or NULL if not enough points / variable missing.
# ------------------------------------------------------------------------------
idw_to_grid <- function(src_sf, value_col, grid_sf, template, idp = 2, nmin = 3) {
  if (!value_col %in% names(src_sf)) return(NULL)
  src_ok <- src_sf[is.finite(src_sf[[value_col]]), ]
  if (nrow(src_ok) < nmin) return(NULL)

  # src and grid are both in dem_m CRS already
  g <- gstat::gstat(formula = as.formula(paste0(value_col, "~ 1")),
                    data = src_ok, set = list(idp = idp))
  pred <- predict(g, newdata = grid_sf, debug.level = 0)$var1.pred
  r <- template
  terra::values(r) <- as.numeric(pred)
  r
}


# ------------------------------------------------------------------------------
# build_for_time()
# Purpose:
#   Build a named list of SpatRasters (one per predictor) for a given timestamp,
#   using only points within ± tol_min minutes of that time. Includes
#   `elev_coarse` for later raster stacking.
#
# Inputs:
#   t0          POSIXct UTC, target time
#   predictors  character[], column names in pts_m_sierra to IDW
#   tol_min     numeric, ± minutes window for filtering points
#   nmin        integer, minimum number of points per variable
#   idp         numeric, IDW power
#
# Returns:
#   named list of SpatRaster, names = predictors + 'elev_coarse'; NULL if too few points
# ------------------------------------------------------------------------------
build_for_time <- function(t0,
                           predictors = predictors_to_grid,
                           tol_min = time_tol_min,
                           nmin = min_points,
                           idp = idw_power) {
  # choose points within ± tol_min minutes of t0
  keep <- abs(as.numeric(difftime(pts_m_sierra$datetime_utc, t0, units = "mins"))) <= tol_min
  src_t <- pts_m_sierra[keep, ]
  if (nrow(src_t) < min_points) {
    message(sprintf("[build_for_time] %s -> only %d pts within +/- %d min; skipping",
                    format(as.POSIXct(t0, tz="UTC"), "%Y-%m-%d %H:%MZ"),
                    nrow(src_t), time_tol_min))
    return(NULL)
  }

  out <- vector("list", length(predictors) + 1)
  names(out) <- c(predictors, "elev_coarse")

  for (v in predictors) {
    out[[v]] <- idw_to_grid(src_t, v, grid_sf, elev_coarse, idp = idp, nmin = nmin)
  }

  # Always include elevation for reference/stack building
  out$elev_coarse <- elev_coarse
  out
}


# ------------------------------------------------------------------------------
# build_stacks_for_times()
# Purpose:
#   Build time stacks (SpatRaster) per predictor across a vector of times.
#
# Inputs:
#   times       POSIXct[] UTC times to include (order used for the stack)
#   predictors  character[] predictors to include
#   tol_min     numeric ± minutes time window
#   nmin, idp   IDW params
#
# Returns:
#   named list: each item is a SpatRaster with nlayers == length(times_kept).
#   Each layer has the same geometry as `elev_coarse`. terra::time() is set.
# ------------------------------------------------------------------------------
build_stacks_for_times <- function(
    times,
    predictors = predictors_to_grid,
    tol_min    = time_tol_min,
    nmin       = min_points,
    idp        = idw_power
) {
  # containers
  stacks   <- setNames(vector("list", length(predictors)), predictors)
  hours_by <- setNames(vector("list", length(predictors)), predictors)

  # progress bar setup
  pb <- progress::progress_bar$new(
    format = "  Building stacks [:bar] :percent in :elapsed eta: :eta",
    total  = length(times),
    clear  = FALSE, width = 60
  )

  t_start <- Sys.time()

  for (t0 in times) {
    maps <- build_for_time(t0, predictors = predictors, tol_min = tol_min, nmin = nmin, idp = idp)
    if (is.null(maps)) next

    for (v in predictors) {
      r_v <- maps[[v]]
      if (is.null(r_v)) next

      if (is.null(stacks[[v]])) {
        stacks[[v]]   <- r_v
        hours_by[[v]] <- as.POSIXct(t0, tz = "UTC")
      } else {
        stacks[[v]]   <- c(stacks[[v]], r_v)
        hours_by[[v]] <- c(hours_by[[v]], as.POSIXct(t0, tz = "UTC"))
      }
    }
    pb$tick()
  }

  # attach time vectors and layer names
  for (v in predictors) {
    if (!is.null(stacks[[v]])) {
      terra::time(stacks[[v]]) <- hours_by[[v]]
      names(stacks[[v]]) <- format(hours_by[[v]], "%Y-%m-%d %H:%M:%S")
    }
  }

  stacks
}


# Implement --------------------------------------------------------------------
map_times <- sort(unique(pts_m_sierra$datetime_utc))

# Build stacks for all unique times, per predictor
stacks <- build_stacks_for_times(
  times       = map_times,
  predictors  = predictors_to_grid,
  tol_min     = time_tol_min,
  nmin        = min_points,
  idp         = idw_power
)

# Write one NetCDF per predictor (full time dimension)
out_dir_nc <- file.path(out_path, "phaseB_nc_stacks")
dir.create(out_dir_nc, recursive = TRUE, showWarnings = FALSE)

for (v in names(stacks)) {
  r <- stacks[[v]]
  if (is.null(r)) next
  unit_v <- dplyr::case_when(
    grepl("^temp", v) ~ "°C",                               # check units of original data
    v %in% c("rh") ~ "percent",
    v %in% c("plp","mros_plp_proxy") ~ "percent",
    TRUE ~ "1"
  )
  fn <- file.path(out_dir_nc, sprintf("%s_stack.nc", v))
  terra::writeCDF(
    r, filename = fn,
    varname = v, unit = unit_v, longname = v,
    overwrite = TRUE, compression = 6, shuffle = FALSE, zname = "time"
  )
}


# # ------------------------------------------------------------------------------
# # write_maps_for_time()
# # Purpose:
# #   Write each raster in `maps` to a GeoTIFF file named <var>_<YYYYmmddTHHMMZ>.tif
# #
# # Inputs:
# #   t0       POSIXct UTC, timestamp tag for filenames
# #   maps     named list of SpatRaster (as from build_for_time)
# #   out_dir  output directory
# #
# # Returns:
# #   logical TRUE if at least one file was written, FALSE otherwise
# # ------------------------------------------------------------------------------
# write_maps_for_time <- function(t0, maps, out_dir = out_dir_tif) {
#   if (is.null(maps)) return(FALSE)
#   stamp <- format(as.POSIXct(t0, tz="UTC"), "%Y%m%dT%H%MZ")
#   wrote_any <- FALSE
#   for (nm in names(maps)) {
#     r <- maps[[nm]]
#     if (is.null(r)) next
#     fn <- file.path(out_dir, sprintf("%s_%s.tif", nm, stamp))
#     terra::writeRaster(r, fn, overwrite = TRUE)
#     wrote_any <- TRUE
#   }
#   wrote_any
# }
#
#
# # Implement --------------------------------------------------------------------
# map_times <- sort(unique(pts_m_sierra$datetime_utc))
#
# cat("\nBuilding a few hours to verify…\n")
# for (t0 in head(map_times, 5)) {
#   maps <- build_for_time(t0)
#   ok <- isTRUE(write_maps_for_time(t0, maps))  # coerce to TRUE/FALSE
#   message(sprintf("%s %s",
#                   format(as.POSIXct(t0, tz="UTC"), "%Y-%m-%d %H:%MZ"),
#                   if (ok) "-> wrote maps" else "-> skipped (not enough points)"))
# }
#
#
# # # Plot for one time ----------------------------------------------------------
# # quicklook_plot <- function(t0, vars = c("plp","temp_wet","rh")) {
# #   maps <- build_for_time(t0)
# #   if (is.null(maps)) { message("No maps for this time"); return(invisible(NULL)) }
# #
# #   keep <- intersect(vars, names(maps))
# #   if (length(keep) == 0) { message("None of the requested vars available"); return(invisible(NULL)) }
# #
# #   rlist <- lapply(keep, function(v) { names(maps[[v]]) <- v; maps[[v]] })
# #   r <- do.call(c, rlist)  # stack the selected layers
# #   plot(r, main = paste0("Surfaces @ ", format(t0, "%Y-%m-%d %H:%MZ")))
# # }
# # quicklook_plot(head(map_times, 4))



# ------------------------------------------------------------------------------
# .r_to_df()
# Purpose:
#   Convert a single-layer SpatRaster to a tidy data.frame for ggplot.
#
# Inputs:
#   r         SpatRaster (single layer)
#   var_name  character, label for the panel/fill legend
#
# Returns:
#   data.frame with columns: x, y, value, var
# ------------------------------------------------------------------------------
.r_to_df <- function(r, var_name) {
  if (is.null(r)) return(NULL)
  names(r) <- var_name
  df <- as.data.frame(r, xy = TRUE, na.rm = FALSE)
  valcol <- setdiff(names(df), c("x","y"))
  if (length(valcol) == 0) return(NULL)
  df <- df[, c("x","y", valcol[1])]
  names(df) <- c("x", "y", "value")
  df$var <- var_name
  df
}


# ------------------------------------------------------------------------------
# plotting_IDW_interpolation_many()
# Purpose:
#   For each time in `times`, build IDW maps and return a list of patchwork
#   plot-grids. Each predictor has its own plot (→ its own fill scale).
#
# Inputs:
#   times        POSIXct[] UTC times to plot
#   vars         predictors to include (must exist in `maps`)
#   tol_min      ± minutes window around each time to include points
#   nmin         minimum points per variable to run IDW
#   ncol_panels  number of columns in the grid for predictors
#   base_size    ggplot theme text size
#
# Returns:
#   named list of patchwork objects (one grid per time). Print with `plots[[i]]`.
# ------------------------------------------------------------------------------
plotting_IDW_interpolation_many <- function(
    times,
    vars = c(
      "plp",
      "mros_plp_proxy",
      "rh",
      "temp_wet",
      "temp_air_avg_obs",
      "temp_dew_avg_obs",
      "temp_air_idw_lapse_const",
      "temp_dew_idw_lapse_const"
    ),
    tol_min     = time_tol_min,
    nmin        = min_points,
    ncol_panels = 3,     # fewer columns = bigger panels
    base_size   = 10     # text size
) {
  if (!"package:patchwork" %in% search()) suppressPackageStartupMessages(library(patchwork))

  out <- list()

  # Stations once (project to DEM/grid CRS and clip to AOI)
  stations_sf <- sf::st_as_sf(meta_df, coords = c("lon","lat"), crs = 4326, remove = FALSE) |>
    sf::st_transform(sf::st_crs(aoi_m))
  in_aoi <- as.logical(sf::st_within(stations_sf, aoi_m, sparse = FALSE)[,1])
  stations_sf <- stations_sf[in_aoi, ]
  st_xy <- sf::st_coordinates(stations_sf)
  stations_df <- data.frame(x = st_xy[,1], y = st_xy[,2], type = "Station")

  # helper: SpatRaster (1 layer) -> df
  .r_to_df <- function(r, var_name) {
    if (is.null(r)) return(NULL)
    names(r) <- var_name
    df <- as.data.frame(r, xy = TRUE, na.rm = FALSE)
    valcol <- setdiff(names(df), c("x","y"))
    if (length(valcol) == 0) return(NULL)
    df <- df[, c("x","y", valcol[1])]
    names(df) <- c("x", "y", "value")
    df$var <- var_name
    df
  }

  # build one grid (patchwork) for a single time
  make_plot_for_time <- function(t0) {
    maps <- build_for_time(
      t0, predictors = predictors_to_grid,
      tol_min = tol_min, nmin = nmin, idp = idw_power
    )
    if (is.null(maps)) {
      message(sprintf("No maps for %s (not enough points).",
                      format(as.POSIXct(t0, tz="UTC"), "%Y-%m-%d %H:%MZ")))
      return(NULL)
    }

    keep <- intersect(vars, names(maps))
    if (length(keep) == 0) return(NULL)

    # MRoS obs within tolerance (already in grid CRS)
    keep_obs <- abs(as.numeric(difftime(pts_m_sierra$datetime_utc, t0, units = "mins"))) <= tol_min
    obs_sf   <- pts_m_sierra[keep_obs, ]
    ob_xy    <- sf::st_coordinates(obs_sf)
    obs_df   <- data.frame(x = ob_xy[,1], y = ob_xy[,2], type = "MRoS Obs")

    pts_df <- rbind(stations_df, obs_df)

    # build one panel per predictor (independent fill scales)
    panels <- vector("list", length(keep))
    for (i in seq_along(keep)) {
      v     <- keep[i]
      df_v  <- .r_to_df(maps[[v]], v)
      if (is.null(df_v)) next

      show_legend <- (i == 1)  # show the points legend only once

      p <- ggplot2::ggplot(df_v, ggplot2::aes(x, y)) +
        ggplot2::geom_raster(ggplot2::aes(fill = value)) +
        ggplot2::coord_equal() +
        ggplot2::scale_fill_viridis_c(na.value = NA) +  # independent per plot
        ggplot2::labs(
          title    = paste0(v, " @ ", format(as.POSIXct(t0, tz = "UTC"), "%Y-%m-%d %H:%MZ")),
          subtitle = sprintf("n_stations=%d   n_obs=%d   tol=±%d min",
                             nrow(stations_df), nrow(obs_df), tol_min),
          x = NULL, y = NULL, fill = v
        ) +
        ggplot2::geom_point(
          data = pts_df,
          ggplot2::aes(x, y, shape = type, color = type),
          size = 1.6, stroke = 0.4, inherit.aes = FALSE,
          show.legend = show_legend
        ) +
        ggplot2::scale_shape_manual(
          values = c("Station" = 21, "MRoS Obs" = 24),
          name   = "Points",
          guide  = if (show_legend) ggplot2::guide_legend(override.aes = list(fill = c("white", "white"))) else "none"
        ) +
        ggplot2::scale_color_manual(
          values = c("Station" = "lightblue", "MRoS Obs" = "red"),
          name   = "Points",
          guide  = if (show_legend) "legend" else "none"
        ) +
        ggplot2::theme_minimal(base_size = base_size) +
        ggplot2::theme(
          panel.grid  = ggplot2::element_blank(),
          plot.title  = ggplot2::element_text(face = "bold"),
          strip.text  = ggplot2::element_text(face = "bold"),
          legend.box  = "vertical",
          legend.key  = ggplot2::element_rect(fill = NA, color = NA),
          plot.margin = grid::unit(rep(4,4), "pt")
        )

      panels[[i]] <- p
    }

    panels <- Filter(Negate(is.null), panels)
    if (!length(panels)) return(NULL)

    patchwork::wrap_plots(panels, ncol = ncol_panels)
  }

  for (t0 in times) {
    grid_plot <- make_plot_for_time(t0)
    if (!is.null(grid_plot)) {
      out[[format(as.POSIXct(t0, tz="UTC"), "%Y-%m-%d %H:%MZ")]] <- grid_plot
    }
  }

  out
}

# Example:
plots <- plotting_IDW_interpolation_many(map_times, ncol_panels = 3, base_size = 10)
plots[[1]]

# Save each plot with its timestamp in the filename
for (nm in names(plots)) {
  fn <- sprintf("outputs/preview_%s.png", gsub(":", "-", nm))
  ggplot2::ggsave(fn, plots[[nm]], width = 14, height = 12, dpi = 200)
  message("Saved ", fn)
}
