################################################################################
# IDW Spatial Interpolation

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
    df$datetime <- suppressWarnings(lubridate::ymd_hms(df$datetime, tz = "UTC"))
  }

  # Add any missing columns with NA
  for (nm in c("temp_air","temp_wet","temp_dew","rh")) {
    if (!nm %in% names(df)) df[[nm]] <- NA_real_
  }

  df <- df %>%
    select(id, datetime, temp_air, temp_wet, temp_dew, rh)

  return(df)

})

stations_long <- bind_rows(stations_list) %>%
  arrange(id, datetime)


# ===============================================================================
## PART A - DATA ASSIMILATIONS
# ===============================================================================

## BUILD PER-OBS PREDICTORS ----------------------------------------------------

# select_meteo() returns the NEAREST-in-time per station/variable, but it does not enforce a max time gap window.
# To enforce ±1h, add a filter on time_gap afterwards.
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
    phase == "mix"  ~ 0.5,
    phase == "rain" ~ 1,
    TRUE            ~ NA_real_
  ))
modeled_met <- modeled_met %>%
  left_join(phase_key %>% rename(id = row_id), by = "id")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# FULL DATASET RUN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
#     phase == "mix"  ~ 0.5,
#     phase == "rain" ~ 1,
#     TRUE            ~ NA_real_
#   ))
# modeled_met <- modeled_met %>%
#   left_join(phase_key %>% rename(id = row_id), by = "id")
# arrow::write_parquet(modeled_met,  "outputs/modeled_met_WY2025.parquet")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



## IMERG (WIDE → LONG → HOURLY) ------------------------------------------------

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

# ATTACH IMERG TO modeled_met
attach_imerg_nn <- function(modeled_met_tbl, imerg_hourly_tbl) {
  modeled_met_tbl <- modeled_met_tbl %>%
    dplyr::mutate(hour_utc = lubridate::floor_date(datetime_utc, "hour"))

  # split by hour for efficient nearest-neighbor per hour
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

    # use coordinate names from modeled_met_tbl
    obs_sf <- sf::st_as_sf(obs, coords = c("lon","lat"), crs = 4326)
    sl_sf  <- sf::st_as_sf(sl,  coords = c("lon","lat"), crs = 4326)
    nn_idx <- sf::st_nearest_feature(obs_sf, sl_sf)
    obs$plp <- sl$plp[nn_idx]

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





# ===============================================================================
## PHASE B — CREATING SURFACES
# ===============================================================================

# Parameters
coarse_res_m   <- 1000     # coarser grid resolution=
idp            <- 2        # IDW power
t_start <- min(modeled_met_with_imerg$datetime_utc, na.rm = TRUE)
t_end   <- max(modeled_met_with_imerg$datetime_utc, na.rm = TRUE)
out_dir_nc     <- "./outputs/nc_coarse"   # NetCDF output dir
dir.create(out_dir_nc, recursive = TRUE, showWarnings = FALSE)

predictors_to_grid <- c(
  "plp",
  "temp_wet",
  "rh",
  "temp_dew_avg_obs",
  "temp_air_avg_obs",
  "mros_plp_proxy",
  "temp_air_idw_lapse_const",
  "temp_dew_idw_lapse_const"
)


# COARSE GRID ALIGNED TO DEM --------------------------
dem <- terra::rast(dem_path)
dem_crs <- crs(dem)
dem_res <- terra::res(dem)[1]
fact    <- max(1, round(coarse_res_m / dem_res))
coarse  <- terra::aggregate(dem, fact = fact, fun = mean)  # 1 km grid, aligned to DEM
names(coarse) <- "elev_coarse"
grid_xy <- terra::crds(coarse, df = TRUE)                  # coarse cell centers
grid_sf <- st_as_sf(grid_xy, coords = c("x","y"), crs = dem_crs)

# IDW HELPER (points → coarse grid) -------------------
idw_to_raster <- function(src_sf, var, grid_sf, template_raster, idp = 2) {
  if (!var %in% names(src_sf)) return(NULL)
  src_ok <- src_sf[!is.na(src_sf[[var]]), ]
  if (nrow(src_ok) < 3) return(NULL)
  g <- gstat::gstat(formula = as.formula(paste0(var, "~ 1")),
                    data = src_ok, set = list(idp = idp))
  pred <- predict(g, newdata = grid_sf, debug.level = 0)$var1.pred
  r <- template_raster
  values(r) <- as.numeric(pred)
  r
}

# PER-HOUR GRIDDING FUNCTION --------------------------
build_grids_for_hour <- function(h, pts_tbl, coarse, grid_sf, idp = 2,
                                 predictors = predictors_to_grid) {
  pts_sel <- pts_tbl %>%
    filter(floor_date(datetime_utc, "hour") == h)
  if (nrow(pts_sel) == 0) return(NULL)

  src_sf <- st_as_sf(pts_sel, coords = c("lon","lat"), crs = 4326, remove = FALSE) %>%
    st_transform(crs = crs(coarse))

  out <- list()
  for (v in predictors) {
    if (!v %in% names(src_sf)) next
    r_v <- idw_to_raster(src_sf, v, grid_sf, coarse, idp = idp)
    out[[v]] <- r_v
  }
  out$elev_coarse <- coarse
  out
}

# DAILY NETCDF WRITER ------------------------
write_daily_nc <- function(day, daily_layers, hours_by_var, out_dir) {
  day <- as.Date(day)

  for (v in names(daily_layers)) {
    layers <- daily_layers[[v]]
    if (is.null(layers)) next

    # Ensure SpatRaster stack
    r_stack <- if (inherits(layers, "SpatRaster")) layers else do.call(c, layers)

    # Use the exact hours recorded for this variable
    hours_vec <- as.POSIXct(hours_by_var[[v]], tz = "UTC")
    if (is.null(hours_vec)) next

    # Final safety: match lengths (order is already consistent with build loop)
    nl <- terra::nlyr(r_stack)
    if (length(hours_vec) != nl) {
      hours_vec <- hours_vec[seq_len(nl)]
    }

    terra::time(r_stack) <- hours_vec
    names(r_stack) <- format(hours_vec, "%Y-%m-%d %H:%M:%S")

    unit_v <- dplyr::case_when(
      grepl("^temp", v) ~ "K",
      v %in% c("rh") ~ "percent",
      v %in% c("plp", "mros_plp_proxy") ~ "1",
      v == "elev_coarse" ~ "m",
      TRUE ~ "1"
    )

    fn <- file.path(out_dir, sprintf("%s_%s.nc", v, format(day, "%Y%m%d")))
    terra::writeCDF(
      r_stack,
      filename   = fn,
      varname    = v,
      unit       = unit_v,
      longname   = v,
      overwrite  = TRUE,
      compression= 6,
      shuffle    = FALSE,   # avoid shuffle warning on float vars
      zname      = "time"
    )
  }
}


# Run over hours present in modeled_met_with_imerg
mm_dt <- modeled_met_with_imerg %>%
  dplyr::mutate(
    hour_utc = lubridate::floor_date(datetime_utc, "hour"),
    date     = as.Date(hour_utc)
  )

days_to_run <- sort(unique(mm_dt$date))

for (d in days_to_run) {
  hrs <- sort(unique(mm_dt$hour_utc[mm_dt$date == d]))

  daily_layers  <- list()  # var -> SpatRaster stack
  hours_by_var  <- list()  # var -> POSIXct vector of hours appended

  for (h in hrs) {
    maps <- build_grids_for_hour(h, modeled_met_with_imerg, coarse, grid_sf, idp, predictors_to_grid)
    if (is.null(maps)) next

    for (nm in names(maps)) {
      lyr <- maps[[nm]]
      if (is.null(lyr)) next

      if (is.null(daily_layers[[nm]])) {
        daily_layers[[nm]] <- lyr
        hours_by_var[[nm]] <- as.POSIXct(h, tz = "UTC")
      } else {
        daily_layers[[nm]] <- c(daily_layers[[nm]], lyr)
        hours_by_var[[nm]] <- c(hours_by_var[[nm]], as.POSIXct(h, tz = "UTC"))
      }
    }
  }

  write_daily_nc(d, daily_layers, hours_by_var, out_dir_nc)
}



# Visualizing -------------------------------
# Convert raster layer → data.frame for ggplot
r_to_df <- function(r, nm, h) {
  if (is.null(r)) return(NULL)
  df <- as.data.frame(r, xy = TRUE, na.rm = FALSE)

  # Rename *all* non-x/y cols to "value"
  other_cols <- setdiff(names(df), c("x","y"))
  if (length(other_cols) == 0) return(NULL)   # safety: nothing to plot
  if (length(other_cols) > 1) {
    warning("Multiple value columns found, taking the first one")
  }

  df$value <- df[[other_cols[1]]]
  df <- df[c("x","y","value")]

  df$var <- nm
  df$time <- h
  df
}



# Function
plot_hours <- function(hours) {
  dfs <- list()

  for (h in hours) {
    maps <- build_grids_for_hour(h, modeled_met_with_imerg, coarse, grid_sf, idp, predictors_to_grid)
    if (is.null(maps)) next

    keep <- c("temp_air_avg_obs","rh","plp","mros_plp_proxy","temp_wet","temp_dew_avg_obs")
    keep <- keep[keep %in% names(maps)]

    dfs[[as.character(h)]] <- dplyr::bind_rows(
      lapply(keep, function(k) r_to_df(maps[[k]], k, h))
    )
  }

  df_all <- dplyr::bind_rows(dfs)
  if (nrow(df_all) == 0) stop("No data frames produced — check build_grids_for_hour output")

  df_all <- dplyr::filter(df_all, !is.na(value))

  stations_sf <- sf::st_as_sf(meta_df, coords = c("lon","lat"), crs = 4326) %>%
    sf::st_transform(crs(coarse))

  ggplot2::ggplot(df_all, ggplot2::aes(x, y, fill = value)) +
    ggplot2::geom_raster() +
    ggplot2::facet_grid(var ~ time, scales = "free") +
    ggplot2::coord_equal() +
    ggplot2::scale_fill_viridis_c() +
    ggplot2::labs(title = "Interpolated surfaces", x = NULL, y = NULL, fill = NULL) +
    ggplot2::theme_minimal() +
    ggplot2::geom_sf(data = stations_sf, inherit.aes = FALSE,
                     shape = 24, fill = "white", color = "black",
                     size = 1.8, alpha = 0.9)
}

hrs_all <- sort(unique(lubridate::floor_date(modeled_met_with_imerg$datetime_utc, "hour")))
plot_hours(hrs_all[1:3])   # try first 3 hours
