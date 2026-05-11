# mros-precipitation-phase-product-prototype

Order of Steps

1. get_elevation.R: Download DEM from USGS AWS; specify areas of extent.
2. preprocessing_DEM.ipynb: reprojecting + resampling to 1km grids.
3. download_station_data.R: collect station data and post-process format
4. imerg_grid_data.R: collect IMERG data and post-process format
5. prism_setup_data.R: collect PRISM data and post-process format
6. preprocessing.ipynb: main data assimilation and clean up
7. resampling_IMERG_PRISM.ipynb: preprocessing and resampling for existing gridded datasets
8. interpolate: IDW_interpolation_updated.ipynb & kriging_interpolation_updated.ipynb
9. primary ML model: ML_XGBoost_binary_uncertainty.ipynb