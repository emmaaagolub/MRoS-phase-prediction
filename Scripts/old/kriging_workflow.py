#!/usr/bin/env python3
"""
Complete Kriging Workflow for Precipitation Phase Prediction

This script implements the full workflow for Kriging interpolation, including:
1. Time synchronization between MRoS, station, and IMERG data
2. Spatial interpolation with Kriging and uncertainty quantification
3. Grid generation and surface creation
4. Output generation matching the R script format

"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime, timedelta
import pyarrow.parquet as pq
from data_assim_spatial_intp_kriging import KrigingInterpolator

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KrigingWorkflow:
    """
    Complete workflow for Kriging interpolation analysis
    """
    
    def __init__(self, config):
        """
        Initialize workflow with configuration
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary with paths and parameters
        """
        self.config = config
        self.kriging = KrigingInterpolator(variogram_model=config['variogram_model'])
        self.mros_data = None
        self.stations_data = None
        self.meta_df = None
        self.imerg_data = None
        
    def load_all_data(self):
        """Load all required datasets"""
        logger.info("Loading all datasets...")
        
        # Load MRoS data
        self.mros_data = self._load_mros_data()
        if self.mros_data is None:
            return False
            
        # Load station data
        self.stations_data, self.meta_df = self._load_station_data()
        if self.stations_data is None:
            return False
            
        # Load IMERG data
        self.imerg_data = self._load_imerg_data()
        if self.imerg_data is None:
            return False
            
        logger.info("All datasets loaded successfully")
        return True
    
    def _load_mros_data(self):
        """Load and preprocess MRoS observations"""
        try:
            df = pq.read_table(self.config['mros_parquet']).to_pandas()
            
            # Convert to datetime and add elevation
            df['datetime_utc'] = pd.to_datetime(df['date_submitted_utc'] + ' ' + df['time_submitted_utc'], utc=True)
            df['latitude'] = pd.to_numeric(df['latitude'])
            df['longitude'] = pd.to_numeric(df['longitude'])
            
            # Add row ID for tracking
            df['row_id'] = range(len(df))
            
            # Select and filter columns
            df = df[['row_id', 'phase', 'latitude', 'longitude', 'datetime_utc']].copy()
            
            logger.info(f"Loaded {len(df)} MRoS observations")
            return df
            
        except Exception as e:
            logger.error(f"Error loading MRoS data: {e}")
            return None
    
    def _load_station_data(self):
        """Load and preprocess station meteorological data"""
        try:
            # Load metadata
            meta_df = pd.read_csv(self.config['stations_meta_csv'])
            meta_df = meta_df.rename(columns={'id': 'station_id', 'lat': 'latitude', 'lon': 'longitude', 'elev': 'elevation'})
            
            # Load station data files
            station_files = [f for f in os.listdir(self.config['stations_dir']) 
                           if f.endswith('.csv') and 'meta' not in f.lower()]
            
            stations_list = []
            for file in station_files:
                file_path = os.path.join(self.config['stations_dir'], file)
                df = pd.read_csv(file_path)
                
                # Ensure required columns exist
                required_cols = ['temp_air', 'temp_wet', 'temp_dew', 'rh']
                for col in required_cols:
                    if col not in df.columns:
                        df[col] = np.nan
                
                # Parse datetime
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
                
                # Add station ID
                if 'id' not in df.columns:
                    df['id'] = os.path.splitext(file)[0]
                
                df = df[['id', 'datetime'] + required_cols]
                stations_list.append(df)
            
            stations_long = pd.concat(stations_list, ignore_index=True)
            stations_long = stations_long.sort_values(['id', 'datetime'])
            
            logger.info(f"Loaded {len(stations_long)} station observations from {len(station_files)} files")
            return stations_long, meta_df
            
        except Exception as e:
            logger.error(f"Error loading station data: {e}")
            return None, None
    
    def _load_imerg_data(self):
        """Load IMERG precipitation data"""
        try:
            imerg_files = [f for f in os.listdir(self.config['imerg_dir']) if f.endswith('.parquet')]
            
            # Load a sample to understand structure
            sample_file = os.path.join(self.config['imerg_dir'], imerg_files[0])
            sample_data = pq.read_table(sample_file).to_pandas()
            
            logger.info(f"Found {len(imerg_files)} IMERG files")
            logger.info(f"Sample file structure: {sample_data.columns.tolist()}")
            
            return imerg_files
            
        except Exception as e:
            logger.error(f"Error loading IMERG data: {e}")
            return None
    
    def time_synchronization(self, time_window_hours=1):
        """
        Synchronize data across different time sources
        
        Parameters:
        -----------
        time_window_hours : int
            Time window for matching observations (hours)
        """
        logger.info("Performing time synchronization...")
        
        # For each MRoS observation, find nearest station data within time window
        synced_data = []
        
        for idx, mros_row in self.mros_data.iterrows():
            mros_time = mros_row['datetime_utc']
            mros_lat = mros_row['latitude']
            mros_lon = mros_row['longitude']
            
            # Find station data within time window
            time_diff = abs(self.stations_data['datetime'] - mros_time)
            within_window = time_diff <= timedelta(hours=time_window_hours)
            
            if within_window.any():
                # Get nearest station data
                window_data = self.stations_data[within_window].copy()
                
                # Calculate distances to stations
                station_coords = self.meta_df[['latitude', 'longitude']].values
                mros_coords = np.array([[mros_lat, mros_lon]])
                distances = cdist(mros_coords, station_coords)[0]
                
                # Find nearest station
                nearest_idx = np.argmin(distances)
                nearest_station = self.meta_df.iloc[nearest_idx]
                
                # Get station data for this time
                station_id = nearest_station['station_id']
                station_data = window_data[window_data['id'] == station_id]
                
                if not station_data.empty:
                    # Get the closest time match
                    closest_idx = (station_data['datetime'] - mros_time).abs().idxmin()
                    closest_data = station_data.loc[closest_idx]
                    
                    # Combine MRoS and station data
                    synced_row = {
                        'row_id': mros_row['row_id'],
                        'phase': mros_row['phase'],
                        'latitude': mros_lat,
                        'longitude': mros_lon,
                        'datetime_utc': mros_time,
                        'station_id': station_id,
                        'station_lat': nearest_station['latitude'],
                        'station_lon': nearest_station['longitude'],
                        'station_elev': nearest_station['elevation'],
                        'temp_air': closest_data['temp_air'],
                        'temp_wet': closest_data['temp_wet'],
                        'temp_dew': closest_data['temp_dew'],
                        'rh': closest_data['rh'],
                        'distance_to_station': distances[nearest_idx]
                    }
                    
                    synced_data.append(synced_row)
        
        self.synced_data = pd.DataFrame(synced_data)
        logger.info(f"Time synchronization completed: {len(self.synced_data)} observations synced")
        
        return self.synced_data
    
    def fit_variograms(self):
        """Fit variogram models for each meteorological variable"""
        logger.info("Fitting variogram models...")
        
        variables = ['temp_air', 'temp_wet', 'temp_dew', 'rh']
        variogram_results = {}
        
        for var in variables:
            # Get valid data for this variable
            valid_mask = ~self.synced_data[var].isna()
            if valid_mask.sum() < 10:  # Need sufficient data
                logger.warning(f"Insufficient data for {var}: {valid_mask.sum()} valid observations")
                continue
            
            var_data = self.synced_data[valid_mask]
            coords = var_data[['longitude', 'latitude']].values
            values = var_data[var].values
            
            # Fit variogram
            logger.info(f"Fitting variogram for {var} with {len(coords)} points")
            variogram_params = self.kriging.fit_variogram(coords, values)
            
            if variogram_params:
                variogram_results[var] = variogram_params
                logger.info(f"Variogram fitted for {var}: {variogram_params}")
            else:
                logger.warning(f"Failed to fit variogram for {var}")
        
        self.variogram_results = variogram_results
        return variogram_results
    
    def create_interpolation_grid(self, resolution_m=1000):
        """
        Create interpolation grid
        
        Parameters:
        -----------
        resolution_m : int
            Grid resolution in meters
        """
        logger.info(f"Creating interpolation grid with {resolution_m}m resolution...")
        
        # Get bounds from synced data
        bounds = self.synced_data[['longitude', 'latitude']].agg(['min', 'max']).values
        
        # DEBUG: Print bounds and data info
        logger.info(f"Data bounds - Longitude: {bounds[0][0]:.6f} to {bounds[0][1]:.6f}")
        logger.info(f"Data bounds - Latitude: {bounds[1][0]:.6f} to {bounds[1][1]:.6f}")
        logger.info(f"Data extent - Lon span: {bounds[0][1] - bounds[0][0]:.6f} degrees")
        logger.info(f"Data extent - Lat span: {bounds[1][1] - bounds[1][0]:.6f} degrees")
        
        # Create grid
        lon_min, lon_max = bounds[0]
        lat_min, lat_max = bounds[1]
        
        # Convert to approximate meters (rough conversion)
        # 1 degree lat ≈ 111,000 meters
        # 1 degree lon ≈ 111,000 * cos(lat) meters
        lat_center = (lat_min + lat_max) / 2
        lon_res = resolution_m / (111000 * np.cos(np.radians(lat_center)))
        lat_res = resolution_m / 111000
        
        # DEBUG: Print resolution calculations
        logger.info(f"Latitude center: {lat_center:.6f}")
        logger.info(f"Calculated lon_res: {lon_res:.8f} degrees")
        logger.info(f"Calculated lat_res: {lat_res:.8f} degrees")
        
        # QUICK FIX: Ensure minimum grid resolution and bounds
        min_res = 0.001  # Minimum 0.001 degrees (~111 meters)
        lon_res = max(lon_res, min_res)
        lat_res = max(lat_res, min_res)
        
        # Ensure we have at least some grid points
        if (lon_max - lon_min) < lon_res:
            lon_res = (lon_max - lon_min) / 10  # At least 10 points
            logger.warning(f"Lon span too small, adjusted lon_res to {lon_res:.8f}")
        
        if (lat_max - lat_min) < lat_res:
            lat_res = (lat_max - lat_min) / 10  # At least 10 points
            logger.warning(f"Lat span too small, adjusted lat_res to {lat_res:.8f}")
        
        # Create grid coordinates
        lon_coords = np.arange(lon_min, lon_max + lon_res, lon_res)
        lat_coords = np.arange(lat_min, lat_max + lat_res, lat_res)
        
        # DEBUG: Print grid dimensions
        logger.info(f"Lon coordinates: {len(lon_coords)} points from {lon_coords[0]:.6f} to {lon_coords[-1]:.6f}")
        logger.info(f"Lat coordinates: {len(lat_coords)} points from {lat_coords[0]:.6f} to {lat_coords[-1]:.6f}")
        
        # Create meshgrid
        lon_grid, lat_grid = np.meshgrid(lon_coords, lat_coords)
        
        # Flatten for processing
        grid_coords = np.column_stack([lon_grid.flatten(), lat_grid.flatten()])
        
        self.grid_coords = grid_coords
        self.grid_shape = lon_grid.shape
        self.grid_bounds = bounds
        
        logger.info(f"Created grid with shape {self.grid_shape} ({len(grid_coords)} points)")
        
        # DEBUG: Verify grid is valid
        if self.grid_shape[0] == 0 or self.grid_shape[1] == 0:
            logger.error(f"Invalid grid shape: {self.grid_shape}")
            logger.error("This will cause errors when saving GeoTIFF files")
            raise ValueError(f"Grid creation failed: invalid shape {self.grid_shape}")
        
        return grid_coords
    
    def interpolate_variables(self):
        """Perform Kriging interpolation for all variables"""
        logger.info("Performing Kriging interpolation...")
        
        if not hasattr(self, 'grid_coords'):
            raise ValueError("Grid must be created before interpolation")
        
        interpolation_results = {}
        
        for var, variogram_params in self.variogram_results.items():
            logger.info(f"Interpolating {var}...")
            
            # Get source data
            valid_mask = ~self.synced_data[var].isna()
            var_data = self.synced_data[valid_mask]
            
            if len(var_data) < 3:
                logger.warning(f"Insufficient data for {var}: {len(var_data)} points")
                continue
            
            src_coords = var_data[['longitude', 'latitude']].values
            src_values = var_data[var].values
            
            # Perform interpolation
            try:
                predictions, variances = self.kriging.interpolate(src_coords, src_values, self.grid_coords)
                
                # Reshape to grid
                pred_grid = predictions.reshape(self.grid_shape)
                var_grid = variances.reshape(self.grid_shape)
                
                interpolation_results[var] = {
                    'predictions': pred_grid,
                    'variances': var_grid,
                    'source_points': len(src_coords)
                }
                
                logger.info(f"Interpolated {var}: {len(src_coords)} source points")
                
            except Exception as e:
                logger.error(f"Error interpolating {var}: {e}")
                continue
        
        self.interpolation_results = interpolation_results
        return interpolation_results
    
    def save_results(self):
        """Save interpolation results as GeoTIFF files"""
        logger.info("Saving results...")
        
        if not hasattr(self, 'interpolation_results'):
            raise ValueError("No interpolation results to save")
        
        # Create output directory
        output_dir = os.path.join(self.config['output_path'], 'phaseB_tifs')
        os.makedirs(output_dir, exist_ok=True)
        
        # Get timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%dT%H%MZ")
        
        # Save each variable
        for var, results in self.interpolation_results.items():
            # Save predictions
            pred_filename = f"{var}_kriging_{timestamp}.tif"
            pred_path = os.path.join(output_dir, pred_filename)
            
            # Save variances
            var_filename = f"{var}_kriging_{timestamp}_var.tif"
            var_path = os.path.join(output_dir, var_filename)
            
            # Create transform and CRS
            bounds = self.grid_bounds
            transform = from_origin(bounds[0][0], bounds[1][1], 
                                  (bounds[0][1] - bounds[0][0]) / self.grid_shape[1],
                                  (bounds[1][1] - bounds[1][0]) / self.grid_shape[0])
            
            crs = CRS.from_epsg(4326)  # WGS84
            
            # Save prediction raster
            with rasterio.open(pred_path, 'w', driver='GTiff',
                             height=self.grid_shape[0], width=self.grid_shape[1],
                             count=1, dtype=results['predictions'].dtype,
                             crs=crs, transform=transform, nodata=-9999) as dst:
                dst.write(results['predictions'], 1)
            
            # Save variance raster
            with rasterio.open(var_path, 'w', driver='GTiff',
                             height=self.grid_shape[0], width=self.grid_shape[1],
                             count=1, dtype=results['variances'].dtype,
                             crs=crs, transform=transform, nodata=-9999) as dst:
                dst.write(results['variances'], 1)
            
            logger.info(f"Saved {var}: {pred_filename}, {var_filename}")
        
        # Save synced data as parquet
        synced_filename = f"modeled_met_kriging_{timestamp}.parquet"
        synced_path = os.path.join(self.config['output_path'], synced_filename)
        self.synced_data.to_parquet(synced_path)
        logger.info(f"Saved synced data: {synced_filename}")
        
        logger.info("All results saved successfully")
    
    def create_visualizations(self):
        """Create visualization plots"""
        logger.info("Creating visualizations...")
        
        if not hasattr(self, 'interpolation_results'):
            raise ValueError("No interpolation results to visualize")
        
        # Create output directory for plots
        plots_dir = os.path.join(self.config['output_path'], 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot each variable
        for var, results in self.interpolation_results.items():
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot predictions
            im1 = ax1.imshow(results['predictions'], cmap='viridis')
            ax1.set_title(f'{var} - Predictions')
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')
            plt.colorbar(im1, ax=ax1)
            
            # Plot variances
            im2 = ax2.imshow(results['variances'], cmap='plasma')
            ax2.set_title(f'{var} - Prediction Variance')
            ax2.set_xlabel('Longitude')
            ax2.set_ylabel('Latitude')
            plt.colorbar(im2, ax=ax2)
            
            plt.tight_layout()
            
            # Save plot
            plot_filename = f"{var}_kriging_visualization.png"
            plot_path = os.path.join(plots_dir, plot_filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Created visualization: {plot_filename}")
        
        logger.info("Visualizations completed")
    
    def run_full_workflow(self):
        """Run the complete workflow"""
        logger.info("Starting complete Kriging workflow...")
        
        try:
            # Step 1: Load data
            if not self.load_all_data():
                logger.error("Failed to load data")
                return False
            
            # Step 2: Time synchronization
            self.time_synchronization()
            
            # Step 3: Fit variograms
            self.fit_variograms()
            
            # Step 4: Create grid
            self.create_interpolation_grid()
            
            # Step 5: Perform interpolation
            self.interpolate_variables()
            
            # Step 6: Save results
            self.save_results()
            
            # Step 7: Create visualizations
            self.create_visualizations()
            
            logger.info("Kriging workflow completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            return False


def main():
    """Main execution function"""
    # Configuration
    config = {
        'mros_parquet': r"../Data/observations/wy25_mros_obs.parquet",
        'stations_dir': r"../Data/Stations/",
        'stations_meta_csv': r"../Data/Stations/station_metadata_20241001_20250531.csv",
        'imerg_dir': r"../Data/IMERG/imerg_data-20250731T220028Z-1-001/imerg_data",
        'dem_path': r"C:/Users/EmmaGolub/Desktop/MRoS_local/local_data/DEM_AOI_TNM_10m.tif",
        'output_path': "./outputs/",
        'coarse_res_m': 1000,  # 1km grid
        'variogram_model': 'spherical',
        'min_points': 3,
        'time_tol_min': 30
    }
    
    # Create and run workflow
    workflow = KrigingWorkflow(config)
    success = workflow.run_full_workflow()
    
    if success:
        logger.info("Kriging analysis completed successfully!")
    else:
        logger.error("Kriging analysis failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()



# INFO - Interpolated rh: 504 source points
# INFO - Saving results...
# INFO - Saved temp_air: temp_air_kriging_20250826T0843Z.tif, temp_air_kriging_20250826T0843Z_var.tif
# INFO - Saved rh: rh_kriging_20250826T0843Z.tif, rh_kriging_20250826T0843Z_var.tif
# INFO - Saved synced data: modeled_met_kriging_20250826T0843Z.parquet
# INFO - All results saved successfully
# INFO - Creating visualizations...
# INFO - Created visualization: temp_air_kriging_visualization.png
# INFO - Created visualization: rh_kriging_visualization.png
# INFO - Visualizations completed
# INFO - Kriging workflow completed successfully!
# INFO - Kriging analysis completed successfully!