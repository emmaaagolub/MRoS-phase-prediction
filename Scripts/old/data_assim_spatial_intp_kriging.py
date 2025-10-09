#!/usr/bin/env python3
"""
Kriging Spatial Interpolation for Precipitation Phase Prediction

This script performs Kriging interpolation for meteorological variables and precipitation
phase prediction, it provides uncertainty quantification and in theory improves spatial modeling
compared to IDW methods.

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
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime
import pyarrow.parquet as pq

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KrigingInterpolator:
    """
    Kriging interpolation class for spatial data analysis
    """
    
    def __init__(self, variogram_model='spherical', nugget=0.0):
        """
        Initialize Kriging interpolator
        
        Parameters:
        -----------
        variogram_model : str
            Type of variogram model ('spherical', 'exponential', 'gaussian')
        nugget : float
            Nugget effect (microscale variation)
        """
        self.variogram_model = variogram_model
        self.nugget = nugget
        self.variogram_params = None
        self.fitted = False
        
    def spherical_variogram(self, h, range_param, sill):
        """Spherical variogram model"""
        h = np.array(h)
        gamma = np.zeros_like(h)
        
        # Within range
        within_range = h <= range_param
        gamma[within_range] = self.nugget + (sill - self.nugget) * (
            1.5 * h[within_range] / range_param - 
            0.5 * (h[within_range] / range_param) ** 3
        )
        
        # Beyond range
        gamma[h > range_param] = self.nugget + (sill - self.nugget)
        
        return gamma
    
    def exponential_variogram(self, h, range_param, sill):
        """Exponential variogram model"""
        h = np.array(h)
        return self.nugget + (sill - self.nugget) * (1 - np.exp(-3 * h / range_param))
    
    def gaussian_variogram(self, h, range_param, sill):
        """Gaussian variogram model"""
        h = np.array(h)
        return self.nugget + (sill - self.nugget) * (1 - np.exp(-3 * (h / range_param) ** 2))
    
    def empirical_variogram(self, coords, values, max_lag=None, n_lags=20):
        """
        Calculate empirical variogram
        
        Parameters:
        -----------
        coords : np.ndarray
            Coordinates (n_points, 2) for [x, y] or [lon, lat]
        values : np.ndarray
            Values at each coordinate
        max_lag : float, optional
            Maximum lag distance
        n_lags : int
            Number of lag bins
            
        Returns:
        --------
        lags : np.ndarray
            Lag distances
        gamma : np.ndarray
            Variogram values
        """
        # Calculate pairwise distances
        distances = cdist(coords, coords)
        
        # Remove diagonal (self-distances)
        upper_tri = np.triu_indices_from(distances, k=1)
        dist_vals = distances[upper_tri]
        value_diffs = (values.reshape(-1, 1) - values.reshape(1, -1))[upper_tri]
        
        # Calculate lag bins
        if max_lag is None:
            max_lag = np.percentile(dist_vals, 95)
        
        lag_bins = np.linspace(0, max_lag, n_lags + 1)
        lag_centers = (lag_bins[:-1] + lag_bins[1:]) / 2
        
        # Calculate variogram for each lag bin
        gamma = np.zeros(n_lags)
        n_pairs = np.zeros(n_lags)
        
        for i in range(n_lags):
            mask = (dist_vals >= lag_bins[i]) & (dist_vals < lag_bins[i + 1])
            if np.sum(mask) > 0:
                gamma[i] = 0.5 * np.mean(value_diffs[mask] ** 2)
                n_pairs[i] = np.sum(mask)
        
        # Remove bins with too few pairs
        min_pairs = 3
        valid_bins = n_pairs >= min_pairs
        lags = lag_centers[valid_bins]
        gamma = gamma[valid_bins]
        
        return lags, gamma
    
    def fit_variogram(self, coords, values, max_lag=None, n_lags=20):
        """
        Fit theoretical variogram to empirical data
        
        Parameters:
        -----------
        coords : np.ndarray
            Coordinates (n_points, 2)
        values : np.ndarray
            Values at each coordinate
        max_lag : float, optional
            Maximum lag distance
        n_lags : int
            Number of lag bins
            
        Returns:
        --------
        dict
            Fitted variogram parameters
        """
        # Calculate empirical variogram
        lags, gamma = self.empirical_variogram(coords, values, max_lag, n_lags)
        
        if len(lags) < 3:
            logger.warning("Insufficient data for variogram fitting")
            return None
        
        # Initial parameter estimates
        sill_est = np.var(values)
        range_est = np.median(lags)
        
        # Objective function for fitting
        def objective(params):
            if self.variogram_model == 'spherical':
                pred = self.spherical_variogram(lags, params[0], params[1])
            elif self.variogram_model == 'exponential':
                pred = self.exponential_variogram(lags, params[0], params[1])
            elif self.variogram_model == 'gaussian':
                pred = self.gaussian_variogram(lags, params[0], params[1])
            else:
                raise ValueError(f"Unknown variogram model: {self.variogram_model}")
            
            # Weighted least squares (more weight to shorter lags)
            weights = 1 / (lags + 1e-6)
            return np.sum(weights * (pred - gamma) ** 2)
        
        # Fit parameters
        try:
            result = minimize(objective, [range_est, sill_est], 
                           bounds=[(0.1, None), (0.1, None)])
            
            if result.success:
                self.variogram_params = {
                    'range': result.x[0],
                    'sill': result.x[1],
                    'nugget': self.nugget,
                    'model': self.variogram_model
                }
                self.fitted = True
                
                logger.info(f"Fitted {self.variogram_model} variogram: "
                          f"range={result.x[0]:.2f}, sill={result.x[1]:.2f}")
                
                return self.variogram_params
            else:
                logger.warning("Variogram fitting failed")
                return None
                
        except Exception as e:
            logger.error(f"Error in variogram fitting: {e}")
            return None
    
    def kriging_weights(self, src_coords, target_coords):
        """
        Calculate Kriging weights
        
        Parameters:
        -----------
        src_coords : np.ndarray
            Source coordinates (n_sources, 2)
        target_coords : np.ndarray
            Target coordinates (n_targets, 2)
            
        Returns:
        --------
        weights : np.ndarray
            Kriging weights (n_targets, n_sources)
        """
        if not self.fitted:
            raise ValueError("Variogram must be fitted before calculating weights")
        
        n_sources = src_coords.shape[0]
        n_targets = target_coords.shape[0]
        
        # Calculate variogram matrix for sources
        src_distances = cdist(src_coords, src_coords)
        src_gamma = self._calculate_variogram(src_distances)
        
        # Add Lagrange multiplier row/column
        src_gamma_aug = np.zeros((n_sources + 1, n_sources + 1))
        src_gamma_aug[:n_sources, :n_sources] = src_gamma
        src_gamma_aug[n_sources, :n_sources] = 1
        src_gamma_aug[:n_sources, n_sources] = 1
        src_gamma_aug[n_sources, n_sources] = 0
        
        # Calculate weights for each target
        weights = np.zeros((n_targets, n_sources))
        
        for i, target_coord in enumerate(target_coords):
            # Calculate variogram between target and sources
            target_distances = cdist([target_coord], src_coords)[0]
            target_gamma = self._calculate_variogram(target_distances)
            
            # Add Lagrange multiplier
            target_gamma_aug = np.append(target_gamma, 1)
            
            # Solve Kriging system
            try:
                target_weights = np.linalg.solve(src_gamma_aug, target_gamma_aug)
                weights[i, :] = target_weights[:n_sources]
            except np.linalg.LinAlgError:
                logger.warning(f"Singular matrix for target {i}, using nearest neighbor")
                # Fallback to nearest neighbor
                distances = cdist([target_coord], src_coords)[0]
                nearest_idx = np.argmin(distances)
                weights[i, nearest_idx] = 1.0
        
        return weights
    
    def _calculate_variogram(self, distances):
        """Calculate variogram values for given distances"""
        if self.variogram_model == 'spherical':
            return self.spherical_variogram(distances, 
                                         self.variogram_params['range'],
                                         self.variogram_params['sill'])
        elif self.variogram_model == 'exponential':
            return self.exponential_variogram(distances,
                                           self.variogram_params['range'],
                                           self.variogram_params['sill'])
        elif self.variogram_model == 'gaussian':
            return self.gaussian_variogram(distances,
                                        self.variogram_params['range'],
                                        self.variogram_params['sill'])
        else:
            raise ValueError(f"Unknown variogram model: {self.variogram_model}")
    
    def interpolate(self, src_coords, src_values, target_coords):
        """
        Perform Kriging interpolation
        
        Parameters:
        -----------
        src_coords : np.ndarray
            Source coordinates (n_sources, 2)
        src_values : np.ndarray
            Values at source coordinates
        target_coords : np.ndarray
            Target coordinates (n_targets, 2)
            
        Returns:
        --------
        predictions : np.ndarray
            Interpolated values
        variances : np.ndarray
            Prediction variances
        """
        if not self.fitted:
            raise ValueError("Variogram must be fitted before interpolation")
        
        # Calculate weights
        weights = self.kriging_weights(src_coords, target_coords)
        
        # Make predictions
        predictions = np.dot(weights, src_values)
        
        # Calculate prediction variances
        variances = np.zeros(len(target_coords))
        for i, target_coord in enumerate(target_coords):
            # Calculate variogram between target and sources
            target_distances = cdist([target_coord], src_coords)[0]
            target_gamma = self._calculate_variogram(target_distances)
            
            # Variance = sum of weights * variogram values
            variances[i] = np.sum(weights[i, :] * target_gamma)
        
        return predictions, variances
    
    def cross_validate(self, coords, values, n_folds=5):
        """
        Perform cross-validation
        
        Parameters:
        -----------
        coords : np.ndarray
            Coordinates (n_points, 2)
        values : np.ndarray
            Values at each coordinate
        n_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict
            Cross-validation metrics
        """
        n_points = len(coords)
        indices = np.arange(n_points)
        np.random.shuffle(indices)
        
        fold_size = n_points // n_folds
        predictions = np.full(n_points, np.nan)
        
        for fold in range(n_folds):
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < n_folds - 1 else n_points
            
            test_idx = indices[start_idx:end_idx]
            train_idx = np.setdiff1d(indices, test_idx)
            
            if len(train_idx) < 3:
                continue
            
            # Fit variogram on training data
            train_coords = coords[train_idx]
            train_values = values[train_idx]
            
            # Fit variogram
            self.fit_variogram(train_coords, train_values)
            
            # Predict on test data
            test_coords = coords[test_idx]
            test_predictions, _ = self.interpolate(train_coords, train_values, test_coords)
            
            predictions[test_idx] = test_predictions
        
        # Calculate metrics
        valid_mask = ~np.isnan(predictions)
        if np.sum(valid_mask) == 0:
            return None
        
        residuals = values[valid_mask] - predictions[valid_mask]
        
        metrics = {
            'rmse': np.sqrt(np.mean(residuals ** 2)),
            'mae': np.mean(np.abs(residuals)),
            'correlation': np.corrcoef(values[valid_mask], predictions[valid_mask])[0, 1],
            'bias': np.mean(residuals)
        }
        
        return metrics


def load_mros_data(parquet_path):
    """Load MRoS observations from parquet file"""
    logger.info(f"Loading MRoS data from {parquet_path}")
    
    try:
        df = pq.read_table(parquet_path).to_pandas()
        logger.info(f"Loaded {len(df)} MRoS observations")
        return df
    except Exception as e:
        logger.error(f"Error loading MRoS data: {e}")
        return None


def load_station_data(stations_dir, metadata_path):
    """Load station meteorological data"""
    logger.info(f"Loading station data from {stations_dir}")
    
    try:
        # Load metadata
        meta_df = pd.read_csv(metadata_path)
        logger.info(f"Loaded metadata for {len(meta_df)} stations")
        
        # Load station data files
        station_files = [f for f in os.listdir(stations_dir) 
                        if f.endswith('.csv') and 'meta' not in f.lower()]
        
        stations_list = []
        for file in station_files:
            file_path = os.path.join(stations_dir, file)
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
        
        logger.info(f"Loaded {len(stations_long)} station observations")
        return stations_long, meta_df
        
    except Exception as e:
        logger.error(f"Error loading station data: {e}")
        return None, None


def load_imerg_data(imerg_dir):
    """Load IMERG precipitation data"""
    logger.info(f"Loading IMERG data from {imerg_dir}")
    
    try:
        imerg_files = [f for f in os.listdir(imerg_dir) if f.endswith('.parquet')]
        logger.info(f"Found {len(imerg_files)} IMERG files")
        
        # For now, return file list - actual loading will be done per date
        return imerg_files
        
    except Exception as e:
        logger.error(f"Error loading IMERG data: {e}")
        return None


def create_output_directory(output_path):
    """Create output directory structure"""
    try:
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, "phaseB_tifs"), exist_ok=True)
        logger.info(f"Created output directory: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating output directory: {e}")
        return False


def save_raster_as_geotiff(data, output_path, transform, crs, nodata=-9999):
    """Save numpy array as GeoTIFF"""
    try:
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=crs,
            transform=transform,
            nodata=nodata
        ) as dst:
            dst.write(data, 1)
        
        logger.info(f"Saved raster: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving raster {output_path}: {e}")
        return False


def main():
    """Main execution function"""
    logger.info("Starting Kriging interpolation analysis")
    
    # Configuration
    config = {
        'mros_parquet': r"../Data/observations/wy25_mros_obs.parquet",
        'stations_dir': r"../Data/Stations/",
        'stations_meta_csv': r"../Data/Stations/station_metadata_20241001_20250531.csv",
        'imerg_dir': r"../Data/IMERG/imerg_data-20250731T220028Z-1-001/imerg_data",
        'dem_path': "C:/Users/EmmaGolub/Desktop/MRoS_local/local_data/DEM_AOI_TNM_10m.tif",
        'output_path': "./outputs/",
        'coarse_res_m': 1000,  # 1km grid
        'variogram_model': 'spherical',
        'min_points': 3,
        'time_tol_min': 30
    }
    
    # Create output directory
    if not create_output_directory(config['output_path']):
        logger.error("Failed to create output directory")
        return
    
    # Load data
    mros_data = load_mros_data(config['mros_parquet'])
    if mros_data is None:
        return
    
    stations_data, meta_df = load_station_data(config['stations_dir'], config['stations_meta_csv'])
    if stations_data is None:
        return
    
    imerg_files = load_imerg_data(config['imerg_dir'])
    if imerg_files is None:
        return
    
    # Initialize Kriging interpolator
    kriging = KrigingInterpolator(variogram_model=config['variogram_model'])
    
    logger.info("Data loading completed successfully")
    logger.info("Ready for Kriging interpolation analysis")
    
    # TODO: Implement the full workflow similar to the R script
    # This includes:
    # 1. Time synchronization
    # 2. Spatial interpolation with Kriging
    # 3. IMERG data integration
    # 4. Grid generation and surface creation
    # 5. Output generation
    
    logger.info("Kriging interpolation analysis completed")


if __name__ == "__main__":
    main()
