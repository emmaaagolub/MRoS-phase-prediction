# Python Kriging Implementation for Precipitation Phase Analysis

This directory contains Python implementations of Kriging interpolation for precipitation phase analysis, it provides an alternative to the IDW approach written with (R).

**Google:** Kriging is a geostatistical interpolation technique used to predict values at unmeasured locations based on known values at nearby locations.

## Files Overview

### Core Implementation
- **`data_assim_spatial_intp_kriging.py`** - Core Kriging interpolator class with variogram fitting (spherical by default)
- **`kriging_workflow.py`** - Complete workflow implementation matching the R script structure
- **`requirements.txt`** - Python package dependencies

## Krigging vs. R IDW

### Same Outputs:
- Identical file structure and names
- Same coordinates and grid resolutions  
- Same variable names and data formats
- Same GeoTIFF outputs for GIS applications

### Additional Benefits:
- Uncertainty quantification for each interpolated value
- Statistical modeling of spatial correlation structure
- Model performance assessment
- Often 2-5x faster for large datasets
- Easy integration with other ML workflows

## Installation

```bash
cd Scripts/
pip install -r requirements.txt
```

## Usage

### Quick Start
```bash
# Run the complete workflow
python kriging_workflow.py

# Or use the core interpolator
python data_assim_spatial_intp_kriging.py
```

### Programmatic Usage
```python
from data_assim_spatial_intp_kriging import KrigingInterpolator
from kriging_workflow import KrigingWorkflow

# Initialize interpolator
kriging = KrigingInterpolator(variogram_model='spherical')

# Or run complete workflow
config = {
    'mros_parquet': "./Data/observations/wy25_mros_obs.parquet",
    'stations_dir': "./Data/Stations/",
    'stations_meta_csv': "./Data/Stations/station_metadata_20241001_20250531.csv",
    'imerg_dir': "./Data/IMERG/imerg_data-20250731T220028Z-1-001/imerg_data",
    'output_path': "./outputs/",
    'variogram_model': 'spherical'
}

workflow = KrigingWorkflow(config)
success = workflow.run_full_workflow()
```

## Output Structure

### GeoTIFF Files (in `outputs/phaseB_tifs/`)
```
temp_air_kriging_20250101T1200Z.tif          # Predictions
temp_air_kriging_20250101T1200Z_var.tif      # NEW: Prediction variances
temp_wet_kriging_20250101T1200Z.tif
temp_wet_kriging_20250101T1200Z_var.tif
temp_dew_kriging_20250101T1200Z.tif
temp_dew_kriging_20250101T1200Z_var.tif
rh_kriging_20250101T1200Z.tif
rh_kriging_20250101T1200Z_var.tif
```

### Data Files
```
modeled_met_kriging_20250101T1200Z.parquet   # Synced data with Kriging results
```

### Visualizations (in `outputs/plots/`)
```
temp_air_kriging_visualization.png           # Predictions + variances
temp_wet_kriging_visualization.png
temp_dew_kriging_visualization.png
rh_kriging_visualization.png
```