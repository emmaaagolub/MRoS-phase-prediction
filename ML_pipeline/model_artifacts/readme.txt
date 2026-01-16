## Model Artifacts and Evaluation Outputs

This directory contains trained model artifacts and evaluation outputs for the precipitation-phase machine-learning pipeline. The model was trained using hourly, 1-km gridded predictors derived from inverse-distance weighted (IDW) interpolation of station data, combined with satellite and climatological datasets. Precipitation-phase labels are derived from an interpolated MRoS proxy, with additional masking based on IMERG precipitation likelihood.

All evaluation outputs correspond to a spatially independent test set defined using random spatial blocking.

---

### `xgb_phase_model.bin`

Serialized XGBoost model (`Booster`) trained to predict hourly precipitation phase (snow / mixed / rain).

* Model type: XGBoost multiclass classifier (`multi:softprob`)
* Outputs probabilistic predictions for three phase classes
* Trained on spatially blocked training data
* Validation data used for early stopping
* Test data held out entirely during training

This file contains the full trained tree ensemble and can be reloaded to:

* generate new predictions
* perform SHAP or other feature-attribution analyses
* reproduce evaluation results

---

### `feature_names.json`

Ordered list of predictor feature names used during model training.

This file defines the exact feature order expected by the trained model and must be used when:

* running SHAP analyses
* generating new predictions with the saved model
* ensuring reproducibility across environments

Predictors include:

* IDW-interpolated station variables (e.g., air temperature, wet-bulb temperature, relative humidity)
* IMERG precipitation likelihood
* PRISM climatological variables
* Static terrain attributes (e.g., elevation)

---

### `model_metadata.json`

Metadata describing the trained model and experimental setup.

Includes:

* class definitions (snow / mix / rain)
* label source (MRoS proxy)
* masking logic (e.g., IMERG precipitation threshold)
* spatial blocking strategy
* training and evaluation time period
* notes on model configuration and intended use

This file provides contextual information needed to correctly interpret results and is intended for transparency and reproducibility.

---

### `X_test.parquet`

Feature matrix for the spatially independent test set, used during final model evaluation.

Each row represents a single gridcell-hour sample:

* one 1-km grid cell
* at one hourly timestep
* with all predictor variables used by the model

This file does not include labels or predictions and is primarily intended for:

* post-hoc feature analysis
* SHAP computation on unseen data
* reproducibility checks

---

### `test_predictions.parquet`

Primary evaluation output for the spatially independent test set.

Each row corresponds to a gridcell-hour sample and includes:

* time stamp
* spatial coordinates (x, y)
* true precipitation-phase label (from MRoS proxy)
* predicted class label
* predicted class probabilities (snow / mix / rain)
* spatial block identifier

This file enables:

* spatial performance mapping
* temporal aggregation of predictions
* uncertainty analysis (e.g., entropy)
* stratified evaluation by elevation, precipitation regime, or label agreement

This is the main file used for generating figures and tables in analysis and manuscript results.

---

### `X_test_predictions_combined.parquet`

Convenience file combining test-set predictor values with corresponding predictions and probabilities.

Each row includes:

* predictor variables
* spatial and temporal metadata
* true labels
* predicted labels
* predicted probabilities

This file is useful for:

* joint feature–prediction diagnostics
* exploratory analysis
* rapid plotting without joining multiple files