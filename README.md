# EWS - Early Warning Signals for Temporal and Spatial Systems

This repository contains a Python implementation of **Early Warning Signal (EWS)** analysis for temporal and spatial systems, including **null model tetsing** following Dakos et al. (2008, 2011).

The code was developed in the context of a Master's thesis, and is more focused on reproducibility and methodology, rather than software packaging.

## What this code does

**Pipeline:**
1. Uses the existing PyCatch model to generate ecological system data
2. Extracts rolling windows from time series, or maps from spatial data
3. Computes Early Warning Signals (EWS)
4. Generates null models (temporal and spatial)
5. Tests observed trends using Kendall's τ
6. Compares observed trends against null distributions

Supported statistical properties include:

**Spatial:**
- Mean
- Standard deviation
- Variance
- Skewness
- Kurtosis
- Correlation (Moran's I)
- DFT (Discrete Fourier Transform)
- Power spectrum slope

**Temporal:**
- Mean
- Standard deviation
- Variance
- Coefficient of variation
- Skewness
- Kurtosis
- AR(1)
- Return rate
- Conditional heteroskedasticity
- Autocorrelation
- DFA (Detrended Fluctiation Analysis)

## Repository Structure

EWS/
- pycatch-master
  - EWS_main_configuration.py  # Main configuration for EWS_pycatch_weekly.py
  - EWS_configuration.py  # EWS centered configuration
  - EWS_pycatch_weekly.py  # PyCatch hillslope model
  - EWS_StateVariables.py  # State variable definitions
  - EWSPy.py  # Core EWS functions
  - EWS_weekly.py  # Main EWS pipeline
  - EWS_null_spatial_weekly.py  # Spatial null models
  - EWS_temporal_weekly.py  # Temporal null models
  - EWS_Tests.py  # Kendall τ & null model tests
  - EWS_weekly_plots.py  # Plots results from EWS_weekly.py
- .gitignore
- README.md
- LICENSE

## How to run the code

### 0. PyCatch Installation

Create a conda environment by running pcraster_pycatch.yaml

### 1. Configuring the model

To edit PyCatch model parameters, make necessary changes in EWS_main_configuration.py

To edit EWS parameters, make necessary changes in EWS_configuration.py - note that some EWS parameters are mirrored from EWS_main_configuration.py.

To remove all output, run clean.sh - this does NOT remove inputs, so there is no major risk of losing input data, but check what is in clean.sh first.

### 2. Compute EWS for the system

Run EWS_weekly.py, this computes EWS (on rolling windows for temporal data) - using EWSPy.py functions - and saves results as .numpy.txt files.

### 3. Generate null models

Null models are generated automatically when enabled in the configuration.

Implemented methods include:
- Method 1: Resampling/shuffling
- Method 2: Phase-randomized Fourier surrogates
- Method 3: AR(1)-based null models

Temporal null models are implemented in EWS_null_temporal_weekly.py, and spatial null models are implemented in EWS_null_spatial_weekly.py.

### 4. Statistical testing & plotting

EWS_Tests.py computes Kendall τ trends for observed EWS, null model distributions, and quantile tresholds when ran.

Notes on missing data:
Some indicators (e.g. DFA or spatial power spectrum slope) may return NaN's for individual windows. Kendall τ is computed only when sufficient valid data is available; coverage information is reporated to aid interpretion.

EWS_weekly_plots.py plots statistical properties over the defined number of weekly time steps the PyCatch model has ran.

### References

- Dakos et al. (2008) Slowing down as an early warning signal
- Dakos et al. (2011) Methods for detecting early warnings of critical transitions
- Yearsley (2021) Adjusted AR(1) null models

### License

See the LICENSE file for details.
