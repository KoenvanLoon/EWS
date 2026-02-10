# EWS - Early Warning Signals for Temporal and Spatial Systems

This repository contains a Python implementation of temporal and spatial **Early Warning Signal (EWS)** analysis for **(ecological) systems**, including **null model tetsing** following Dakos et al. (2008, 2011) and related extensions.

The code was developed in the context of a Master’s thesis and prioritizes **methodological transparency, reproducibility, and interpretability** over software packaging or performance optimization.

## What this code does

EWS are statistical indicators designed to detect **critical slowing down** and other precursors of regime shifts in complex systems.

This code provides a **complete EWS pipeline** that allowes one to:
- Compute EWS from temporal and/or spatial data,
- Assess trends using **Kendall's τ**,
- Compare observed trends against **null model distributions**,
- Explicitly account for **missing data and coverage limitations**.

The emphasis is on **statistical inference**, not just indicator visualization.

## Conceptual pipeline

The implemented workflow follows the structure proposed by Dakos et al.:

1. **System simulation or data input**
   - Temporal time series or spatial maps (using the PyCatch hillslope model).

2. **Windowing / snapshot extraction**
   - Rolling windows for temporal indicators.
   - Sequential spatial snapshots for spatial indicators.

3. **EWS computation**
   - Indicators computed per window or snapshot.

4. **Null model generation**
   - Temporal and spatial surrogate datasets preserving selected properties.

5. **Trend estimation**
   - Kendall’s τ between EWS values and time (or forcing).
  
6. **Statistical comparison**
   - Observed τ compared against null distributions and quantile tresholds.

## Supported statistical properties

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

## Null models

Null models are used to test whether observed trends exceed what is expected under stochastic variability.

Implemented methods (temporal and spatial where applicable):

1. **Method 1 - Resampling/Shuffling**
   - Preserves marginal distribution

2. **Method 2 - Phase-randomized Fourier surrogates**
   - Preserves power spectrum and autocorrelation structure.
  
3. **Method 3 - AR(1)-based null models**
   - Includes:
       - Standard AR(1)
       - Adjusted AR(1) following Yearsley (2021)
   - Variance is explicitly matched post hoc.

Null models are generated automatically when enabled in the configuration.

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
- README.md
- requirements.txt
- LICENSE
- .gitignore

## Installation & requirements

This code was developed in a Conda environment. Core Python dependencies are listed in `requirements.txt`.

For PyCatch-specific functionality, PCRaster is required.

### PyCatch/PCRaster

Create a Conda environment using:
```bash
conda env create -f pcraster_pycatch.yaml
```
Activate the environment before running any scripts.

## How to run the pipeline

### 1. Configuring the model

To edit PyCatch model parameters, make necessary changes in EWS_main_configuration.py - note that one needs to run the EWS_pycatch_weekly.py model to get the necessary data to run EWS analysis in the current setup.

To edit EWS parameters (window sizes, indicators, null models, etc.), make necessary changes in EWS_configuration.py - note that some EWS parameters are mirrored from EWS_main_configuration.py.

### 2. Run the EWS analysis

```bash
python EWS_weekly.py
```

Run EWS_weekly.py, this computes EWS (on rolling windows for temporal data) for all enabled indicators - using EWSPy.py functions - and saves results as .numpy.txt files.

### 3. Generate null models

If enabled in the configuration, null models are generated automatically during the EWS run.

Temporal null models are implemented in EWS_null_temporal_weekly.py, and spatial null models are implemented in EWS_null_spatial_weekly.py.

### 4. Statistical testing & plotting

```bash
python EWS_Tests.py
```

EWS_Tests.py computes Kendall τ trends for observed EWS, null model distributions, and quantile tresholds when ran.

Notes on missing data:
Some indicators (e.g. DFA or spatial power spectrum slope) may return NaN's for individual windows. Kendall τ is computed only when sufficient valid data is available; coverage information is reporated to aid interpretion.

```bash
python EWS_weekly_plots.py
```

EWS_weekly_plots.py plots statistical properties over the defined number of weekly time steps the PyCatch model has ran.

### References

- Dakos et al. (2008) Slowing down as an early warning signal
- Dakos et al. (2011) Methods for detecting early warnings of critical transitions
- Yearsley (2021) Adjusted AR(1) null models

### License

See the LICENSE file for details.
