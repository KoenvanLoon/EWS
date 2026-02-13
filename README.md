# EWS - Early Warning Signals for Temporal and Spatial Systems

This repository contains a Python implementation of temporal and spatial **Early Warning Signal (EWS)** analysis for **(ecological) systems**, including **null model testing** following Dakos et al. (2008, 2011) and related extensions.

The code was developed in the context of a Master's thesis and prioritizes **methodological transparency, reproducibility, and interpretability** over software packaging or performance optimization.

## What this code does

EWS are statistical indicators designed to detect **critical slowing down** and other precursors of regime shifts in complex systems.

This code provides a **complete EWS pipeline** which allows one to:
- Compute EWS from temporal and/or spatial data,
- Assess trends using **Kendall's &tau;**,
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
   - Kendall's &tau; between EWS values and time (or forcing).
  
6. **Statistical comparison**
   - Observed &tau; compared against null distributions and quantile thresholds.

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
- DFA (Detrended Fluctuation Analysis)

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
  
4. **Method 2B - IAAFT-based null models**
   - Iterative Amplitude Adjusted Fourier Transform (Schreiber & Schmitz, 1996, 2000).
   - Preserves amplitude structure and linear structure.

Null models are generated automatically when enabled in the configuration.

## Repository Structure

EWS/
- pycatch-master
  - EWS_main_configuration.py   # Main configuration for EWS_pycatch_weekly.py
  - EWS_configuration.py        # EWS centered configuration
  - EWS_pycatch_weekly.py       # PyCatch hillslope model
  - EWS_StateVariables.py       # State variable definitions
  - EWSPy.py                    # Core EWS functions
  - EWS_weekly.py               # Main EWS pipeline
  - EWS_null_spatial_weekly.py  # Spatial null models
  - EWS_temporal_weekly.py      # Temporal null models
  - EWS_Tests.py                # Kendall &tau; & null model tests
  - EWS_weekly_plots.py         # Plots results from EWS_weekly.py
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

EWS_Tests.py computes Kendall &tau; trends for observed EWS, null model distributions, and quantile thresholds when run.

Notes on missing data:
Some indicators (e.g. DFA or spatial power spectrum slope) may return NaN's for individual windows. Kendall &tau; is computed only when sufficient valid data is available; coverage information is reporeted to aid interpretation.

```bash
python EWS_weekly_plots.py
```

EWS_weekly_plots.py plots statistical properties over the defined number of weekly time steps the PyCatch model has ran.

## Reproducibility

The main PyCatch model execution uses an explicit random seed, set in `EWS_pycatch_weekly.py`, ensuring that **system simulations are reproducible** given identical configuration files and software versions.

Randomness in this pipeline arises from two sources:

1. **System simulation (PyCatch)**
   - Controlled via a fixed seed.
   - Fully reproducible.

2. **Null model generation**
   - Temporal and spatial null models (e.g. Fourier surrogates, AR(1)-based methods) rely on NumPy's random number generator.
   - These routines inherit the global NumPy RNG state at runtime.

For fully deterministic reproduction of null model realizations, users may optionally set an explicit NumPy seed (e.g. `np.random.seed(...)`) before null model generation.
This is not enforced by default to allow independent surrogate realizations across runs.

All statistical conclusions (Kendall's &tau; distributions and quantiles) are robust to individual surrogate realizations when a sufficient number of null datasets is used.

## Known limitations

- **NaN values in EWS indicators**
  Some indicators (e.g. DFA, PS slope) may return NaN values for individual windows and/or spatial snapshots due to insufficient datapoints or numerical constraints.

- **Trend detection requires sufficient coverage**
  Kendall's &tau; is only meaningful when a sufficient number of valid EWS values exist. Coverage is explicitly reported when &tau; cannot be computed under strict NaN handling.

- **&tau;0.95 is a heuristic threshold**
  Exceeding the 95yh percentile of the null distribution is a strong indication of a non-random trend, but failure to exceed it does not imply absence of an EWS.
  Effect size and consistency across indicators should be considered.

- **Spatial assumptions**
  Spatial indicators assume:
  - Regular grids,
  - Equal cell spacing,
  - Rook-neighbourhood adjacency for Moran's I.

- **Computational cost**
  Spatial null models, especially AR(1)-based methods, can be computationally intensive for large grids.

## Common pitfalls

- **Interpreting Kendall's &tau; with missing data**
  A NaN Kendall &tau; does not imply absence of a trend, as it often reflects insufficient coverage. In such cases, &tau; is computed with NaN omission and reported for diagnostic purposes, but should be interpreted cautiously.

- **Comparing observed &tau; directly to &tau;0.95**
  Observed &tau; values far outside the null-distribution can be informative even when they do not exceed t0.95 - null model comparison is distributional, not binary.

- **Window-size sensitivity**
  EWS trends can be sensitive to window size and overlap. Window-size sensitivity tests are provided and should be consulted before drawing conclusions.

- **Assuming null models represent "no dynamics"**
  Null models preserve selected statistical properties (e.g. autocorrelation and power spectrum). They do not represent the absence of structure, only absence of spectral EWS.

- **Expecting identical surrogate results across runs**
  Null model realizations are stochastic unless the RNG seed is fixed explicitly.

### References

- Dakos et al. (2008) Slowing down as an early warning signal
- Dakos et al. (2011) Methods for detecting early warnings of critical transitions
- Yearsley (2021) Adjusted AR(1) null models
- Schreiber & Schmitz (1996, 2000)

### License

Source files include SPDX license identifiers for automated license detection.

See the LICENSE file for details.
