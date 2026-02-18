# EWS - Early Warning Signals for Temporal and Spatial Systems

This repository contains a Python implementation of temporal and spatial **Early Warning Signal (EWS)** analysis for **(ecological) systems**, including **null model testing** following Dakos et al. (2008, 2011) and related extensions.

The code was developed in the context of a Master's thesis and prioritizes **methodological transparency, reproducibility, and statistical interpretability** over software packaging or performance optimization.

## What this code does

EWS are statistical indicators used to detect **critical slowing down** and other precursors of regime shifts in complex systems.

This code provides a **complete EWS pipeline** which allows one to:
- Compute EWS from temporal and/or spatial data,
- Assess trends using **Kendall's &tau;**,
- Compare observed trends against **null model distributions**,
- Quantify statistical significance via empirical quantiles,
- Explicitly account for **missing data and coverage limitations**.

The emphasis is on **statistical inference**, not just indicator visualization.

## Installation & requirements

This code was developed in a Conda environment. Core Python dependencies are listed in `requirements.txt`.

For PyCatch-specific functionality, **PCRaster** is required.

### PyCatch/PCRaster

Create a Conda environment using:
```bash
conda env create -f pcraster_pycatch.yaml
```
Activate the environment before running any scripts.

## How to run the pipeline

### 1. Configuring the model

To edit PyCatch model parameters, make necessary changes in `EWS_main_configuration.py` - note that one needs to run the EWS_pycatch_weekly.py model to get the necessary data to run EWS analysis in the current setup.

To edit EWS parameters (window sizes, indicators, null models, etc.), make necessary changes in `EWS_configuration.py` - note that some EWS parameters are mirrored from EWS_main_configuration.py.

### 2. Run the EWS analysis

```bash
python EWS_weekly.py
```

Run EWS_weekly.py, this computes EWS (on rolling windows for temporal data) for all enabled indicators - using `EWSPy.py functions` - and saves results as `.numpy.txt` files.

If enabled, null models are generated automatically.

### 3. Statistical testing

```bash
python EWS_Tests.py
```

This script computes Kendall's &tau; for observed EWS and computes null model &tau; distributions, reporting empirical quantiles and exceedance probablilities, reporting coverage diagnostics when NaNs are present.

### 4. Plotting

```bash
python EWS_weekly_plots.py
```

EWS_weekly_plots.py plots raw indicators over the defined number of weekly time steps the PyCatch model has ran (temporal evolution), with the option for multi-variable overlays.


## Conceptual pipeline

The implemented workflow follows the structure proposed by Dakos et al.:

1. **System simulation or data input**
   - Temporal time series or spatial maps (in this repo, the PyCatch hillslope model is used for system simulation).

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
   - Observed &tau; compared against null distributions.
   - Reporting empirical quantiles (e.g. &tau;0.05 and &tau;0.95).
  
All indicators are aligned to the **end of rolling windows**, ensuring temporal consistency between signal evaluation and trend estimation.

## Supported statistical properties

**Spatial:**
- Mean
- Standard deviation
- Variance
- Skewness
- Kurtosis
- Correlation (Moran's I)
- Discrete Fourier Transform (DFT)
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

Null models are used to test whether observed trends exceed expectations under stochastic variability while preserving selected properties of the data.

Implemented methods (temporal and spatial where applicable):

1. **Method 1 - Resampling/Shuffling**
   - Preserves marginal distribution.
   - Destorys temporal autocorrelation.

2. **Method 2 - Phase-randomized Fourier surrogates**
   - Preserves power spectrum and autocorrelation structure.
   - Randomizes Fourier phases.
  
3. **Method 3 - AR(1)-based null models**
   - Includes:
       - Standard AR(1).
       - Adjusted AR(1) following Yearsley (2021).
   - Stationarity explicitly enforced.
   - Variance is explicitly matched post hoc.
  
4. **Method 2B - IAAFT-based null models**
   - Iterative Amplitude Adjusted Fourier Transform (Schreiber & Schmitz, 1996; 2000).
   - Preserves exact amplitude structure and approximate power spectrum.
   - Iteratively enforces rank-order structure and spectral consistency, more strictly constrained than Method 2.

Null models are generated automatically when enabled in the configuration.

## Reproducibility

Reproducibility is controlled at two levels:

1. **System simulation (PyCatch)**
   - Fixed random seed in `EWS_pycatch_weekly.py`.
   - Fully reproducible for identical configurations.
     
2. **Null model generation**
   - Uses NumPy RNG.
   - Not fixed by default (independent realizations per run).
   - Users may optionally set:
     ```python
     numpy.random.seed(...)
     ```

All statistical conclusions (Kendall's &tau; distributions and quantiles) are robust to individual surrogate realizations when a sufficient number of null datasets is used.

## Statistical interpratation notes
- Both upper and lower empirical quantiles (&tau;0.95 and &tau;0.05) are evaluated.
- Strong negative deviations from null expectations may be as informative as strong positive ones.
- Exceeding the 95th percentile is evidence of non-random trend behaviour.
- Failure to exceed a threshold does not imply absence of early warning dynamics.
- Effect size, consistency across indicators, and ecological plausibility remain essential.

## Known limitations

- **NaN values in EWS indicators:**
  Some indicators (e.g. DFA, PS slope) may return NaN values for individual windows and/or spatial snapshots due to insufficient datapoints or numerical constraints.

- **Trend detection requires sufficient coverage:**
  Kendall's &tau; is only meaningful when a sufficient number of valid EWS values exist. Coverage is explicitly reported when &tau; cannot be computed under strict NaN handling.

- **&tau;0.05 and &tau;0.95 are heuristic thresholds:**
  Exceeding the 95th percentile of the null distribution is a strong indication of a non-random trend, but failure to exceed it does not imply absence of an EWS.
  Effect size and consistency across indicators should be considered.

- **Spatial assumptions:**
  Spatial indicators assume:
  - Regular grids,
  - Equal cell spacing,
  - Rook-neighbourhood adjacency for Moran's I.

- **Computational cost:**
  Spatial null models, especially AR(1)-based methods, can be computationally intensive for large grids.

## Common pitfalls

- **Interpreting Kendall's &tau; with missing data:**
  A NaN Kendall &tau; does not imply absence of a trend, as it often reflects insufficient coverage. In such cases, &tau; is computed with NaN omission and reported for diagnostic purposes, but should be interpreted cautiously.

- **Binary threshold thinking:**
  Null model comparison is distributional, not purely threshold-based.

- **Window-size sensitivity:**
  EWS trends can be sensitive to window size and overlap. Window-size sensitivity tests are provided and should be consulted before drawing conclusions.

- **Assuming null models represent "no dynamics":**
  Null models preserve selected statistical properties (e.g. autocorrelation and power spectrum). They do not represent the absence of structure, only absence of spectral EWS.

- **Expecting identical surrogate results across runs:**
  Null model realizations are stochastic unless the RNG seed is fixed explicitly.

### References

- Dakos et al. (2008) Slowing down as an early warning signal
- Dakos et al. (2011) Methods for detecting early warnings of critical transitions
- Yearsley (2021) Adjusted AR(1) null models
- Schreiber & Schmitz (1996, 2000) IAAFT surrogate methods

### License

Source files include SPDX license identifiers for automated license detection.

See the LICENSE file for details.
