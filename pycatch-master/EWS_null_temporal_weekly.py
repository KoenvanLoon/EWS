# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (c) 2026 Koen van Loon
#
# See the LICENSE file in the repository root for full license text.

"""
EWS - Early Warning Signals
Null models timeseries weekly

@authors: KoenvanLoon
"""

import numpy as np
from scipy import fft, ndimage, signal
import statsmodels.api
import os
import warnings
import EWSPy as ews
import EWS_configuration as cfg

### Null models timeseries (Dakos et al. 2008) ###

# Detrend dataset
"""
Detrends the given dataset making use of either Gaussian filtering or linear detrending, as specified in the 
configuration. Optionally, this data is saved.

Args:
-----

data : numpy array, the timeseries data.

realizations : int, the number of datasets generated. Used for folder name in the case of detrending.

path : str, the filepath where the original dataset can be found.

variable : name of the variable.

Return:
-----

detrended_data : The detrended timeseries data.

"""


def detrend_(data, realizations=1, path='./1/', variable='xxx'):
    data = np.asarray(data)
    assert data.ndim == 1, "Temporal null models require 1D time series"

    detrended_data = data.copy()
    gaussian_filter = None

    if cfg.detrended_method == 'Gaussian':
        if cfg.detrended_sigma == 0:
            # Sigma = 0 corresponds to no smoothing (identity operation), so, in sensitivity analyses, sigma = 0 is
            #   interpreted as "no detrending".
            detrended_data = data.copy()
        else:
            gaussian_filter = ndimage.gaussian_filter1d(data, cfg.detrended_sigma)
            detrended_data -= gaussian_filter
    elif cfg.detrended_method == 'MeanLinear':
        detrended_data = signal.detrend(data)
        # Piecewise detrending per window (as below) may artificially remove EWS
        # detrended_data = signal.detrend(data, bp=np.arange(0, cfg.number_of_timesteps_weekly, variable.window_size))
    elif cfg.detrended_method != 'None':
        warnings.warn("Invalid input for detrend_ (temporal) in generate_datasets (EWS_weekly.py). No detrending done.")

    if cfg.save_detrended_data:
        generated_number_length = ews.generated_number_length(realizations)
        generated_number_string = 'dtr' + str(0).zfill(generated_number_length)
        dir_name = os.path.join(path, generated_number_string)

        os.makedirs(dir_name, exist_ok=True)

        fname1 = ews.file_name_str(variable.name, cfg.number_of_timesteps_weekly)
        fpath1 = os.path.join(dir_name, fname1)
        np.savetxt(fpath1 + '.numpy.txt', detrended_data)

        if cfg.detrended_method == 'Gaussian' and gaussian_filter is not None:
            fname2 = ews.file_name_str(variable.name + 'g', cfg.number_of_timesteps_weekly)
            fpath2 = os.path.join(dir_name, fname2)
            # noinspection PyTypeChecker
            np.savetxt(fpath2 + '.numpy.txt', gaussian_filter)

        if cfg.detrended_method == 'MeanLinear':
            fname2 = ews.file_name_str(variable.name + 'l', cfg.number_of_timesteps_weekly)
            fpath2 = os.path.join(dir_name, fname2)
            np.savetxt(fpath2 + '.numpy.txt', data - detrended_data)

    return detrended_data


# Generate datasets method 1
"""
Generates dataset(s) with similar mean and variance by randomly picking values from the original dataset. In the case
of replace==False, this is similar to shuffling the dataset.

Args:
-----

data : numpy array, the spatial datasets.

realizations : int, the number of datasets generated.

path : str, the filepath where the original dataset can be found.

variable : name of the variable.

replace : bool, selects whether new values are picked from the original dataset or the original dataset minus previously
    picked values. Usually set to False to ensure similar mean and variance for smaller datasets.

"""


def method1_(data, realizations=1, path='./1/', variable='xxx', replace=False):
    data = np.asarray(data)
    assert data.ndim == 1, "Temporal null models require 1D time series"

    generated_number_length = ews.generated_number_length(realizations)

    for realization in range(realizations):

        try:
            generated_dataset = np.random.choice(data, len(data), replace=replace)

        except ValueError:
            warnings.warn(f"Dataset too small or has repeated values; falling back to replace=True for variable {variable.name}")
            generated_dataset = np.random.choice(data, len(data), replace=True)

        generated_number_string = 'm1g' + str(realization).zfill(generated_number_length)
        dir_name = os.path.join(path, generated_number_string)

        os.makedirs(dir_name, exist_ok=True)

        fname = ews.file_name_str(variable.name, cfg.number_of_timesteps_weekly)
        fpath = os.path.join(dir_name, fname)
        np.savetxt(fpath + '.numpy.txt', generated_dataset)


# Generate datasets method 2
"""
Generates dataset(s) with similar autocorrelation, mean and variance by generating datasets with the same Fourier 
spectrum and amplitudes.

Args:
-----

data : numpy array, the spatial datasets.

realizations : int, the number of datasets generated.

method : str, either 'None' or 'Detrending', if detrended data is used as input, no further detrending is needed. If
    not-detrended data is used, linear detrending is applied before the Fourier spectrum and amplitudes are calculated,
    with the linear detrend added after the generation of datasets. 

path : str, the filepath where the original dataset can be found.

variable : name of the variable.

replace : bool, selects whether new values are picked from the original dataset or the original dataset minus previously
    picked values.

"""


def method2_(data, realizations=1, method='Detrending', path='./1/', variable='xxx'):
    data = np.asarray(data)
    assert data.ndim == 1, "Temporal null models require 1D time series"
    if np.isnan(data).any():
        raise ValueError("NaNs present in time series")

    generated_number_length = ews.generated_number_length(realizations)

    lin_detr = None
    if method == 'Detrending':
        detrended_data = signal.detrend(data)
        # Piecewise detrending per window (as below) may artificially remove EWS
        # detrended_data = signal.detrend(data, bp=np.arange(0, cfg.number_of_timesteps_weekly, variable.window_size))
        lin_detr = data - detrended_data
        data = detrended_data

    fft_ = fft.rfft(data)
    fft_mag = np.abs(fft_)

    for realization in range(realizations):

        # Random phases
        random_phases = np.random.uniform(0, 2*np.pi, len(fft_mag))

        # Preserve DC component
        random_phases[0] = 0.0

        # If even length, preserve Nyquist frequency (must be real, keep it real)
        if len(data) % 2 == 0:
            random_phases[-1] = 0.0

        fft_new = fft_mag * np.exp(1j * random_phases)

        generated_dataset = fft.irfft(fft_new, n=len(data))

        if method == 'Detrending':
            generated_dataset += lin_detr

        generated_number_string = 'm2g' + str(realization).zfill(generated_number_length)
        dir_name = os.path.join(path, generated_number_string)

        os.makedirs(dir_name, exist_ok=True)

        fname = ews.file_name_str(variable.name, cfg.number_of_timesteps_weekly)
        fpath = os.path.join(dir_name, fname)
        np.savetxt(fpath + '.numpy.txt', generated_dataset)


# Generate datasets method 3
"""
Generates dataset(s) with similar autocorrelation, mean and variance by generating datasets with an AR(1) model trained
on the original dataset.

Args:
-----

data : numpy array, the spatial datasets.

realizations : int, the number of datasets generated.

method : str, either 'Normal' or 'Adjusted'. For 'Normal', the standard AR(1) format is used. For 'Adjusted', the AR(1)
    format of Jon Yearsley (2021) is used.

path : str, the filepath where the original dataset can be found.

variable : name of the variable.

stdev_error : int/float, the standard deviation of the white noise process.

"""


def method3_(data, realizations=1, method='Normal', path='./1/', variable='xxx'):
    data = np.asarray(data)
    assert data.ndim == 1, "Temporal null models require 1D time series"

    generated_number_length = ews.generated_number_length(realizations)

    alpha1 = statsmodels.api.tsa.acf(data, nlags=1)
    sig2 = np.nanvar(data) * (1 - alpha1[1] ** 2)
    alpha0_1 = np.nanmean(data) * (1 - alpha1[1])
    alpha0_2 = np.nanmean(data)

    phi = alpha1[1]
    den = 1 - phi ** 2

    if den <= 0:
        warnings.warn(
            f"Non-stationary AR(1) detected (phi={phi:.5f})."
            "Returning NaN surrogate."
            )
        AR1m = np.full_like(data, np.nan)

        for realization in range(realizations):
            generated_number_string = 'm3g' + str(realization).zfill(generated_number_length)
            dir_name = os.path.join(path, generated_number_string)

            os.makedirs(dir_name, exist_ok=True)

            fname = ews.file_name_str(variable.name, cfg.number_of_timesteps_weekly)
            fpath = os.path.join(dir_name, fname)
            np.savetxt(fpath + '.numpy.txt', AR1m)

        return

    for realization in range(realizations):
        e = np.random.normal(0.0, 1.0, size=len(data))
        if method == 'Adjusted':
            AR1m = np.zeros(len(data))
            AR1m[0] = np.random.normal(loc=np.nanmean(data), scale=np.sqrt(sig2 / den))
            for i in range(1, len(data)):
                AR1m[i] = phi * AR1m[i - 1] + alpha0_1 + np.sqrt(sig2) * e[i]
        elif method == 'Normal':
            AR1m = np.zeros(len(data))
            AR1m[0] = np.random.normal(loc=np.nanmean(data), scale=np.sqrt(sig2 / den))
            for i in range(1, len(data)):
                AR1m[i] = phi * AR1m[i - 1] + alpha0_2 + np.sqrt(sig2) * e[i]
        else:
            assert False, "Incorrect method."
        generated_number_string = 'm3g' + str(realization).zfill(generated_number_length)
        dir_name = os.path.join(path, generated_number_string)

        os.makedirs(dir_name, exist_ok=True)

        fname = ews.file_name_str(variable.name, cfg.number_of_timesteps_weekly)
        fpath = os.path.join(dir_name, fname)
        np.savetxt(fpath + '.numpy.txt', AR1m)

# Generate datasets method 2b - IAAFT (or; how we moved on from Dakos to Dragons)

"""
Generate IAAFT (Iterative Amplitude Adjusted Fourier Transform (Schreiber & Schmitz, 1996, 2000) surrogate preserving
    power spectrum & exact marginal distribution
"""


def iaaft_temporal(data, max_iter=200, tol=1e-8):
    x = np.asarray(data)
    n = len(x)

    # Sorted original values (for rank remapping)
    sorted_x = np.sort(x)

    # Original Fourier amplitude spectrum
    fft_orig = fft.rfft(x)
    amplitude = np.abs(fft_orig)

    # Initial surrogate: random shuffle
    surrogate = np.random.permutation(x)

    prev_error = np.inf

    for _ in range(max_iter):
        # Enforce power spectrum
        fft_surr = fft.rfft(surrogate)
        phases = np.angle(fft_surr)
        fft_new = amplitude * np.exp(1j * phases)
        surrogate = fft.irfft(fft_new, n=n)

        # Enforce marginal distribution
        ranks = surrogate.argsort().argsort()
        surrogate = sorted_x[ranks]

        # Convergence check
        current_amp = np.abs(fft.rfft(surrogate))
        error = np.linalg.norm(current_amp - amplitude)

        if abs(prev_error - error) < tol:
            break

        prev_error = error

    else:  # Only executes if the loop did not break
        warnings.warn(
            f"IAAFT surrogate did not converge after {max_iter} iterations"
            f"(final error={error:.3e})"
        )

    return surrogate


def method2b_(data, realizations=1, path='./1/', variable='xxx'):
    data = np.asarray(data)
    assert data.ndim == 1, "Temporal null models require 1D time series"

    generated_number_length = ews.generated_number_length(realizations)

    for realization in range(realizations):
        generated_dataset = iaaft_temporal(data)

        generated_number_string = 'm2bg' + str(realization).zfill(generated_number_length)
        dir_name = os.path.join(path, generated_number_string)

        os.makedirs(dir_name, exist_ok=True)

        fname = ews.file_name_str(variable.name, cfg.number_of_timesteps_weekly)
        fpath = os.path.join(dir_name, fname)
        np.savetxt(fpath + '.numpy.txt', generated_dataset)
