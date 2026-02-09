"""
EWS - Early Warning Signals
Null models spatial weekly

@authors: KoenvanLoon
"""

import numpy as np
from scipy import fft
from scipy import ndimage
from scipy.sparse import coo_matrix, identity
from scipy.sparse.linalg import spsolve
import os
import EWSPy as ews
import EWS_configuration as cfg
from pcraster import numpy2pcr, report, Scalar

### Null models adapted from (Dakos et al. 2008) ###

# Detrend dataset
"""
Detrends the given dataset making use of either Gaussian filtering or mean-centred detrending, as specified in the 
configuration. Optionally, this data is saved.

!NOTE: Spatial trends are considered nuisance structure rather than part of the EWS.

Args:
-----

data : numpy array, the spatial datasets.

realizations : int, the number of datasets generated. Used for folder name in the case of detrending.

path : str, the filepath where the original dataset can be found.

file_name : str, name of the variable.

Return:
-----

detrended_data : The detrended spatial datasets.

"""


def detrend_(dataset, realizations=1, path='./1/', variable='xxx'):
    dataset = np.asarray(dataset)
    assert dataset.ndim == 3, "Spatial null models require 3D arrays of shape (time, y, x)"

    generated_number_length = ews.generated_number_length(realizations)
    steps = np.arange(cfg.interval_map_snapshots, cfg.number_of_timesteps_weekly + cfg.interval_map_snapshots,
                      cfg.interval_map_snapshots)

    detrended_dataset = [0.0] * len(dataset)

    for k, data in enumerate(dataset):
        assert data.ndim == 2, "Each spatial snapshot must be 2D"
        detrended_data = data.copy()
        if cfg.detrended_method == 'Gaussian':
            gaussian_filter = ndimage.gaussian_filter(data, cfg.detrended_sigma)
            detrended_data -= gaussian_filter
        elif cfg.detrended_method == 'MeanLinear':
            mean = np.nanmean(data)
            detrended_data -= mean
        elif cfg.detrended_method != 'None':
            print("Invalid input for detrending_spat in generate_datasets (EWS_weekly.py). No detrending done.")

        detrended_dataset[k] = detrended_data

        if cfg.save_detrended_data:
            generated_number_string = 'dtr' + str(0).zfill(generated_number_length)
            dir_name = os.path.join(path + generated_number_string)

            if os.path.isdir(dir_name) == False:
                os.makedirs(dir_name)

            fname1 = ews.file_name_str(variable.name, steps[k])
            fpath1 = os.path.join(dir_name, fname1)
            detrended_data_pcr = numpy2pcr(Scalar, detrended_data, np.NaN)
            report(detrended_data_pcr, fpath1)

            gaussian_filter = None
            if cfg.detrended_method == 'Gaussian':
                fname2 = ews.file_name_str(variable.name + 'g', steps[k])
                fpath2 = os.path.join(dir_name, fname2)
                gaussian_filter_pcr = numpy2pcr(Scalar, gaussian_filter, np.NaN)
                report(gaussian_filter_pcr, fpath2)

    return detrended_dataset


# Generate datasets method 1
"""
Generates dataset(s) with similar mean and variance by randomly picking values from the original dataset. In the case
of replace==False, this is similar to shuffling the dataset.

Args:
-----

data : numpy array, the spatial datasets.

realizations : int, the number of datasets generated.

path : str, the filepath where the original dataset can be found.

variable : str, name of the variable.

replace : bool, selects whether new values are picked from the original dataset or the original dataset minus previously
    picked values. Usually set to False to ensure similar mean and variance for smaller datasets.

"""


def method1_(dataset, realizations=1, path='./1/', variable='xxx', replace=False):
    generated_number_length = ews.generated_number_length(realizations)

    data_shape = dataset[0].shape
    steps = np.arange(cfg.interval_map_snapshots, cfg.number_of_timesteps_weekly + cfg.interval_map_snapshots,
                      cfg.interval_map_snapshots)

    for k, data in enumerate(dataset):
        data_new = data.copy()
        data_1d = data_new.ravel()
        for realization in range(realizations):
            generated_dataset_numpy = np.random.choice(data_1d, len(data_1d), replace=replace).reshape(data_shape)
            generated_dataset = numpy2pcr(Scalar, generated_dataset_numpy, np.NaN)

            generated_number_string = 'm1g' + str(realization).zfill(generated_number_length)
            dir_name = os.path.join(path + generated_number_string)

            if os.path.isdir(dir_name) == False:
                os.makedirs(dir_name)

            fname = ews.file_name_str(variable.name, steps[k])
            fpath = os.path.join(dir_name, fname)
            report(generated_dataset, fpath)


# Generate datasets method 2
"""
Generates dataset(s) with similar autocorrelation, mean and variance by generating datasets with the same Fourier 
spectrum and amplitudes.

Args:
-----

data : numpy array, the spatial datasets.

realizations : int, the number of datasets generated.

method : str, either 'None' or 'Detrending', if detrended data is used as input, no further detrending is needed. If
    not-detrended data is used, mean-centred detrending is applied before the Fourier spectrum and amplitudes are calculated,
    with the mean-centred detrend added after the generation of datasets. 

path : str, the filepath where the original dataset can be found.

variable : name of the variable.

replace : bool, selects whether new values are picked from the original dataset or the original dataset minus previously
    picked values.

"""


def method2_(dataset, realizations=1, method='None', path='./1/', variable='xxx', replace=False):
    generated_number_length = ews.generated_number_length(realizations)

    steps = np.arange(cfg.interval_map_snapshots, cfg.number_of_timesteps_weekly + cfg.interval_map_snapshots,
                      cfg.interval_map_snapshots)

    for k, data in enumerate(dataset):
        if method == 'Detrending':
            mean = np.nanmean(data)
            data = data - mean

        fft2_ = fft.fft2(data)
        fft2_mag = np.abs(fft2_)
        fft2_phases = np.angle(fft2_)

        for realization in range(realizations):
            fft2_phases_new = np.random.uniform(-np.pi, np.pi, fft2_phases.shape)
            fft2_phases_new = (fft2_phases_new + np.flipud(np.fliplr(fft2_phases_new))) / 2

            fft2_sym = fft2_mag * (np.cos(fft2_phases_new) + 1j * np.sin(fft2_phases_new))
            generated_dataset_numpy = np.real(fft.ifft2(fft2_sym))

            mean = 0.0
            if method == 'Detrending':
                generated_dataset_numpy += mean

            generated_dataset = numpy2pcr(Scalar, generated_dataset_numpy, np.NaN)

            generated_number_string = 'm2g' + str(realization).zfill(generated_number_length)
            dir_name = os.path.join(path + generated_number_string)

            if os.path.isdir(dir_name) == False:
                os.makedirs(dir_name)

            fname = ews.file_name_str(variable.name, steps[k])
            fpath = os.path.join(dir_name, fname)
            report(generated_dataset, fpath)


# Third method note
"""
Combination of Jon Yearsley (2021). Generate AR1 spatial data (https://www.mathworks.com/matlabcentral/fileexchange/5099-generate-ar1-spatial-data), MATLAB Central File Exchange. Retrieved November 30, 2021.
and Dakos et al. 10.1073/pnas.0802430105
"""


# Sparse rook adjecency matrix or: How I Learned to Stop Worrying and Love the Math
def rook_weight_matrix(ny, nx, rho):
    rows = []
    cols = []
    data = []

    def idx(i, j):
        return i * nx + j

    for i in range(ny):
        for j in range(nx):
            p = idx(i,j)

            if i > 0:       # up
                rows.append(p)
                cols.append(idx(i - 1, j))
                data.append(rho / 4)

            if i < ny - 1:  # down
                rows.append(p)
                cols.append(idx(i + 1, j))
                data.append(rho / 4)

            if j > 0:       # left
                rows.append(p)
                cols.append(idx(i, j - 1))
                data.append(rho / 4)

            if j < nx - 1:  # right
                rows.append(p)
                cols.append(idx(i, j + 1))
                data.append(rho / 4)

    N = ny * nx
    return coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()


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


def method3_(dataset, realizations=1, method='Normal', path='./1/', variable='xxx'):
    dataset = np.asarray(dataset)
    assert dataset.ndim == 3, "Spatial null models require 3D arrays of shape (time, y, x)"

    generated_number_length = ews.generated_number_length(realizations)
    steps = np.arange(cfg.interval_map_snapshots, cfg.number_of_timesteps_weekly + cfg.interval_map_snapshots,
                      cfg.interval_map_snapshots)

    spatial_correlation = ews.spatial_corr(dataset, local=True)

    for k, data in enumerate(dataset):
        Morans_I = spatial_correlation[k]

        if abs(Morans_I) > 0.99:
            print(f"Warning: strong spatial correlation (I={Morans_I:.3f}) at snapshot {k}")

        # If spsolve ever starts acting up, shut it up with adding the following lines:
        # eps = 1e-6
        # if abs(Morans_I) >= 1:
        #     Morans_I = np.sign(Morans_I) * (1 - eps)
        alpha0_1 = np.nanmean(data) * (1 - Morans_I)
        alpha0_2 = np.nanmean(data)

        dim = data.shape
        ny, nx = dim
        N = ny * nx

        W = rook_weight_matrix(ny, nx, Morans_I)
        # M = I - W (W already includes Moran's I (rho) scaling.
        M = identity(N, format="csr") - W

        for realization in range(realizations):

            # Step 1 - generate unit-variance noise
            random_error = np.random.normal(loc=0.0, scale=1.0, size=N)

            raw = spsolve(M, random_error).reshape(dim)

            if method == 'Adjusted':
                raw_mean = alpha0_1
            elif method == 'Normal':
                raw_mean = alpha0_2
            else:
                raise ValueError("Unknown method: choose 'Normal' or 'Adjusted'")

            # Step 2 - variance matching
            raw_var = np.nanvar(raw)
            target_var = np.nanvar(data)

            if raw_var > 0:
                raw *= np.sqrt(target_var / raw_var)

            generated_dataset = raw + raw_mean
            generated_dataset = numpy2pcr(Scalar, generated_dataset, np.NaN)

            generated_number_string = 'm3g' + str(realization).zfill(generated_number_length)
            dir_name = os.path.join(path + generated_number_string)

            if os.path.isdir(dir_name) == False:
                os.makedirs(dir_name)

            fname = ews.file_name_str(variable.name, steps[k])
            fpath = os.path.join(dir_name, fname)
            report(generated_dataset, fpath)
