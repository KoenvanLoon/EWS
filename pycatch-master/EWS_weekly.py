# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (c) 2026 Koen van Loon
#
# See the LICENSE file in the repository root for full license text.

"""
EWS - Early Warning Signals
EWS_weekly

@authors: KoenvanLoon & TijmenJanssen
"""

# ============================================================
# Imports
# ============================================================

from pcraster import *
import numpy as np
import os
import time

import EWSPy as ews
import EWS_configuration as cfg
import EWS_null_temporal_weekly as temp_NULL
import EWS_null_spatial_weekly as spat_NULL
import EWS_StateVariables as ews_sv

import sys
sys.path.append("./pcrasterModules/")


# ============================================================
# Configuration & globals
# ============================================================

# State variables for EWS calculations
"""
Variables (state variables) can be both 'ews_sv.variables_weekly' or 'ews_sv.variables_hourly' for calculating
early-warning signals for the week or hour model respectively. State variables present in EWS_StateVariables.py can
be added through the configuration.

Args:
-----

variables : The state variables for which calculations are done.

"""

variables = ews_sv.variables_weekly


# Spatial interval
"""
The spatial interval differs if a cutoff point is selected or not. If there is a cutoff point, no calculations are done
on spatial datasets after this point.

Args:
-----

spatial_ews_interval : 2D numpy array containing the time steps at which a spatial dataset was created.

"""

if not cfg.cutoff:
    spatial_ews_interval = np.arange(cfg.interval_map_snapshots, cfg.number_of_timesteps_weekly +
                                     cfg.interval_map_snapshots, cfg.interval_map_snapshots)
else:
    spatial_ews_interval = np.arange(cfg.interval_map_snapshots, cfg.cutoff_point + cfg.interval_map_snapshots,
                                     cfg.interval_map_snapshots)


# ============================================================
# File loading functions
# ============================================================

# Loading temporal data file(s)
"""
Loads files containing temporal data.

Args:
-----

variable : The state variable from the variable class presented in EWS_StateVariables.py 

path : str, the filepath where the original dataset can be found.

Returns:
-----

state_variable_timeseries : np.ndarray
                The timeseries containing the temporal data, 1D array if data represents a single signal, 2D array if
                multiple locations are stored.

EWS_calculations : bool
                True if data file was found and loaded successfully.

"""


def temporal_data_file_loading(variable, path='./1/'):
    state_variable_timeseries = []
    EWS_calculations = True
    if variable.datatype == 'numpy':
        file_name = ews.file_name_str(variable.name, cfg.number_of_timesteps_weekly)
        fpath = os.path.join(path, file_name + ".numpy.txt")
        if os.path.exists(fpath):
            state_variable_timeseries = np.loadtxt(path + file_name + ".numpy.txt")
        else:
            print(f"{file_name}.numpy.txt not found in {path}")
            EWS_calculations = False
    else:
        print(f"Datatype for {variable.name} currently not supported.")
        EWS_calculations = False

    return state_variable_timeseries, EWS_calculations


# Loading spatial data file(s)
"""
Loads files containing spatial data.

Args:
-----

variable : The state variable from the variable class presented in EWS_StateVariables.py 

path : str, the filepath where the original dataset can be found.

Returns:
-----

state_variable_snapshots : the snapshots containing the spatial data.

EWS_calculations : bool, whether the datafiles are found and if EWS calculations can be performed.

"""


def spatial_data_file_loading(variable, path='./1/'):
    state_variable_snapshots = [0.0] * len(spatial_ews_interval)
    EWS_calculations = True
    if variable.datatype == 'numpy':
        for k, timestep in enumerate(spatial_ews_interval):
            file_name = ews.file_name_str(variable.name, timestep)
            fpath = os.path.join(path, file_name + ".numpy.txt")
            if os.path.exists(fpath):
                state_variable_snapshots[k] = np.loadtxt(path + file_name + '.numpy.txt')
            else:
                print(f"{file_name}.numpy.txt not found in {path}.")
                EWS_calculations = False

    elif variable.datatype == 'map':
        for k, timestep in enumerate(spatial_ews_interval):
            file_name = ews.file_name_str(variable.name, timestep)
            fpath = os.path.join(path, file_name)
            if os.path.exists(fpath):
                state_variable_snapshots[k] = pcr2numpy(readmap(path + file_name), np.NaN)
            else:
                print(f"{file_name} not found in {path}.")
                EWS_calculations = False
    else:
        print(f"Datatype for {variable.name} currently not supported.")
        EWS_calculations = False

    return state_variable_snapshots, EWS_calculations


# ============================================================
# Windowing & preprocessing
# ============================================================

# Time series to time windows
"""
Divide a 1D time series into overlapping windows using NumPy stride tricks.

Args:
-----

timeseries : np.ndarray
                A 1D numpy array containing the time series.
window_size : int
                Number of data points per window.

window_overlap : int
                Number of data points shared between consecutive windows.

Returns:
-----

windows : np.ndarray
                2D numpy array of shape (n_windows, window_size). If the time series is shorter than window_size,
                an empty array of shape (0, window_size) is returned.

    ! - Note that the amount of data points in 'windows' does not need to be equal to the amount of data points in 
    'timeseries' due to the possibility of dropping data points if they do not fill the last time window completely.

"""


def time_series2time_windows(timeseries, window_size=100, window_overlap=0):
    # actual step size between windows
    actual_window_overlap = window_size - window_overlap
    if window_overlap >= window_size:
        raise ValueError

    sh = timeseries.size - window_size + 1, window_size
    st = (timeseries.strides[0], timeseries.strides[0])
    if timeseries.size < window_size:
        return np.empty((0, window_size))
    if window_overlap != 0:
        return np.lib.stride_tricks.as_strided(timeseries, strides=st, shape=sh)[0::actual_window_overlap]

    return np.lib.stride_tricks.as_strided(timeseries, strides=st, shape=sh)[0::window_size]


# Dividing timeseries into windows
"""
Divides a timeseries (optionally of multiple locations) (2D or 3D numpy array) into multiple windows.

Args:
-----

variable : StateVariable
            State variable configuration (contains window_size and window_overlap).
state_variable_timeseries : np.ndarray
            The timeseries - 1D (time,) or 2D (time, n_locations) - of the state variable.

Returns:
-----

stack_of_windows : np.ndarray
            2D array of stacked windows - 1D input results in shape (n_windows, window_size), multi-location input
            results in shape (n_windows * n_locations, window_size)
n_dim : int
            Dimensionality of the original input (1 or 2D).
n_windows , window_length : int, int
            Number of windows per location and length of each window respectively.

"""


def window_stacker(variable, state_variable_timeseries):
    n_dim = state_variable_timeseries.ndim

    if n_dim == 1:
        if cfg.cutoff:
            state_variable_timeseries = state_variable_timeseries[:cfg.cutoff_point]

        stack_of_windows = time_series2time_windows(state_variable_timeseries, variable.window_size,
                                                    variable.window_overlap)
        n_windows, window_length = stack_of_windows.shape
        n_locations = 1

    else:
        stack_of_windows = []

        for timeseries in state_variable_timeseries.T:
            ts = timeseries[:cfg.cutoff_point] if cfg.cutoff else timeseries
            stack_of_windows.append(time_series2time_windows(ts, variable.window_size, variable.window_overlap))

        stack_of_windows = np.asarray(stack_of_windows)
        n_locations, n_windows, window_length = stack_of_windows.shape

        # Flatten for EWS calculation
        stack_of_windows = stack_of_windows.reshape(-1, window_length)

    return stack_of_windows, n_dim, n_windows, window_length, n_locations


# ============================================================
# Core EWS calculation logic
# ============================================================

# Calculating and saving EWS
"""
Calculates early-warning signals and saves the results.

Args:
-----

variable : The state variable from the variable class presented in EWS_StateVariables.py

data : The spatial or temporal data from the model.

method_name : str, element of the name under which the EWS is saved which refers to the dimension (s for spatial, t for
    temporal) and the statistic/method (e.g. mn for mean).

method_function : function, selects the statistic/method used to calculate the (possible) EWS.

path : str, the filepath where the original dataset can be found.

n_dim : int, the number of dimensions of the original timeseries.

n_windows , window_length : x and y component of a stack of windows for multiple locations before flattening.

"""

temporal_ews_functions = [
    ('.t.mn', ews.temporal_mean),
    ('.t.std', ews.temporal_std),
    ('.t.var', ews.temporal_var),
    ('.t.cv', ews.temporal_cv),
    ('.t.skw', ews.temporal_skw),
    ('.t.krt', ews.temporal_krt),
    ('.t.acr', ews.temporal_autocorrelation),
    ('.t.AR1', ews.temporal_AR1),
    ('.t.rr', ews.temporal_returnrate),
    ('.t.dfa', ews.temporal_dfa),
]

spatial_ews_functions = [
    ('.s.mn', ews.spatial_mean),
    ('.s.std', ews.spatial_std),
    ('.s.var', ews.spatial_var),
    ('.s.skw', ews.spatial_skw),
    ('.s.krt', ews.spatial_krt),
    ('.s.mI', ews.spatial_corr),
    ('.s.ps', ews.spatial_power_spec),
    ('.s.dft', ews.spatial_DFT),
]


def ews_calc_and_save(variable, data, method_name, method_function, path='./1/',
                      n_dim=1, n_windows=1, n_locations=1):

    fpath = os.path.join(path, variable.name + method_name)
    signal = method_function(data)

    # Assumes no NaNs or infs
    if n_dim > 1:
        signal = signal.reshape(n_locations, n_windows)

    np.savetxt(fpath + '.numpy.txt', signal)

# Initializing calculating and saving EWS for temporal data
"""
Initializes calculating early-warning signals and saving the results for temporal data.

Args:
-----

variable : The state variable from the variable class presented in EWS_StateVariables.py

state_variable_timeseries : The temporal data from the model.

path : str, the filepath where the original dataset can be found.

"""


def temporal_ews_calculations(variable, state_variable_timeseries, path='./1/'):

    stack_of_windows, n_dim, n_windows, window_length, n_locations = window_stacker(variable, state_variable_timeseries)

    for suffix, func in temporal_ews_functions:
        ews_calc_and_save(variable, stack_of_windows, suffix, func, path=path, n_dim=n_dim, n_windows=n_windows, n_locations=n_locations)

    # Temporal cond. het.
    fpath = os.path.join(path, variable.name + '.t.coh')
    save_p = True

    # Save (statistic, p-value) for 1D
    if save_p and n_dim == 1:
        temporal_statistic = np.array(ews.temporal_cond_het(stack_of_windows))

    # Save only statistic for spatial/multi-loc
    else:
        temporal_statistic, _ = ews.temporal_cond_het(stack_of_windows)  # _ is the p-value of the test, not saved
        if n_dim > 1:
            temporal_statistic = temporal_statistic.reshape(n_locations, n_windows)

    np.savetxt(fpath + '.numpy.txt', temporal_statistic)


# Initializing calculating and saving EWS for spatial data
"""
Initializes calculating early-warning signals and saving the results for spatial data.

Args:
-----

variable : The state variable from the variable class presented in EWS_StateVariables.py

state_variable_maps : The spatial data from the model.

path : str, the filepath where the original dataset can be found.

"""


def spatial_ews_calculations(variable, state_variable_maps, path='./1/'):
    for suffix, func in spatial_ews_functions:
        ews_calc_and_save(variable, state_variable_maps, suffix, func, path=path)


# ============================================================
# Dataset generation (null models)
# ============================================================

# Generate datasets (initial)
"""
Initializes dataset generation. Datasets are generated for method(s) selected in the configuration when 
generate_dummy_datasets is set to True. 

Args:
-----

variable : The state variable from the variable class presented in EWS_StateVariables.py 

path : str, the filepath where the original dataset can be found.

nr_realizations : int, the number of datasets generated.

method1 : bool, selects whether this method is utilized.

method2 : bool, selects whether this method is utilized.

method3 : bool, selects whether this method is utilized.

method2b : bool, selects whether this method is utliized.

"""


def generate_datasets_init(variable, path='./1/', nr_realizations=1, method1=False, method2=False, method3=False, method2b=False):

    if variable.temporal:
        generate_temporal_datasets_init(method1, method2, method3, method2b, nr_realizations, path, variable)

    if variable.spatial:
        generate_spatial_datasets_init(method1, method2, method3, method2b, nr_realizations, path, variable)


def generate_spatial_datasets_init(method1, method2, method3, method2b, nr_realizations, path, variable):
    state_variable_snapshots, files_present = spatial_data_file_loading(variable, path=path)
    if files_present:
        state_variable_snapshots = np.asarray(state_variable_snapshots)

        # Generate dummy datasets
        # Detrending: 'None', 'MeanLinear', 'Gaussian'
        generate_datasets_main(variable, state_variable_snapshots, spat_NULL.detrend_, nr_realizations=nr_realizations,
                               path=path)
        if method1:
            generate_datasets_main(variable, state_variable_snapshots, spat_NULL.method1_,
                                   nr_realizations=nr_realizations, path=path)
        if method2:
            generate_datasets_main(variable, state_variable_snapshots, spat_NULL.method2_,
                                   nr_realizations=nr_realizations, path=path)
        if method3:
            generate_datasets_main(variable, state_variable_snapshots, spat_NULL.method3_,
                                   nr_realizations=nr_realizations, path=path)
        if method2b:
            generate_datasets_main(variable, state_variable_snapshots, spat_NULL.method2b_,
                                   nr_realizations=nr_realizations, path=path)


def generate_temporal_datasets_init(method1, method2, method3, method2b, nr_realizations, path, variable):
    state_variable_timeseries, files_present = temporal_data_file_loading(variable, path=path)
    if files_present:
        if state_variable_timeseries.ndim == 1:
            # Detrending: 'None', 'Gaussian'
            state_variable_timeseries = generate_datasets_main(variable, state_variable_timeseries,
                                                               temp_NULL.detrend_, nr_realizations=nr_realizations,
                                                               path=path)
            # Generate dummy datasets
            if method1:
                generate_datasets_main(variable, state_variable_timeseries, temp_NULL.method1_,
                                       nr_realizations=nr_realizations, path=path)
            if method2:
                generate_datasets_main(variable, state_variable_timeseries, temp_NULL.method2_,
                                       nr_realizations=nr_realizations, path=path)
            if method3:
                generate_datasets_main(variable, state_variable_timeseries, temp_NULL.method3_,
                                       nr_realizations=nr_realizations, path=path)
            if method2b:
                generate_datasets_main(variable, state_variable_timeseries, temp_NULL.method2b_,
                                       nr_realizations=nr_realizations, path=path)
        else:
            print(f"Multiple dimensions are currently not supported for generated datasets, so no datasets are being "
                  f"generated for {variable.name}.")


# Generate datasets (main)
"""
Initializes dataset generation. Datasets are generated for method(s) selected in the configuration when 
generate_dummy_datasets is set to True. 

Args:
-----

variable : The state variable from the variable class presented in EWS_StateVariables.py

state_variable : The data for which datasets are to be generated. Can be either temporal or spatial data.

method : function, either detrend_, method1_, method2_, method3_, or method2b_ from the spatial or temporal null models.

nr_realizations : int, the number of datasets generated.

path : str, the filepath where the original dataset can be found.

Rerturns:
-----

detrended_data : Optional return, only returns when method==detrend_ for time series.

"""


def generate_datasets_main(variable, state_variable, method, nr_realizations=1, path='./1/'):
    print(f"Started generating dataset(s) for {variable.name} using {method.__name__}")
    result = method(state_variable, realizations=nr_realizations, path=path, variable=variable)
    print(f"Finished generating dataset(s) for {variable.name} using {method.__name__} \n")
    if method is temp_NULL.detrend_:
        return result
    return None


# Calculate EWS generated datasets (initial)
"""
Initializes calculation of generated datasets by passing them to the ews_calculations_main() function. 

Args:
-----

variable : The state variable from the variable class presented in EWS_StateVariables.py 

path : str, the filepath where the original dataset can be found.

nr_realizations : int, the number of datasets generated.

timer_on : bool, selects whether calculation time is shown in the console.

method : function, selects which method (1, 2, 3, 2b) is utilized.

"""


def ews_calculations_generated_datasets_init(variable, path='./1/', nr_realizations=1, timer_on=False, method1=False,
                                             method2=False, method3=False, method2b=False):
    generated_number_length = ews.generated_number_length(nr_realizations)

    if cfg.save_detrended_data and variable.temporal:
        ews_calculations_generated_datasets_main(variable, 'dtr', gen_nr_len=generated_number_length, path=path,
                                                 nr_realizations=1, timer_on=timer_on)
    if method1:
        ews_calculations_generated_datasets_main(variable, 'm1g', gen_nr_len=generated_number_length, path=path,
                                                 nr_realizations=nr_realizations, timer_on=timer_on)
    if method2:
        ews_calculations_generated_datasets_main(variable, 'm2g', gen_nr_len=generated_number_length, path=path,
                                                 nr_realizations=nr_realizations, timer_on=timer_on)
    if method3:
        ews_calculations_generated_datasets_main(variable, 'm3g', gen_nr_len=generated_number_length, path=path,
                                                 nr_realizations=nr_realizations, timer_on=timer_on)
    if method2b:
        ews_calculations_generated_datasets_main(variable, 'm2bg', gen_nr_len=generated_number_length, path=path,
                                                 nr_realizations=nr_realizations, timer_on=timer_on)


# Calculate EWS generated datasets (main)
"""
Initializes calculation of generated datasets by passing them to the ews_calculations_init() function.

Args:
-----

variable : The state variable from the variable class presented in EWS_StateVariables.py 

path : str, the filepath where the original dataset can be found.

nr_realizations : int, the number of datasets generated.

method : function, selects which method (1, 2, 3, 2b) is utilized.

"""


def ews_calculations_generated_datasets_main(variable, method, gen_nr_len=4, path='./1/', nr_realizations=1, timer_on=False):
    for realization in range(nr_realizations):
        generated_number_string = method + str(realization).zfill(gen_nr_len) + '/'
        dir_name = os.path.join(path, generated_number_string)
        ews_calculations_init(variable, path=dir_name, timer_on=timer_on)


# ============================================================
# Pipeline control layer
# ============================================================

# Initializing calculating and saving EWS for both spatial and temporal data
"""
Initializes calculating early-warning signals and saving the results for both temporal and spatial data.

Args:
-----

variable : The state variable from the variable class presented in EWS_StateVariables.py

path : str, the filepath where the original dataset can be found.

timer_on : bool, selects whether calculation time is shown in the console.

"""


def ews_calculations_init(variable, path='./1/', timer_on=False):
    if variable.temporal:
        if cfg.mean_timeseries_data:
            ews_calculations_main(variable, temporal_data_file_loading, temporal_ews_calculations, path=path,
                                  timer_on=timer_on)
        else:
            print(f"Mean timeseries data == False in the configuration, could not calculate EWS for "
                  f"{variable.name}.")
    if variable.spatial:
        if cfg.map_data:
            ews_calculations_main(variable, spatial_data_file_loading, spatial_ews_calculations, path=path,
                                  timer_on=timer_on)
        else:
            print(f"Map data == False in the configuration, could not calculate EWS for {variable.name}.")


# Initializing calculating and saving EWS for either spatial and temporal data
"""
Initializes calculating early-warning signals and saving the results for either temporal and spatial data.

Args:
-----

variable : The state variable from the variable class presented in EWS_StateVariables.py

loading_function : function, refers to temporal_data_file_loading() or spatial_data_file_loading().

calculation_function : function, refers to temporal_ews_calculations() or spatial_ews_calculations().

path : str, the filepath where the original dataset can be found.

timer_on : bool, selects whether calculation time is shown in the console.

"""


def ews_calculations_main(variable, loading_function, calculation_function, path='./1/', timer_on=False):
    state_variable, files_present = loading_function(variable, path=path)

    if files_present:
        print(f"Started EWS calculations for {variable.name} in {path}")
        if timer_on:
            start_time = time.time()
            calculation_function(variable, state_variable, path=path)
            end_time = time.time()
            print(f"Elapsed time for EWS calculations for {variable.name} in {path} equals:", end_time - start_time,
                  '\n')
        else:
            calculation_function(variable, state_variable, path=path)


# EWS calculations & optional data generation and EWS calculations for results of the weekly model
"""
Starts calculations, optional data generation & calculations for results of the weekly model. Takes no arguments and 
returns nothing, as settings from the configuration are used instead. Calculations are saved on disk.

"""


def EWS_weekly_calculations():
    start_time = time.time()
    for realization in range(1, cfg.nrOfSamples + 1):
        for variable in variables:
            ews_calculations_init(variable, path=f'./{realization}/', timer_on=True)
            if cfg.generate_dummy_datasets:
                generate_datasets_init(variable, path=f'./{realization}/', nr_realizations=cfg.nr_generated_datasets,
                                       method1=cfg.method_1, method2=cfg.method_2, method3=cfg.method_3,
                                       method2b=cfg.method_2b)
                ews_calculations_generated_datasets_init(variable, path=f'./{realization}/',
                                                         nr_realizations=cfg.nr_generated_datasets,
                                                         timer_on=True, method1=cfg.method_1, method2=cfg.method_2,
                                                         method3=cfg.method_3, method2b=cfg.method_2b)
    end_time = time.time() - start_time
    print(f"Total elapsed time equals: {end_time} seconds")


# EWS calculations & optional data generation and EWS calculations for results of the hourly model
"""
Starts calculations, optional data generation & calculations for results of the hourly model. Takes no arguments and 
returns nothing, as settings from the configuration are used instead. Calculations are saved on disk.

"""


def EWS_hourly_calculations():
    start_time = time.time()
    for i in range(cfg.stepsTotal):
        fpath = str(i).zfill(2)
        for variable in variables:
            ews_calculations_init(variable, path=f'./h{fpath}/', timer_on=True)
    end_time = time.time() - start_time
    print(f"Total elapsed time equals: {end_time} seconds")


if __name__ == "__main__":
    EWS_weekly_calculations()
