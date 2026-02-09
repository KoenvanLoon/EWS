"""
EWS - Early Warning Signals
EWS Tests

@authors: KoenvanLoon
"""

import numpy as np
import os
import scipy.stats
import matplotlib.pyplot as plt
from scipy import ndimage, signal

import EWSPy as ews
import EWS_configuration as cfg
import EWS_StateVariables as ews_sv

tau_treshold = 0.  # heuristic, exploratory


# Helper function for consistent style for plots
def apply_ews_plot_style(ax, grid=False):
    if grid:
        ax.grid(which='major', linestyle='--', alpha=0.5)
        ax.grid(which='minor', linestyle=':', alpha=0.25)
    ax.minorticks_on()
    ax.tick_params(direction='out', which='both', length=4, width=1)
    ax.spines['top'].set_visible(False) # Cleans top spine for less clutter

# State variables for EWS
"""
State variables present in EWS_StateVariables.py can be added through EWS_configuration.py

Args:
----

variables : list of either the hourly or weekly variables from EWS_StateVariables as specified in EWS_configuration.

names : list of the names (full and shortened) of the variables specified in the hourly or weekly variables.

"""

variables = ews_sv.variables_weekly

names = []
for variable in variables:
    names.append([f'{variable.full_name} as {variable.name}'])


# Early warning signals names
"""
Early warning signals (both temporal and spatial) which are included in EWSPy.py

Args:
----

ews_temporal_signals : dict of shorthand notation and name of the temporal early warning signals.

ews_spatial_signals : dict of shorthand notation and name of the spatial early warning signals.

"""

ews_temporal_signals = {'mn': "mean", 'std': "standard deviation", 'var': "variance",
                        'cv': "coefficient of variation", 'skw': "skewness", 'krt': "kurtosis",
                        'dfa': "detrended fluctuation analysis", 'acr': "autocorrelation", 'AR1': "AR1",
                        'rr': "return rate", 'coh': "conditional heteroskedasticity", 'timeseries': "timeseries",
                        'gauss': "gauss"}
ews_spatial_signals = {'mn': "mean", 'std': "standard deviation", 'var': "variance", 'skw': "skewness", 'krt': "kurtosis",
                       'mI': "Moran's I", 'dft': "discrete Fourier transform", 'ps': "power spectrum"}
ews_function_map = {'mn': 'mean', 'std': 'std', 'var': 'var', 'cv': 'cv', 'skw': 'skw', 'krt': 'krt', 'dfa': 'dfa',
                    'acr': 'autocorrelation', 'AR1': 'AR1', 'mI': 'mI', 'dft': 'dft', 'ps': 'ps', 'rr': 'returnrate',
                    'coh': 'cond_het', 'timeseries': 'timeseries', 'gauss': 'gauss'}


def get_temporal_ews_function(sum_stat):
    if sum_stat not in ews_function_map:
        raise ValueError(f"Unknown EWS shorthand '{sum_stat}")

    func_name = f"temporal_{ews_function_map[sum_stat]}"
    if not hasattr(ews, func_name):
        raise AttributeError(f"EWSPy has no function '{func_name}'")

    return getattr(ews, func_name)


# Which EWS support Kendall tau trend analysis
ews_supports_tau = {
    'mn': True,
    'std': True,
    'var': True,
    'cv': True,
    'skw': True,
    'krt': True,
    'dfa': True,    # after extraction
    'AR1': True,
    'mI': True,
    'dft': True,
    'ps': True,

    # temporal but not trend-testable
    'acr': False,
    'rr': False,
    'coh': False,
    'timeseries': False,
    'gauss': False,
}



def extract_ews_data(X, ews_short_name):
    X = np.asarray(X)
    assert X.ndim <= 2, (f"EWS '{ews_short_name}' returned unexpected array shape {X.shape}")
    if ews_short_name == 'coh':
        if X.ndim == 2:
            return X[0, :]
    if ews_short_name == 'dfa':
        if X.ndim == 2:
            return X[-1, :]
    return X

# Kendall tau stats
"""
Returns the Kendall tau value and its significance (p-value).

Args:
----

state_variable : The state variable of interest.

sum_stat : str, the summary statistic for which the Kendall tau value is calculated.
 
comp2 : str, either 'Same' or 'Forcing', sets the comparison to time (window index) same state variable or the forcing 
    (grazing) rate

path : str, path where inputs from the hourly/weekly model are stored.

Returns:
----

tau : float, the Kendall tau value (rank correlation coefficient).

p : float, the p-value (significance) of the calculated Kendall tau value.

"""


def kendalltau_stats(state_variable, sum_stat, comp2='Same', path='./1/'):
    if not ews_supports_tau.get(sum_stat, False):
        print(f"[INFO] Kendall tau not defined for EWS '{sum_stat}'.")
        return np.NaN, np.NaN

    dim = None
    if state_variable.temporal:
        dim = '.t.'
    elif state_variable.spatial:
        dim = '.s.'

    tau, p = np.NaN, np.NaN
    fdict = os.path.join(path + state_variable.name + dim)
    if os.path.exists(fdict + sum_stat + '.numpy.txt'):
        X = np.loadtxt(fdict + sum_stat + '.numpy.txt')
        X = extract_ews_data(X, sum_stat)

        Y = None
        if comp2 == 'Same': # Dakos et al., 2008
            Y = np.arange(len(X))

        elif comp2 == 'Forcing':  # Dakos et al, 2011 - Does not work if window sizes are different for the forcing & SV
            Y = np.loadtxt('./1/gA.t.mn.numpy.txt')

        tau, p = scipy.stats.kendalltau(X, Y, nan_policy='propagate')

        if np.isnan(tau):
            tau_omit, p_omit = scipy.stats.kendalltau(X, Y, nan_policy='omit')
            n_valid = np.sum(np.isfinite(X) & np.isfinite(Y))

            coverage = n_valid / len(X)

            print(
                f"[INFO] Kendall τ undefined with propagate for {sum_stat};\n"
                f"({n_valid}/{len(X)} valid points - coverage: {coverage:.1%}).\n"
                f"τ_omit = {tau_omit:.3f}, p_omit = {p_omit:.3g}\n"
            )

    return tau, p


# Kendall tau stats for dummy data
"""
Returns the Kendall tau value and its significance (p-value).

Args:
----

state_variable : The state variable of interest.

sum_stat : str, the summary statistic for which the Kendall tau value is calculated.

method : str, either 'm1g', 'm2g', or 'm3g', the dummy data generation method for which the Kendall tau value is 
    calculated.
 
comp2 : str, either 'Same' or 'Forcing', sets the comparison to the time (window index) of the same state variable or the forcing 
    (grazing) rate

path : str, path where inputs from the hourly/weekly model are stored.

Returns:
----

tau : array, the Kendall tau values (rank correlation coefficient) for each realization of dummy data.

p : array, the p-value (significance) of the calculated Kendall tau values for each realization of dummy data.

"""


def kendalltau_stats_dummy(state_variable, sum_stat, method='m1g', comp2='Same', path='./1/'):
    if not ews_supports_tau.get(sum_stat, False):
        return [np.NaN] * cfg.nr_generated_datasets, [np.NaN] * cfg.nr_generated_datasets

    dim = None
    if state_variable.temporal:
        dim = '.t.'
    elif state_variable.spatial:
        dim = '.s.'

    generated_number_length = ews.generated_number_length(cfg.nr_generated_datasets)

    taurray = [np.NaN] * cfg.nr_generated_datasets
    parray = [np.NaN] * cfg.nr_generated_datasets
    for realization in range(cfg.nr_generated_datasets):
        base = os.path.join( path, f"{method}{str(realization).zfill(generated_number_length)}")

        fdict = os.path.join(base, f"{state_variable.name}{dim}")
        fname = fdict + sum_stat + ".numpy.txt"

        if realization == 0:
            print("Example surrogate path being checked:")
            print(fname)

        if os.path.exists(fname):
            X = np.loadtxt(fname)
            X = extract_ews_data(X, sum_stat)

            Y = None
            if comp2 == 'Same': # Dakos et al., 2008
                Y = np.arange(len(X))

            elif comp2 == 'Forcing':  # Dakos et al, 2011 - Does not work if window sizes are different for the forcing & SV, detrending == Gaus
                Y = np.loadtxt('./1/gA.t.mn.numpy.txt')

            taurray[realization], parray[realization] = scipy.stats.kendalltau(X, Y, nan_policy='propagate')

    return taurray, parray


# Real system trend
"""
Plots the EWS indicator for the real system and quantifies its trend.
This function visualizes the EWS summary statistic over time (window index) and annotates the Kendall tau trend
    statistic and p-value.

Args:
----

state_variable : The state variable of interest.

sum_stat : str, the summary statistic for which the Kendall tau value is calculated.

comp2 : str, either 'Same' or 'Forcing', sets the comparison to time (window index) same state variable or the forcing 
    (grazing) rate

path : str, path where inputs from the hourly/weekly model are stored.

Returns:
----

Line plot with Kendall tau and p-value annotation.
"""


def plot_real_trend(state_variable, sum_stat, comp2='Same', path='./1/'):
    dim = '.t.' if state_variable.temporal else '.s.'
    fname = os.path.join(path, state_variable.name + dim + sum_stat + '.numpy.txt')

    if not os.path.exists(fname):
        raise FileNotFoundError(fname)

    X = np.loadtxt(fname)
    X = extract_ews_data(X, sum_stat)
    t = np.arange(len(X))

    tau, p = kendalltau_stats(
        state_variable=state_variable,
        sum_stat=sum_stat,
        comp2=comp2,
        path=path
    )

    fig, ax = plt.subplots()
    ax.plot(t, X, lw=cfg.EWS_linewidth, color=cfg.EWS_colour_cycle[0])
    apply_ews_plot_style(ax, grid=True)
    ax.text(0.7, 0.95, f"Kendall τ = {tau:.3f}\np = {p:.3g}", transform=ax.transAxes, va='top',
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none'))

    print("Save the plot as a .pdf and .svg? [Y/n]")
    save_plot = input()
    if save_plot == 'Y' or save_plot == 'y':
        fig.text(0.995, 0.005, "EWSPy - KvL", ha='right', va='bottom', fontsize=6, alpha=0.25)
        for fmt in ["svg", "pdf"]:
            plt.savefig(path + f"plot_real_trend_{state_variable.full_name}_{sum_stat}.{fmt}", format=fmt, dpi=300)

    plt.show()


# Null model
"""
Compares the observed Kendall tau trend to a null-model distribution.

Plots a histogram of Kendall tau values obtained from surrogate (null) datasets and overlays the observed tau from the
real system.

Args:
----

state_variable : The state variable of interest.

sum_stat : str, the summary statistic for which the Kendall tau value is calculated.

comp2 : str, either 'Same' or 'Forcing', sets the comparison to time (window index) same state variable or the forcing 
    (grazing) rate

path : str, path where inputs from the hourly/weekly model are stored.

method : str, either 'm1g', 'm2g', or 'm3g', the dummy data generation method for which the Kendall tau value is 
    calculated.

Returns:
----

Line plot with Kendall tau and p-value annotation.
"""


def plot_null_model(state_variable, sum_stat, comp2='Same', path='./1/', method='m1g'):
    if not ews_supports_tau.get(sum_stat, False):
        raise ValueError(f"Null-model tests are not meaningful for EWS '{sum_stat}'.")

    tau_obs, _ = kendalltau_stats(
        state_variable=state_variable,
        sum_stat=sum_stat,
        comp2=comp2,
        path=path
    )

    tau_null, _ = kendalltau_stats_dummy(
        state_variable=state_variable,
        sum_stat=sum_stat,
        method=method,
        comp2=comp2,
        path=path
    )

    tau_null = np.array(tau_null, dtype=float)

    # Remove invalid surrogate realizations (Dakos et al. convention - trust me, it was necessary)
    tau_null_valid = tau_null[np.isfinite(tau_null)]

    print(f"Valid surrogates: {tau_null_valid.size} / {tau_null.size}")

    if tau_null_valid.size == 0:
        raise ValueError(
            f"No valid Kendall τ values found for null model '{method}'. "
            "Check surrogate generation or EWS output files."
        )

    fig, ax = plt.subplots()

    # Freedman-Diaconis histogram binning (robust to skewed null distributions)
    N = tau_null_valid.size
    q25, q75 = np.percentile(tau_null_valid, [25, 75])
    iqr = q75 - q25

    if iqr > 0:
        bin_width = 2 * iqr * N ** (-1/3)
        bins = int(np.ceil((tau_null_valid.max() - tau_null_valid.min()) / bin_width))
    else:
        bins = int(np.sqrt(N))  # fallback. Yes, this is also a valid method - but FD is the better flex.

    ax.hist(tau_null_valid, bins=bins, edgecolor='black', color=cfg.EWS_colour_cycle[1], alpha=0.85)

    ax.axvline(tau_obs, color=cfg.EWS_colour_cycle[3], lw=cfg.EWS_linewidth, label=r'$\tau_{\mathrm{obs}}$')

    q95 = np.nanpercentile(tau_null_valid, 95)
    ax.axvline(q95, color='k', ls='--', lw=1.5, label=r'$\tau_{0.95}$')

    ax.set_xlabel(r'Kendall rank correlation ($\tau$)')
    ax.set_ylabel("Frequency")
    ax.set_title(f"Null-model test ({method})\n" f"{state_variable.full_name} – {ews_temporal_signals.get(sum_stat, sum_stat)}")

    ax.legend(loc='upper right')
    apply_ews_plot_style(ax, grid=True)

    print("Save the plot as a .pdf and .svg? [Y/n]")
    save_plot = input()
    if save_plot == 'Y' or save_plot == 'y':
        fig.text(0.995, 0.005, "EWSPy - KvL", ha='right', va='bottom', fontsize=6, alpha=0.25)
        for fmt in ["svg", "pdf"]:
            plt.savefig(path + f"plot_null_model_{method}_{state_variable.full_name}_{sum_stat}.{fmt}", format=fmt, dpi=300)

    plt.show()


# Sensitivity
"""
Sensitivity analysis of EWS trends across parameter combinations. Computes Kendall tau between an EWS indicator and time
    for each parameter combination and visualizes the result as a heatmap.
    
Args:
-----

timeseries : A 2D numpy array containing data points of a early-warning signal.

ews_function : -

x_values : array, x coords.

y_values : array, y coords.

window_overlap : The number (int) of data points in the window equal to the last data points of the previous time 
    window.
    
detrend : str, optional, method of detrending used.

xlabel : str, title given to the x-axis.

ylabel : str, title given to the y-axis.

title : str, title given to the heatmap.

Returns:

Plotted heatmap with p-contours.
"""


def plot_sensitivity(timeseries, ews_function, x_values, y_values, window_overlap=0, detrend=None, xlabel='Parameter x',
                     ylabel='Window size', title='Sensitivity analysis', path='./1/'):
    tau_arr = np.zeros((len(y_values), len(x_values)))
    p_arr = np.zeros_like(tau_arr)

    for j, x in enumerate(x_values):
        ts = timeseries.copy()

        if detrend == 'gaussian':
            ts = ts - ndimage.gaussian_filter1d(ts, x)

        for i, window_size in enumerate(y_values):
            windows = window(ts, window_size, window_overlap)
            stat = ews_function(windows)
            t = np.arange(len(stat))

            tau, p = scipy.stats.kendalltau(stat, t)
            tau_arr[i, j] = tau
            p_arr[i, j] = p

    X, Y = np.meshgrid(x_values, y_values)

    fig, ax = plt.subplots(figsize=(8, 5))
    cs = ax.contourf(X, Y, tau_arr, levels=20, cmap='coolwarm')
    fig.colorbar(cs, ax=ax, label="Kendall τ")
    ax.contour(X, Y, p_arr, levels=[0.05], colors='k', linewidths=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    apply_ews_plot_style(ax, grid=False)

    print("Save the plot as a .pdf and .svg? [Y/n]")
    save_plot = input()
    if save_plot == 'Y' or save_plot == 'y':
        fig.text(0.995, 0.005, "EWSPy - KvL", ha='right', va='bottom', fontsize=6, alpha=0.25)
        for fmt in ["svg", "pdf"]:
            plt.savefig(path + f"plot_sensitivity_{ews_function}.{fmt}", format=fmt, dpi=300)

    plt.show()


# Window size bounds as fractions of time series length
def get_window_bounds(ts_length, min_fraction=0.000096, max_fraction=0.5):
    w_min = int(np.ceil(min_fraction * ts_length))
    w_max = int(np.floor(max_fraction * ts_length))
    return max(w_min, 10), max(w_min + 1, w_max)


# Time series to time windows
"""
Divides a time series (2D numpy array) into an array of evenly sized time windows (2D numpy arrays). If remaining data-
points do not fill the last time window, they are dropped from the stack of time windows.

Args:
-----

timeseries : A 2D numpy array containing data points of a early-warning signal.

window_size : The size (int) of the windows into which the time series is to be divided.

window_overlap : The number (int) of data points in the window equal to the last data points of the previous time 
    window.

Returns:
-----

view : A 3D numpy array containing evenly sized time windows (2D numpy arrays).

    ! - Note that the amount of data points in 'view' does not need to be equal to the amount of data points in 
    'timeseries' due to the possibility of dropping data points if they do not fill the last time window completely.

"""


def window(timeseries, window_size, window_overlap):
    actual_window_overlap = window_size - window_overlap
    sh = (timeseries.size - window_size + 1, window_size)
    st = timeseries.strides * 2
    if window_overlap != 0:
        return np.lib.stride_tricks.as_strided(timeseries, strides=st, shape=sh)[::actual_window_overlap]
    elif window_overlap == 0:
        return np.lib.stride_tricks.as_strided(timeseries, strides=st, shape=sh)[::window_size]


# Windowsize tests
"""
Kendall tau is computed between the EWS indicator and time (window index), testing the robustness of trend detection 
against different window sizes and overlaps.

Args:
----

state_variable : The state variable of interest.

sum_stat : str, the summary statistic for which the windowsize is tested.

path : str, path where inputs from the hourly/weekly model are stored.

method : str, either 'None' or 'MeanLinear'. If 'MeanLinear', a linear detrending is performed for each window.

Returns:
----

Two contourplots with Kendall tau and p-values for different window sizes and overlaps. Optionally saved to disk.

"""


def test_windowsize(state_variable, sum_stat, path='./1/', method='None'):
    if not state_variable.temporal:
        raise ValueError("Window-size tests are only defined for temporal variables.")

    # Loading files
    fname = ews.file_name_str(state_variable.name, cfg.number_of_timesteps_weekly)
    fpath = os.path.join(path + fname)
    ts = np.loadtxt(fpath + '.numpy.txt')

    if cfg.cutoff:
        ts = ts[:cfg.cutoff_point]

    # Select EWS function dynamically
    ews_func = get_temporal_ews_function(sum_stat)

    # Zoom
    ts_len = len(ts)
    w_min, w_max = get_window_bounds(ts_len)

    window_sizes = np.arange(w_min, w_max + 1, max(1, (w_max - w_min) // 50))
    window_overlaps = np.arange(0, int(0.5 * w_min) + 1, max(1, w_min // 20))

    # X and Y coords
    x, y = np.meshgrid(window_overlaps, window_sizes)

    # Calculating and storing tau and p values
    tau_arr = np.zeros_like(x, dtype=float)
    p_arr = np.zeros_like(x, dtype=float)

    for i, window_size in enumerate(window_sizes):
        print(f"Moved to windowsize {window_size}")
        for j, window_overlap in enumerate(window_overlaps):

            if window_overlap >= window_size:
                continue

            windows = window(ts, window_size, window_overlap)

            if method == 'MeanLinear':
                windows = signal.detrend(windows)

            stat = extract_ews_data(ews_func(windows), sum_stat)
            t = np.arange(len(stat))

            tau_arr[i, j], p_arr[i, j] = scipy.stats.kendalltau(stat, t)

    # Making the plots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey='row')
    titles = [r"Kendall $\tau$", r"$p$-value"]

    for ax, Z, title in zip(axs, [tau_arr, p_arr], titles):
        cs = ax.contourf(x, y, Z, levels=20, cmap='coolwarm')
        ax.contour(x, y, p_arr, levels=[0.05], colors='k')
        ax.set_title(title)
        ax.set_xlabel("Window overlap")
        apply_ews_plot_style(ax, grid=False)
        fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)

    axs[0].set_ylabel("window size")
    fig.suptitle(f"{state_variable.full_name} - {ews_temporal_signals[sum_stat]}\n"
                 "Window-size sensitivity", fontsize=12)

    print("Save the plot as a .pdf & svg? [Y/n]")
    save_plot = input()
    if save_plot == 'Y' or save_plot == 'y':
        fig.text(0.995, 0.005, "EWSPy - KvL", ha='right', va='bottom', fontsize=6, alpha=0.25)
        for fmt in ["svg", "pdf"]:
            plt.savefig(path + f"windowsize_test_{method}_{state_variable.full_name}_{sum_stat}.{fmt}", format=fmt, dpi=300)

    plt.show()


# Windowsize and gaussian filtering tests
"""
Makes a contourplot with Kendall tau and p-values for different window sizes and Gaussian filter sizes.

Args:
----
state_variable : temporal state variable to analyze.

sum_stat : str, short name of the EWS.

path : str, path where inputs from the hourly/weekly model are stored.

Returns:
----

Two contourplots with Kendall tau and p-values for different window sizes and Gaussian filter sizes. Plot optionally
    saved to disk.

"""


def test_windowgauss(state_variable, sum_stat, path='./1/'):
    if not state_variable.temporal:
        raise ValueError("Window-Gaussian tests are only defined for temporal variables.")

    # Loading files
    fname = ews.file_name_str(state_variable.name, cfg.number_of_timesteps_weekly)
    fpath = os.path.join(path + fname)
    ts = np.loadtxt(fpath + '.numpy.txt')

    if cfg.cutoff:
        ts = ts[:cfg.cutoff_point]

    ews_func = get_temporal_ews_function(sum_stat)

    # Zoom
    ts_len = len(ts)
    w_min, w_max = get_window_bounds(ts_len)

    window_sizes = np.arange(w_min, w_max + 1, max(1, (w_max - w_min) // 50))
    gaussian_sigmas = np.arange(0, int(0.25 * w_min) + 1, max(1, w_min // 20))

    # X and Y coords
    x, y = np.meshgrid(gaussian_sigmas, window_sizes)

    # Calculating and storing tau and p values
    tau_arr = np.zeros_like(x, dtype=float)
    p_arr = np.zeros_like(x, dtype=float)

    for j, sigma in enumerate(gaussian_sigmas):
        print(f"Gaussian σ {sigma}/{gaussian_sigmas[-1]}")

        if sigma > 0:
            ts_detr = ts - ndimage.gaussian_filter1d(ts, sigma)
        else:
            ts_detr = ts.copy()

        for i, w in enumerate(window_sizes):
            windows = window(ts_detr, w, 0)
            stat = extract_ews_data(ews_func(windows), sum_stat)
            t = np.arange(len(stat))

            tau_arr[i, j], p_arr[i, j] = scipy.stats.kendalltau(stat, t)

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey='row')
    titles = [r"Kendall $\tau$", r"$p$-value"]

    for ax, Z, title in zip(axs, [tau_arr, p_arr], titles):
        cs = ax.contourf(x, y, Z, levels=20, cmap='coolwarm')
        ax.contour(x, y, p_arr, levels=[0.05], colors='k')
        ax.set_title(title)
        ax.set_xlabel("Gaussian filter width σ")
        apply_ews_plot_style(ax, grid=False)
        fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)

    axs[0].set_ylabel("Window size")
    fig.suptitle(f"{state_variable.full_name} – {ews_temporal_signals[sum_stat]}\n"
                 "Window–Gaussian sensitivity (Dakos-style robustness test)", fontsize=12)

    print("Save the plot as a .pdf and .svg? [Y/n]")
    save_plot = input()
    if save_plot == 'Y' or save_plot == 'y':
        fig.text(0.995, 0.005, "EWSPy - KvL", ha='right', va='bottom', fontsize=6, alpha=0.25)
        for fmt in ["svg", "pdf"]:
            plt.savefig(path + f"window_gauss_test_{state_variable.full_name}_{sum_stat}.{fmt}", format=fmt, dpi=300)

    plt.show()


# User input Tests-plot looper
"""
Loops the function for user inputs for plot maker.

Args:
----

path : str, path where model/EWS_weekly/hourly.py outputs are stored.

Runs the different plot functions in this file for given inputs

"""


def user_input_tests_looper(path='./1/'):
    print("\nAvailable state variables:")
    for variable in variables:
        print(f" - {variable.name}: {variable.full_name}")

    state_name = input("\nEnter state variable short name:")
    try:
        state_variable = next(v for v in variables if v.name == state_name)
    except StopIteration:
        print("Unknown state variable.")
        return

    if state_variable.temporal:
        print("\nAvailable temporal EWS:", list(ews_temporal_signals.keys()))
    else:
        print("\nAvailable spatial EWS:", list(ews_spatial_signals.keys()))

    sum_stat = input("enter EWS short name: ")

    print("\nChoose analysis type:")
    print(" 1 - Plot real-system trend")
    print(" 2 - Null-model test")
    print(" 3 - Window-size sensitivity test")
    print(" 4 - Window–Gaussian sensitivity test")

    choice = input("Enter choice (1–4): ")

    if choice == '1':
        plot_real_trend(
            state_variable=state_variable,
            sum_stat=sum_stat,
            path=path
        )

    elif choice == '2':
        print("Choose null model [m1g, m2g, m3g]:")
        method = input()
        plot_null_model(
            state_variable=state_variable,
            sum_stat=sum_stat,
            method=method,
            path=path
        )

    elif choice == '3':
        test_windowsize(state_variable=state_variable, sum_stat=sum_stat, path=path)

    elif choice == '4':
        test_windowgauss(state_variable=state_variable, sum_stat=sum_stat, path=path)

    else:
        print("Invalid choice.")

    again = input("\nRun another test? [Y/n] ")
    if again.lower() == 'y':
        user_input_tests_looper(path=path)
    elif again.lower() == 'n':
        print("Exited EWS test suite. Goodbye.")
    else:
        print("Invalid input, terminated plotmaker. Goodbye.")


if __name__ == "__main__":
    user_input_tests_looper()
