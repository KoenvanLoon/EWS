"""
EWS - Early Warning Signals
EWS weekly plots

@authors: KoenvanLoon & TijmenJanssen
"""

import numpy as np
import os
from cycler import cycler
import matplotlib.pyplot as plt
from datetime import datetime

import EWS_configuration as cfg
import EWSPy as ews
import EWS_StateVariables as ews_sv


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

timeseries : list of weekly variables from EWS_StateVariables as specified in EWS_configuration used for coupled plots.

names : list of the names (full and shortened) of the variables specified in the hourly or weekly variables.

names_timeseries : list of the names (full and shortened) of the timeseries specified in weekly timeseries variables.

"""

variables = ews_sv.variables_weekly
# variables = ews_sv.variables_hourly

timeseries = ews_sv.variables_weekly

names = []
for variable in variables:
    names.append([f'{variable.full_name} as {variable.name}'])

names_timeseries = []
for ts in timeseries:
    names_timeseries.append([f'{ts.full_name} as {ts.name}'])

# Number of timesteps
"""
Timesteps over which EWS can be calculated if no cutoff point is used (see: EWS_configuration.py)

Args:
----

number of timesteps : either cfg.number_of_timesteps_weekly or cfg.number_of_timesteps_hourly for either the plotting of
    EWS for weekly or hourly model, respectively.

"""

number_of_timesteps = cfg.number_of_timesteps_weekly
# number_of_timesteps = cfg.number_of_timesteps_hourly

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
                        'gauss': "gauss", 'meanlinear': "meanlinear"}
ews_spatial_signals = {'mn': "mean", 'std': "standard deviation", 'var': "variance", 'skw': "skewness",
                       'krt': "kurtosis", 'mI': "Moran's I"}

# EWS time-axis (x-axis) logic
"""
Constructs the time axis for plotting early-warning signals.

The time axis depends on:
- whether the variable is spatial or temporal
- the snapshot interval (spatial) or rolling window structure (temporal)
- whether a cutoff point is used

Temporal EWS are plotted at the END of each rolling window.
Spatial EWS are plotted at each saved snapshot.

Args:
----
variable : StateVariable
    The state variable for which the EWS is plotted.

number_of_timesteps : int
    Total number of timesteps of the model run.

Returns:
----
x_axis : numpy.ndarray
    Time axis corresponding to the EWS values.
"""


def ews_time_axis(variable, number_of_timesteps):
    if variable.spatial:
        start = cfg.interval_map_snapshots
        step = start
    else:
        start = variable.window_size
        step = variable.window_size - variable.window_overlap

    end = cfg.cutoff_point if cfg.cutoff else number_of_timesteps
    x = np.arange(start, end + 1, step)
    return x[x <= end]

# EWS signal-label logic
"""
Returns a human-readable label for an early-warning signal.

Selects the appropriate signal dictionary depending on whether
the variable is temporal or spatial.

Args:
----
variable : StateVariable
    The state variable for which the signal is plotted.

signal : str
    Short-hand name of the EWS (e.g. 'std', 'acr', 'mI').

Returns:
----
label : str
    Human-readable name of the EWS.
"""


def signal_label(variable, signal):
    if variable.temporal:
        if signal not in ews_temporal_signals:
            raise ValueError(f"Unknown temporal ES '{signal}'")
        return ews_temporal_signals[signal]
    else:
        if signal not in ews_spatial_signals:
            raise ValueError(f"Unknown spatial ES '{signal}'")
        return ews_spatial_signals[signal]

# User inputs for weekly-hourly coupled plots
"""
Takes user inputs from the console to construct a plot combining weekly timeseries and early-warning signals as
calculated from model runs of the hourly model.

Asks for:
    Timeseries (from the weekly model, up to 9 in combination with signals)
    State variables (from the hourly model)
    Signals (from the hourly model, up to 9 in combination with timeseries)
    Legend (whether a legend is included in the plot)
    Save (whether the plot is saved)
    Show (whether the plot is shown)
    
Runs plot_maker_weekly_hourly_coupled() for given inputs

"""


def user_input_weekly_hourly_coupled():
    timeseries_list = []
    variables_list = []
    signals_list = []

    print("Timeseries present in the current run are:")
    for name in names_timeseries:
        print(name)

    nr_of_variables = 0
    cont = True
    while cont:
        nr_of_variables += 1

        print("Enter the short name for the timeseries:")
        timeseries_input = input()

        matches = [ts for ts in timeseries if ts.name == timeseries_input]
        if not matches:
            raise ValueError(f"Unknown timeseries '{timeseries_input}'")
        ts_weekly = matches[0]
        if ts_weekly.temporal:
            timeseries_list.append(ts_weekly)
        elif ts_weekly.spatial:
            print("Spatial data can not be used for timeseries.")

        if nr_of_variables == 9:
            cont = False
        elif nr_of_variables < 9:
            print("Include another timeseries? [Y/n]")
            another_input = input()
            if another_input == 'Y' or another_input == 'y':
                cont = True
            else:
                cont = False

    if nr_of_variables < 9:
        cont = True

        print("Variables present in the current run are:")
        for name in names:
            print(name)

    while cont:
        nr_of_variables += 1

        print(f"Enter the short name for state variable {nr_of_variables}")
        variable_input = input()
        matches = [var for var in variables if var.name == variable_input]
        if not matches:
            raise ValueError(f"Unknown variable '{variable_input}'")
        variable_name = matches[0]
        variables_list.append(variable_name)

        if variable_name.temporal:
            print("EW signals present are:", ews_temporal_signals)
        elif variable_name.spatial:
            print("EW signals present are:", ews_spatial_signals)

        print(f"Enter the short name for the signal for variable {nr_of_variables}")
        signal_input = input()
        signals_list.append(signal_input)

        if nr_of_variables == 9:
            cont = False
        elif nr_of_variables < 9:
            print("Include another variable? [Y/n]")
            another_input = input()
            if another_input == 'Y' or another_input == 'y':
                cont = True
            else:
                cont = False

    print("Add a legend to the plot? [Y/n]")
    legend_input = input()
    if legend_input == 'Y' or legend_input == 'y':
        legend = True
    else:
        legend = False

    print("Save the plot as a .pdf? [Y/n]")
    save_plot = input()
    if save_plot == 'Y' or save_plot == 'y':
        save = True
    else:
        save = False

    print("Show the plot when finished? [Y/n]")
    print("Note that the program is still running if the plot stays open.")
    show_plot = input()
    if show_plot == 'Y' or show_plot == 'y':
        show = True
    else:
        show = False

    plot_maker_weekly_hourly_coupled(timeseries=timeseries_list, variables=variables_list, signals=signals_list,
                                     legend=legend, save=save, show=show)


# Plot maker function weekly-hourly coupled
"""
Constructs a plot combining weekly timeseries and early-warning signals as calculated from model runs of the hourly 
model.

Args:
----

timeseries : array of state variables from the weekly model to be plotted as timeseries

variables : array of state variables from the hourly model for which the signals are plotted

signals : array of early-warning signals from the hourly model to be plotted as scatterplot (points).

trendline_on : bool, selects whether a trendline for the signal-points is shown. Not selected through user input.

numbers_on : bool, selects whether the signal-point is numbered corresponding to the snapshot. Not selected through user 
    input.

legend : bool, selects whether a legend is added to the plot.

save : bool, selects whether plot is saved or not. Standard is to save as .pdf, can be saved as .png

show : bool, selects whether plot is shown or not.

Returns:
----

Optional : Plot of weekly timeseries and hourly EWS, optionally saved to disk.

"""


def plot_maker_weekly_hourly_coupled(timeseries, variables, signals, trendline_on=False, numbers_on=False,
                                     legend=False, save=False, show=False):
    fig, ax1 = plt.subplots(constrained_layout=True)

    snapshot_timeseries = np.loadtxt('./snapshot_times.numpy.txt')
    timeseries_x_axis = np.arange(0, cfg.number_of_timesteps_weekly, 1)

    nr_of_variables = len(timeseries) + len(variables)
    axes = [ax1]
    plots = []
    offset = max(40, 300 // max(1, nr_of_variables))
    for i in np.arange(nr_of_variables):
        if nr_of_variables > i + 1:
            ax = ax1.twinx()
            ax.spines["right"].set_position(("outward", offset * i))
            axes.append(ax)

    # Grid
    apply_ews_plot_style(ax1, grid=True)

    # X axis label (uses only 1 for the whole plot)
    ax1.set_xlabel("time (weeks, window end)")

    # Linestyles
    linestyles = [
        (0, ()),  # solid
        (0, (1, 1)),  # dotted
        (0, (5, 5)),  # dashed
        (0, (3, 1, 1, 1)),  # densely dashdotted
        (0, (3, 1, 1, 1, 1, 1)),  # densely dashdotdotted
        (0, (1, 1)),  # dotted
        (0, (5, 5)),  # dashed
        (0, (3, 1, 1, 1)),  # densely dashdotted
        (0, (3, 1, 1, 1, 1, 1))  # densely dashdotdotted
    ]

    # Colours list
    colours_list = cfg.EWS_colour_cycle

    for i in np.arange(nr_of_variables):
        ax = axes[i]

        # Cycle colours and linestyles
        colours = np.concatenate((np.asarray(colours_list)[i:], np.asarray(colours_list)[:i]))
        ax.set_prop_cycle(cycler(color=colours, linestyle=linestyles))

        if i <= len(timeseries) - 1:
            fname_ts_weekly = ews.file_name_str(timeseries[i].name, cfg.number_of_timesteps_weekly)
            fpath_ts_weekly = f"./inputs_from_weekly/{fname_ts_weekly}"
            timeseries_y_axis = np.loadtxt(fpath_ts_weekly + '.numpy.txt')
            plot = ax.plot(timeseries_x_axis, timeseries_y_axis, linewidth=cfg.EWS_linewidth, label=f"{timeseries[i].full_name} timeseries")
            ax.set_ylabel(f"{timeseries[i].full_name} timeseries ({timeseries[i].unit})", color=plot[0].get_color())

            # Ticks
            apply_ews_plot_style(ax, grid=False)
            ax.tick_params(axis='y', which='both', colors=colours[0])
            plt.setp(ax.get_yticklabels(), color=colours[0])

            for p in plot:
                plots.append(p)

        else:
            snapshot_y_axis = []
            snapshot_x_axis = snapshot_timeseries

            dim = None
            if variables[i - len(timeseries)].temporal:
                dim = '.t.'
            elif variables[i - len(timeseries)].spatial:
                dim = '.s.'

            fname_signal_hourly = variables[i - len(timeseries)].name + dim + signals[i - len(timeseries)] + '.numpy.txt'

            for nr, _ in enumerate(snapshot_timeseries):
                fpath_signal_hourly = os.path.join("./h" + str(nr).zfill(2) + "/" + fname_signal_hourly)
                statistic = np.loadtxt(fpath_signal_hourly)
                if statistic.ndim == 0:
                    snapshot_y_axis.append(statistic)
                elif statistic.ndim == 1:
                    snapshot_y_axis.append(statistic[-1])

            var = variables[i - len(timeseries)]
            sig = signals[i - len(timeseries)]

            label = signal_label(var, sig)
            slabel = f"{var.full_name} {label}"

            plot = ax.scatter(snapshot_x_axis, snapshot_y_axis, label=slabel, s=cfg.EWS_scatter_size,
                              alpha=cfg.EWS_scatter_alpha, edgecolors='k', linewidths=cfg.EWS_scatter_edgewith)

            ax.set_ylabel(label, color=colours[0])

            if trendline_on:
                z = np.polyfit(snapshot_x_axis, snapshot_y_axis, 1)
                p = np.poly1d(z)
                ax.plot(snapshot_x_axis, p(snapshot_x_axis), "r--")

            if numbers_on:
                for k, nr in enumerate(snapshot_timeseries):
                    ax.annotate(int(k), (snapshot_x_axis[k], snapshot_y_axis[k]))

            # Ticks
            ax.minorticks_on()
            ax.tick_params(axis='y', which='both', colors=colours[0])
            plt.setp(ax.get_yticklabels(), color=colours[0])

            plots.append(plot)

    # Legend
    if legend:
        labs = [p.get_label() for p in plots]
        ax1.legend(plots, labs, loc='upper right', frameon=False, fontsize=9)

    # Saving file
    if save:
        fig.text(0.995, 0.005, "EWSPy - KvL", ha='right', va='bottom', fontsize=6, alpha=0.25)
        formats = ["svg", "pdf"]
        timestamp = datetime.now()

        dir_name = './plots/'
        if os.path.isdir(dir_name) == False:
            os.makedirs(dir_name)

        for fmt in formats:
            if len(variables) > 1:
                fig.savefig(dir_name + f"{variables[0].name}_{signals[0]}_{signals[1]}_with_{nr_of_variables}_variables"
                            f"_{timestamp.hour}.{timestamp.minute}.{timestamp.second}.{fmt}", dpi=300, format=fmt)
            else:
                fig.savefig(dir_name + f"{variables[0].name}_{signals[0]}_{timestamp.hour}.{timestamp.minute}."
                            f"{timestamp.second}.{fmt}", dpi=300, format=fmt)


    # Showing plot
    if show:
        plt.draw()
        plt.show()
    elif not show:
        plt.close()


# Plot maker function weekly/hourly
"""
Constructs a plot of early-warning signals as calculated from model runs of the hourly or weekly model.

Args:
----

variables_input : array of state variables from the hourly/weekly model for which the signals are plotted.

signals : array of early-warning signals from the hourly/weekly model to be plotted.

path : str, path where inputs from the hourly/weekly model are stored.

legend : bool, selects whether a legend is added to the plot.

save : bool, selects whether plot is saved or not. Standard is to save as .pdf, can be saved as .png

show : bool, selects whether plot is shown or not.

Returns:
----

Optional : Plot of weekly timeseries and hourly EWS, optionally saved to disk.

"""


def plot_maker(variables_input, signals, path='/1/', legend=False, save=False, show=False):
    fig, ax1 = plt.subplots(constrained_layout=True)

    nr_of_variables = len(variables_input)
    axes = [ax1]
    plots = []
    offset = 70
    for i in range(nr_of_variables):
        if nr_of_variables > i + 1:
            ax = ax1.twinx()
            ax.spines["right"].set_position(("outward", offset * i))
            axes.append(ax)

    # Grid
    apply_ews_plot_style(ax1, grid=True)

    # X axis label (uses only 1 for the whole plot)
    ax1.set_xlabel("time (weeks, window end)")

    # Linestyles
    linestyles = [
        (0, ()),  # solid
        (0, (1, 1)),  # dotted
        (0, (5, 5)),  # dashed
        (0, (3, 1, 1, 1)),  # densely dashdotted
        (0, (3, 1, 1, 1, 1, 1)),  # densely dashdotdotted
        (0, (1, 1)),  # dotted
        (0, (5, 5)),  # dashed
        (0, (3, 1, 1, 1)),  # densely dashdotted
        (0, (3, 1, 1, 1, 1, 1))  # densely dashdotdotted
    ]

    # Colours list
    colours_list = cfg.EWS_colour_cycle

    for i in np.arange(nr_of_variables):
        plot = None
        ax = axes[i]
        variable = variables_input[i]
        signal = signals[i]

        # Cycle colours and linestyles
        colours = np.concatenate((np.asarray(colours_list)[i:], np.asarray(colours_list)[:i]))
        ax.set_prop_cycle(cycler(color=colours, linestyle=linestyles))

        # Dimension and x axis
        dim = None
        if variable.spatial:
            dim = '.s.'
        elif variable.temporal:
            dim = '.t.'

        x_axis = ews_time_axis(variable, number_of_timesteps)

        # Signal
        if signal == "timeseries":
            fname = ews.file_name_str(variable.name, number_of_timesteps)
            fpath = os.path.join(path, fname)
            timeseries_y_axis = np.loadtxt(fpath + '.numpy.txt')
            timeseries_x_axis = np.arange(0, number_of_timesteps, 1)
            label = signal_label(variable, signal)
            plot = ax.plot(timeseries_x_axis, timeseries_y_axis, linewidth=cfg.EWS_linewidth, label=f"{variable.full_name} {label}")

            if timeseries_y_axis.ndim > 1:
                lines = ax.get_lines()
                ax.set_ylabel(f"{variable.full_name} {label} ({variable.unit})", color=plot[0].get_color())
                for loc, line in enumerate(lines):
                    line.set_label(f"{variable.full_name} {loc + 1} - {label}")
            else:
                ax.set_ylabel(f"{variable.full_name} {label} ({variable.unit})", color=plot[0].get_color())

        elif signal == "gauss":
            fname = ews.file_name_str(variable.name + 'g', number_of_timesteps)
            fpath = os.path.join(path, fname)
            timeseries_y_axis = np.loadtxt(fpath + '.numpy.txt')
            timeseries_x_axis = np.arange(0, number_of_timesteps, 1)
            label = signal_label(variable, signal)
            plot = ax.plot(timeseries_x_axis, timeseries_y_axis, linewidth=cfg.EWS_linewidth, label=f"{variable.full_name} {label}")

            if timeseries_y_axis.ndim > 1:
                lines = ax.get_lines()
                ax.set_ylabel(f"{variable.full_name} {label}", color=plot[0].get_color())
                for loc, line in enumerate(lines):
                    line.set_label(f"{variable.full_name} {loc + 1} - {label}")
            else:
                ax.set_ylabel(f"{variable.full_name} {label}", color=plot[0].get_color())

        elif signal == "meanlinear":
            fname = ews.file_name_str(variable.name + 'l', number_of_timesteps)
            fpath = os.path.join(path, fname)
            timeseries_y_axis = np.loadtxt(fpath + '.numpy.txt')
            timeseries_x_axis = np.arange(0, number_of_timesteps, 1)
            label = signal_label(variable, signal)
            plot = ax.plot(timeseries_x_axis, timeseries_y_axis, linewidth=cfg.EWS_linewidth, label=f"{variable.full_name} {label}")

            if timeseries_y_axis.ndim > 1:
                lines = ax.get_lines()
                ax.set_ylabel(f"{variable.full_name} {label}", color=plot[0].get_color())
                for loc, line in enumerate(lines):
                    line.set_label(f"{variable.full_name} {loc + 1} - {label}")
            else:
                ax.set_ylabel(f"{variable.full_name} {label}", color=plot[0].get_color())

        elif signal != 'None':
            fpath = os.path.join(path + variable.name + dim + signal)
            signal_array = np.loadtxt(fpath + '.numpy.txt')
            label = signal_label(variable, signal)
            if signal_array.ndim > 1 and variable.temporal:
                signal_array = signal_array.T
                plot = ax.plot(x_axis, signal_array)
                lines = ax.get_lines()
                ax.set_ylabel(label, color=plot[0].get_color())
                for loc, line in enumerate(lines):
                    line.set_label(f"{variable.full_name} {loc + 1} - {label}")
            elif variable.spatial:
                plot = ax.plot(x_axis, signal_array, label=f"{variable.full_name} {label}")
                ax.set_ylabel(label, color=plot[0].get_color())
            elif variable.temporal:
                plot = ax.plot(x_axis, signal_array, label=f"{variable.full_name} {label}")
                ax.set_ylabel(label, color=plot[0].get_color())

        # Ticks & spines
        apply_ews_plot_style(ax, grid=False)
        ax.tick_params(axis='y', which='both', colors=colours[0])
        plt.setp(ax.get_yticklabels(), color=colours[0])

        for p in plot:
            plots.append(p)

    # Legend
    if legend:
        labs = [p.get_label() for p in plots]
        ax1.legend(plots, labs, loc='upper right', frameon=False, fontsize=9)

    # Saving file
    if save:
        formats = ["svg", "pdf"]
        timestamp = datetime.now()

        fig.text(0.995, 0.005, "EWSPy - KvL", ha='right', va='bottom', fontsize=6, alpha=0.25)

        for fmt in formats:
            if nr_of_variables > 1:
                fig.savefig(path + f"{variables_input[0].name}_{signals[0]}_{signals[1]}_with_{nr_of_variables}_variables_"
                                   f"{timestamp.hour}.{timestamp.minute}.{timestamp.second}.{fmt}", dpi=300, format=fmt)
            else:
                fig.savefig(path + f"{variables_input[0].name}_{signals[0]}_{timestamp.hour}.{timestamp.minute}."
                                   f"{timestamp.second}.{fmt}", dpi=300, format=fmt)

    # Showing plot
    if show:
        plt.draw()
        plt.show()
    elif not show:
        plt.close()


# User inputs for plot maker
"""
Takes user inputs from the console to construct a plot of early-warning signals as calculated from model runs of either
the hourly or weekly model.

Args:
----

path : str, path where model/EWS_weekly/hourly.py outputs are stored.

Asks for:
    State variables (from the hourly model)
    Signals (from the hourly model, up to 9)
    Legend (whether a legend is included in the plot)
    Save (whether the plot is saved)
    Show (whether the plot is shown)
    
Runs plot_maker() for given inputs

"""


def user_input_plotmaker(path='./1/'):
    variables_list = []
    signals_list = []

    print("Variables present in the current run are:")
    for name in names:
        print(name)

    nr_of_variables = 0
    cont = True
    while cont:
        nr_of_variables += 1

        print(f"Enter the short name for state variable {nr_of_variables}")
        variable_input = input()
        matches = [var for var in variables if var.name == variable_input]
        if not matches:
            raise ValueError(f"Unknown variable '{variable_input}'")
        variable_name = matches[0]

        variables_list.append(variable_name)

        if variable_name.temporal:
            print("EW signals present are:", ews_temporal_signals)
        elif variable_name.spatial:
            print("EW signals present are:", ews_spatial_signals)
        print(f"Enter the short name for the signal for variable {nr_of_variables}")
        signal_input = input()
        signals_list.append(signal_input)

        if nr_of_variables == 9:
            cont = False
        elif nr_of_variables < 9:
            print("Include another variable? [Y/n]")
            another_input = input()
            if another_input == 'Y' or another_input == 'y':
                cont = True
            else:
                cont = False

    print("Add a legend to the plot? [Y/n]")
    legend_input = input()
    if legend_input == 'Y' or legend_input == 'y':
        legend = True
    else:
        legend = False

    print("Save the plot as a .pdf & .svg? [Y/n]")
    save_plot = input()
    if save_plot == 'Y' or save_plot == 'y':
        save = True
    else:
        save = False

    print("Show the plot when finished? [Y/n]")
    print("Note that the program is still running if the plot stays open.")
    show_plot = input()
    if show_plot == 'Y' or show_plot == 'y':
        show = True
    else:
        show = False

    plot_maker(variables_input=variables_list, signals=signals_list, path=path, legend=legend, save=save, show=show)


# Loop-function for user inputs for plot maker
"""
Loops the function for user inputs for plot maker.

Args:
----

path : str, path where model/EWS_weekly/hourly.py outputs are stored.

Asks for:
    Whether user_input_plot_maker() is to be ran again.
    
Runs user_input_plot_maker() for given inputs

"""


def user_input_plotmaker_looper(path='./1/'):
    user_input_plotmaker(path=path)
    print("Would you like to make another plot? [Y/n]")
    answer = input()
    if answer == 'Y' or answer == 'y':
        user_input_plotmaker_looper(path=path)
    elif answer == 'N' or answer == 'n':
        print("Terminated plotmaker. Goodbye.")
    else:
        print("Invalid input, terminated plotmaker. Goodbye.")


if __name__ == "__main__":
    user_input_plotmaker_looper()
