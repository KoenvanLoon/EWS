import numpy as np
import os
from cycler import cycler
import matplotlib.pyplot as plt
from datetime import datetime

import EWS_configuration as cfg
import EWSPy as ews
import EWS_StateVariables as ews_sv

## State variables for EWS ##
# State variables present in EWS_StateVariables can be added through EWS_main_configuration.py
variables = ews_sv.variables_weekly
# variables = ews_sv.variables_hourly

timeseries = ews_sv.variables_weekly

names = []
for variable in variables:
    names.append([f'{variable.full_name} as {variable.name}'])

names_timeseries = []
for ts in timeseries:
    names_timeseries.append([f'{ts.full_name} as {ts.name}'])

## Number of timesteps over which EWS can be calculated ##
# This number can be different for the weekly and hourly model
number_of_timesteps = cfg.number_of_timesteps_weekly
# number_of_timesteps = cfg.number_of_timesteps_hourly


## Statistical EWS ##
ews_temporal_signals = {'mn': "mean", 'std': "standard deviation", 'var': "variance",
                        'cv': "coefficient of variation", 'skw': "skewness", 'krt': "kurtosis",
                        'dfa': "detrended fluctuation analysis", 'acr': "autocorrelation", 'AR1': "AR1",
                        'rr': "return rate", 'coh': "conditional heteroskedasticity", 'timeseries': "timeseries",
                        'gauss': "gauss"}
ews_spatial_signals = {'mn': "mean", 'std': "standard deviation", 'var': "variance", 'skw': "skewness",
                       'krt': "kurtosis", 'mI': "Moran's I"}


def plot2(variable1, signal1='None', variable2='None', signal2='None', path='./1/', legend=False, save=False, show=False):
    fig, ax1 = plt.subplots()

    # Grid
    ax1.minorticks_on()
    ax1.grid(which='minor', linestyle=':', alpha=0.2)
    ax1.grid(which='major', linestyle='--', alpha=0.5)

    # Linestyles
    linestyles = [
        (0, ()),                        # solid
        (0, (1, 1)),                    # dotted
        (0, (5, 5)),                    # dashed
        (0, (3, 1, 1, 1)),              # densely dashdotted
        (0, (3, 1, 1, 1, 1, 1)),        # densely dashdotdotted
        (0, (1, 1)),                    # dotted
        (0, (5, 5)),                    # dashed
        (0, (3, 1, 1, 1)),              # densely dashdotted
        (0, (3, 1, 1, 1, 1, 1))         # densely dashdotdotted
    ]

    # Colours (colours2 only set if second variable present)
    cycle_indx = np.linspace(0, 1, 10)
    colours1 = [plt.cm.tab10(i) for i in cycle_indx[:-1]]
    ax1.set_prop_cycle(cycler(color=colours1, linestyle=linestyles))

    # X axis
    ax1.set_xlabel('time (weeks)')

    if variable1.spatial:
        ax1.set_title(f"{variable1.full_name} {ews_spatial_signals[signal1]}")
        if cfg.cutoff:
            x_axis1 = np.arange(cfg.interval_map_snapshots, cfg.cutoff_point + 1, cfg.interval_map_snapshots)
        else:
            x_axis1 = np.arange(cfg.interval_map_snapshots, number_of_timesteps + cfg.interval_map_snapshots,
                                cfg.interval_map_snapshots)
        dim1 = '.s.'
    elif variable1.temporal:
        ax1.set_title(f"{variable1.full_name} {ews_temporal_signals[signal1]}")
        if cfg.cutoff:
            x_axis1 = np.arange(variable1.window_size, cfg.cutoff_point + 1, variable1.window_size - variable1.window_overlap)
        else:
            x_axis1 = np.arange(variable1.window_size, number_of_timesteps, variable1.window_size - variable1.window_overlap)
        dim1 = '.t.'

    # Signal 1
    if signal1 == 'timeseries':
        fname = ews.file_name_str(variable1.name, number_of_timesteps)
        fpath = os.path.join(path + fname)
        timeseries_y_axis = np.loadtxt(fpath + '.numpy.txt')
        timeseries_x_axis = np.arange(0, number_of_timesteps, 1)
        plot1 = ax1.plot(timeseries_x_axis, timeseries_y_axis, label=f'Continues measurement of {variable1.full_name}')
        ax1.set_ylabel(f"{ews_temporal_signals[signal1]} ({variable1.unit})", color=colours1[0])
    elif signal1 == 'gauss':
        fname = ews.file_name_str(variable1.name + 'g', number_of_timesteps)
        fpath = os.path.join(path + fname)
        timeseries_y_axis = np.loadtxt(fpath + '.numpy.txt')
        timeseries_x_axis = np.arange(0, number_of_timesteps, 1)
        plot1 = ax1.plot(timeseries_x_axis, timeseries_y_axis, label=f'Gaussian detrending {variable1.full_name}')
        ax1.set_ylabel(f"{ews_temporal_signals[signal1]} ({variable1.unit})", color=colours1[0])
    elif signal1 != 'None':
        fpath = os.path.join(path + variable1.name + dim1 + signal1)
        signal1_array = np.loadtxt(fpath + '.numpy.txt')
        if signal1_array.ndim > 1 and variable1.temporal:
            signal1_array = signal1_array.T
            plot1 = ax1.plot(x_axis1, signal1_array)
            lines = ax1.get_lines()
            for location, line in enumerate(lines):
                line.set_label(f'{variable1.full_name} {location + 1} - {ews_temporal_signals[signal1]}')
        elif variable1.spatial:
            plot1 = ax1.plot(x_axis1, signal1_array, label=f'{variable1.full_name} {ews_spatial_signals[signal1]}')
            ax1.set_ylabel(f"{ews_temporal_signals[signal1]}", color=colours1[0])
        elif variable1.temporal:
            plot1 = ax1.plot(x_axis1, signal1_array, label=f'{variable1.full_name} {ews_temporal_signals[signal1]}')
            ax1.set_ylabel(f"{ews_temporal_signals[signal1]}", color=colours1[0])

    # Signal 2
    if variable2 != 'None':
        ax2 = ax1.twinx()
        ax2.minorticks_on()
        colours2 = [plt.cm.tab10(i) for i in cycle_indx[1:]]
        ax2.set_prop_cycle(cycler(color=colours2, linestyle=linestyles))
        ax1.tick_params(axis='y', which='both', colors=colours1[0])
        ax2.tick_params(axis='y', which='both', colors=colours2[0])

        if variable2.spatial:
            if cfg.cutoff:
                x_axis2 = np.arange(cfg.interval_map_snapshots, cfg.cutoff_point + 1, cfg.interval_map_snapshots)
            else:
                x_axis2 = np.arange(cfg.interval_map_snapshots, number_of_timesteps + cfg.interval_map_snapshots,
                                    cfg.interval_map_snapshots)
            dim2 = '.s.'
        elif variable2.temporal:
            if cfg.cutoff:
                x_axis2 = np.arange(variable1.window_size, cfg.cutoff_point + 1, variable2.window_size - variable2.window_overlap)
            else:
                x_axis2 = np.arange(variable1.window_size, number_of_timesteps, variable2.window_size - variable2.window_overlap)
            dim2 = '.t.'

        if signal2 == 'timeseries':
            fname = ews.file_name_str(variable2.name, number_of_timesteps)
            fpath = os.path.join(path + fname)
            timeseries_y_axis = np.loadtxt(fpath + '.numpy.txt')
            timeseries_x_axis = np.arange(0, number_of_timesteps, 1)
            plot2 = ax2.plot(timeseries_x_axis, timeseries_y_axis, label=f'Continues measurement of {variable2.full_name}')
            ax2.set_ylabel(f"{ews_temporal_signals[signal2]} ({variable2.unit})", color=colours2[0])
        elif signal2 == 'gauss':
            fname = ews.file_name_str(variable1.name + 'g', number_of_timesteps)
            fpath = os.path.join(path + fname)
            timeseries_y_axis = np.loadtxt(fpath + '.numpy.txt')
            timeseries_x_axis = np.arange(0, number_of_timesteps, 1)
            plot2 = ax2.plot(timeseries_x_axis, timeseries_y_axis, label=f'Gaussian detrending {variable2.full_name}')
            ax2.set_ylabel(f"{ews_temporal_signals[signal2]} ({variable2.unit})", color=colours2[0])
        elif signal2 != 'None':
            fpath = os.path.join(path + variable2.name + dim2 + signal2)
            signal2_array = np.loadtxt(fpath + '.numpy.txt')
            if signal2_array.ndim > 1 and variable2.temporal:
                signal2_array = signal2_array.T
                plot2 = ax2.plot(x_axis2, signal2_array)
                lines = ax2.get_lines()
                for location, line in enumerate(lines):
                    line.set_label(f'{variable2.full_name} {location + 1} - {ews_temporal_signals[signal2]}')
            elif variable2.spatial:
                plot2 = ax2.plot(x_axis2, signal2_array, label=f'{variable2.full_name} {ews_spatial_signals[signal2]}')
                ax2.set_ylabel(f"{ews_temporal_signals[signal2]}", color=colours2[0])
            elif variable2.temporal:
                plot2 = ax2.plot(x_axis2, signal2_array, label=f'{variable2.full_name} {ews_temporal_signals[signal2]}')
                ax2.set_ylabel(f"{ews_temporal_signals[signal2]}", color=colours2[0])

        if variable1.temporal:
            if variable2.temporal:
                ax1.set_title(f"{variable1.full_name} {ews_temporal_signals[signal1]} & {variable2.full_name} "
                          f"{ews_temporal_signals[signal2]}")
            elif variable2.spatial:
                ax1.set_title(f"{variable1.full_name} {ews_temporal_signals[signal1]} & {variable2.full_name} "
                          f"{ews_spatial_signals[signal2]}")
        elif variable1.spatial:
            if variable2.temporal:
                ax1.set_title(f"{variable1.full_name} {ews_spatial_signals[signal1]} & {variable2.full_name} "
                          f"{ews_temporal_signals[signal2]}")
            elif variable2.spatial:
                ax1.set_title(f"{variable1.full_name} {ews_spatial_signals[signal1]} & {variable2.full_name} "
                          f"{ews_spatial_signals[signal2]}")

        plots = plot1 + plot2

    # If signal 2 not present
    else:
        plots = plot1

    # Legend
    if legend:
        labs = [p.get_label() for p in plots]
        ax1.legend(plots, labs)

    # Saving file
    fig.tight_layout()
    if save:
        if variable2 != 'None':
            fig.savefig(path + f"{variable1.name}_{signal1}_and_{variable2.name}_{signal2}.pdf", format="pdf")
        else:
            fig.savefig(path + f"{variable1.name}_{signal1}.pdf", format="pdf")

    # Showing plot
    if show:
        plt.show()
    elif not show:
        plt.close()


def user_plotmaker(path='./1/'):
    print("Variables present in the current run are:")
    for name in names:
        print(name)

    print("Enter the short name for state variable 1:")
    variable1_input = input()
    variable1 = [variable for variable in variables if variable.name == variable1_input][0]
    if variable1.temporal:
        print("EW signals present are:", ews_temporal_signals)
    elif variable1.spatial:
        print("EW signals present are:", ews_spatial_signals)
    print("Enter the short name for the signal for variable 1:")
    signal1_input = input()

    print("Include a second variable? [Y/n]")
    second_variable_input = input()
    if second_variable_input == 'Y' or second_variable_input == 'y':
        print("Enter the short name for state variable 2:")
        variable2_input = input()

        variable2 = [variable for variable in variables if variable.name == variable2_input][0]
        if variable2.temporal:
            print("EW signals present are:", ews_temporal_signals)
        elif variable2.spatial:
            print("EW signals present are:", ews_spatial_signals)
        print("Enter the short name for the signal for variable 1:")
        signal2_input = input()
    else:
        variable2 = 'None'
        signal2_input = 'None'

    print("Add a legend to the plot? [Y/n]")
    legend = input()
    if legend == 'Y' or legend == 'y':
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

    plot2(variable1=variable1, signal1=signal1_input, variable2=variable2, signal2=signal2_input, path=path,
          legend=legend, save=save, show=show)


def user_plotmaker_looper(path='./1/'):
    user_plotmaker(path=path)
    print("Would you like to make another plot? [Y/n]")
    answer = input()
    if answer == 'Y' or answer == 'y':
        user_plotmaker_looper(path=path)
    elif answer == 'N' or answer == 'n':
        print("Terminated plotmaker. Goodbye.")
    else:
        print("Invalid input, terminated plotmaker. Goodbye.")


def weekly_hourly_coupled():
    fig, ax1 = plt.subplots()

    ax1.minorticks_on()
    ax1.grid(which='minor', linestyle=':', alpha=0.2)
    ax1.grid(which='major', linestyle='--', alpha=0.5)

    ax1.set_xlabel('time (weeks)')

    snapshot_timeseries = np.loadtxt('./snapshot_times.numpy.txt')

    print("Timeseries present in the current run are:")
    for name in names_timeseries:
        print(name)

    print("Enter the short name for the timeseries:")
    timeseries_input = input()

    timeseries_ = [ts for ts in timeseries if ts.name == timeseries_input][0]
    if timeseries_.temporal:
        #timeseries = [ts for ts in timeseries if timeseries.name == timeseries_input][0]

        fname_week = ews.file_name_str(timeseries_.name, cfg.number_of_timesteps_weekly)
        fpath = f'./inputs_from_weekly/{fname_week}'

        timeseries_y_axis = np.loadtxt(fpath + '.numpy.txt')
        timeseries_x_axis = np.arange(0, cfg.number_of_timesteps_weekly, 1)

    elif timeseries_.spatial:
        print("Spatial data can not be used for timeseries.")

    print("Variables present in the current run are:")
    for name in names:
        print(name)

    print("Enter the short name for state variable:")
    variable_input = input()
    variable = [variable for variable in variables if variable.name == variable_input][0]
    if variable.temporal:
        print("EWS signals present are:", ews_temporal_signals)
    elif variable.spatial:
        print("EWS signals present are:", ews_spatial_signals)

    statistic_input = input()

    snapshot_y_axis = []
    snapshot_x_axis = snapshot_timeseries

    for nr, _ in enumerate(snapshot_timeseries):

        if variable.temporal:
            dim = '.t.'
        elif variable.spatial:
            dim = '.s.'

        fname = variable.name + dim + statistic_input + '.numpy.txt'
        fpath = os.path.join('./h' + str(nr).zfill(2) + '/' + fname)
        statistic = np.loadtxt(fpath)
        if statistic.ndim == 0:
            snapshot_y_axis.append(statistic)
        elif statistic.ndim == 1:
            snapshot_y_axis.append(statistic[-1])

    #colour = plt.cm.tab10(1)

    ax1.plot(timeseries_x_axis, timeseries_y_axis, label=f'Continues measurement of {timeseries_.full_name}', color=plt.cm.tab10(0))
    #ax1.set_ylabel()

    ax2 = ax1.twinx()
    ax2.minorticks_on()

    ax1.tick_params(axis='y', which='both', colors=plt.cm.tab10(0))
    ax2.tick_params(axis='y', which='both', colors=plt.cm.tab10(1))

    if variable.spatial:
        ax2.scatter(snapshot_x_axis, snapshot_y_axis, label=f'{variable.full_name} {ews_spatial_signals[statistic_input]}', c=plt.cm.tab10(1))
        ax2.set_ylabel(f'{ews_spatial_signals[statistic_input]}', c=plt.cm.tab10(1))
        ax1.set_title(f'{timeseries_.full_name} timeseries from weekly & {variable.full_name} {ews_spatial_signals[statistic_input]}')
    elif variable.temporal:
        ax2.scatter(snapshot_x_axis, snapshot_y_axis, label=f'{variable.full_name} {ews_temporal_signals[statistic_input]}', c=plt.cm.tab10(1))
        ax2.set_ylabel(f'{ews_temporal_signals[statistic_input]}', c=plt.cm.tab10(1))
        ax1.set_title(f'{timeseries_.full_name} timeseries from weekly & {variable.full_name} {ews_temporal_signals[statistic_input]}')

    # Trendline
    z = np.polyfit(snapshot_x_axis, snapshot_y_axis, 1)
    p = np.poly1d(z)
    ax2.plot(snapshot_x_axis, p(snapshot_x_axis), "r--")

    #plots = plot1 + plot2
    fig.tight_layout()

    print("Put number of snapshot next to point? [Y/n]")
    nr_on = input()
    if nr_on == 'Y'  or nr_on == 'y':
        for i, nr in enumerate(snapshot_timeseries):
            #ax2.annotate(int(nr), (snapshot_x_axis[i], snapshot_y_axis[i]))
            ax2.annotate(int(i), (snapshot_x_axis[i], snapshot_y_axis[i]))

    print("Save the plot as a .pdf? [Y/n]")
    save_plot = input()
    if save_plot == 'Y' or save_plot == 'y':
        timestamp = datetime.now()
        dir_name = './plots/'

        if os.path.isdir(dir_name) == False:
            os.makedirs(dir_name)

        fig.savefig(dir_name + f'{timeseries_.full_name}_timeseries_from_weekly_and_{variable.name}_{ews_temporal_signals[statistic_input]}_'
                               f'{timestamp.hour}.{timestamp.minute}.{timestamp.second}.pdf', format="pdf")

    plt.show()


# user_plotmaker_looper(path='./1/dtr0000/')
user_plotmaker_looper(path='./1/')
