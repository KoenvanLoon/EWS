# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (c) 2026 Koen van Loon
#
# See the LICENSE file in the repository root for full license text.

"""
EWS - Early Warning Signals
EWS Configuration (cfg)

@authors: KoenvanLoon & TijmenJanssen
"""

import EWS_main_configuration as mcfg
# !NOTE: This configuration mirrors EWS_main_configuration.py unless explicitly overridden in USER OVERRIDES FOR EWS RUNS
#           or EWS ANALYSIS SETTINGS.


# ==================================================================
# USER OVERRIDES FOR EWS RUNS
# ==================================================================

"""
User overrides for EWS_pycatch_weekly.py EWS runs.

Args:
-----

use_fixed_seed : 

random_seed_value :

initialSoilMoistureFractionCFG : float, the initial soil moisture fraction at the start of the model run.

return_ini_grazing : bool, selects whether the grazing rates return to the initial value after the halfway point of the
    grazing period. Usually False. (Note that this halfway point occurs on (1 - rel_start) * total time / 2) .

swapCatchments : bool, select to swap parameter values between two catchments, first time users will need to set this to 
    False.
    
default_window_size : int, global default for standard window size for state variables.

cutoff : bool, selects whether a cutoff point is used at a defined point for null model realizations, calculations and plots.
    Usually True, speeds up calculation time and makes null model realizations more accurate (see null models).

cutoff_point : int, time at which states shift. Retrieved from plotting the biomass timeseries.

-----
"""

# Reproducibility
use_fixed_seed = mcfg.use_fixed_seed
random_seed_value = mcfg.random_seed_value

# Initial soil moisture fraction for EWS runs
initialSoilMoistureFractionCFG = mcfg.initialSoilMoistureFractionFromDiskValue  # Has been set to 0.22 to avoid hourly spin-up dependency

# Grazing behaviour
return_ini_grazing = False

# Swap catchments
swapCatchments = False

# Default window size for state variables
default_window_size = 100

# Cutoff behaviour
cutoff = False
cutoff_point = 48000  # Cutoff at set timestep; refers to last timestep included (not snapshot index).


# ==================================================================
# EWS ANALYSIS SETTINGS
# ==================================================================

"""
Settings for EWS analysis.

Args:
-----

state_variables_for_ews_hourly/weekly : State variables for which early warning signals/statistics are calculated.
    Either list of strings or 'full'.

stepsInShift : int, number of snapshots for which the hourly model is going to be run in a specified period.

stepsTotal : int, number of snapshots for which the hourly model is going to be run over the whole period as specified
    for the weekly model. Evenly distributed before and after the stepsInShift period if possible.

generate_dummy_datasets : Selects whether null model realizations are generated or not.

nr_generated_datasets : Selects how many null model realizations are generated.

method_1 : If True, the given number of null model realizations are created by shuffling data (similar mean and 
    variance).

method_2 : If True, the given number of null model realizations are created with the same Fourier spectrum and 
    amplitudes (similar autocorrelations, mean and variance).

method_3 : If True, the given number of null model realizations are created with an AR(1) model fitted to the data
    (similar autocorrelations, mean and variance).

detrended_method : Either 'None' or 'Gaussian', selects whether none detrending occurs or Gaussian filtering detrending 
    using scipy.ndimage.gaussian_filter1d().

detrended_sigma : If detrended_method is 'Gaussian', selects the sigma used in scipy.ndimage.gaussian_filter1d().

save_detrended_data : Selects whether detrended temporal data used in the generation of null models is saved. Only
    relevent when detrending in EWS_weekly.py is not set to 'None'.

-----
"""

# State variables
state_variables_for_ews_hourly = mcfg.state_variables_for_ews_hourly
state_variables_for_ews_weekly = mcfg.state_variables_for_ews_weekly

# Number of weekly snapshots used around the transition for hourly model (snapshots, not raw timesteps)
stepsInShift = 20
stepsTotal = 30

# Generate null models
generate_dummy_datasets = True
nr_generated_datasets = 10  # 100 for final analyses (where computationally feasible - takes some time & power!)

# Methods for generated null models

# !NOTE: Enabling all null-model methods simultaneously increases runtime significantly, *especially* when coupled with
#           a high number of generated datasets.

method_1 = True     # Shuffle
method_2 = True     # Fourier
method_3 = True     # AR(1)
method_2b = True    # IAAFT

# Data detrending
detrended_method = 'None'   # 'None', 'MeanLinear', 'Gaussian' - !NOTE: 'MeanLinear': mean-centring for spatial data, linear detrending for timeseries
detrended_sigma = 100
save_detrended_data = True


# ==================================================================
# PYCATCH MODEL SETTINGS (Inherited from EWS_main_configuration.py)
# ==================================================================

"""
PyCatch model settings relevant to EWS calculations.
"""

# Hourly model input folder/file(s)
inputFolder = mcfg.inputFolder
cloneString = mcfg.cloneString
locations = mcfg.locations

# Duration of model runs
number_of_timesteps_hourly = mcfg.number_of_timesteps_hourly
number_of_timesteps_weekly = mcfg.number_of_timesteps_weekly

# Saving spatial data
map_data = mcfg.map_data
interval_map_snapshots = mcfg.interval_map_snapshots

# Saving temporal data
mean_timeseries_data = mcfg.mean_timeseries_data
loc_timeseries_data = mcfg.loc_timeseries_data

# Reporting of variables
setOfVariablesToReport = mcfg.setOfVariablesToReport

# Timesteps of reporting variables
timesteps_to_report_all_hourly = mcfg.timesteps_to_report_all_hourly
timesteps_to_report_all_weekly = mcfg.timesteps_to_report_all_weekly

timesteps_to_report_some_hourly = mcfg.timesteps_to_report_some_hourly
timesteps_to_report_some_weekly = mcfg.timesteps_to_report_some_weekly

# Hourly model report as numpy
doReportComponentsDynamicAsNumpy = mcfg.doReportComponentsDynamicAsNumpy

# Monte Carlo (MC) realizations
nrOfSamples = mcfg.nrOfSamples

# Read set of parameters for all MC realizations from disk
readDistributionOfParametersFromDisk = mcfg.readDistributionOfParametersFromDisk

# Particle filtering
filtering = mcfg.filtering

# Create realizations
createRealizations = mcfg.createRealizations

# Calculate upstream totals
calculateUpstreamTotals = mcfg.calculateUpstreamTotals

# Fixed regolith and vegetation states
fixedStates = mcfg.fixedStates

# Change geomorphology
changeGeomorphology = mcfg.changeGeomorphology

# Rainstorm parameters
rainstorm_probability = mcfg.rainstorm_probability
rainstorm_duration = mcfg.rainstorm_duration
rainstorm_expected_intensity = mcfg.rainstorm_expected_intensity
rainstorm_gamma_shape_param = mcfg.rainstorm_gamma_shape_param

# Surface store
maxSurfaceStoreValue = mcfg.maxSurfaceStoreValue

# Groundwater (saturated flow) and soil (moisture) parameters
saturatedConductivityMetrePerDayValue = mcfg.saturatedConductivityMetrePerDayValue
limitingPointFractionValue = mcfg.limitingPointFractionValue
mergeWiltingPointFractionFSValue = mcfg.mergeWiltingPointFractionFSValue
fieldCapacityFractionValue = mcfg.fieldCapacityFractionValue

# Soil porosity fraction value
soilPorosityFractionValue = mcfg.soilPorosityFractionValue

# Grazing increase start
# Make sure 0 <= rel_start_grazing < rel_end_grazing <= 1
rel_start_grazing = mcfg.rel_start_grazing
rel_end_grazing = mcfg.rel_end_grazing

# Carrying capacity
tot_increase_grazing = mcfg.tot_increase_grazing
initial_grazing = mcfg.initial_grazing
waterUseEfficiency = mcfg.waterUseEfficiency
maintenanceRate = mcfg.maintenanceRate

# Real-time of first time step of model run(s)
# ! - Note: This is now UTC time almost certainly at least for shading.
startTimeYearValue = mcfg.startTimeYearValue
startTimeMonthValue = mcfg.startTimeMonthValue
startTimeDayValue = mcfg.startTimeDayValue

# Shading
fractionReceivedValue = mcfg.fractionReceivedValue
fractionReceivedFlatSurfaceValue = mcfg.fractionReceivedFlatSurfaceValue


# ==================================================================
# REPORTING RASTERS (imported from EWS_main_configuration.py)
# ==================================================================

"""
Reporting model components of the pycatch model (see time_steps_to_report_all/some_hourly/weekly and 
timeStepsToReportRqs)
"""

interception_report_rasters = mcfg.interception_report_rasters
infiltration_report_rasters_weekly = mcfg.infiltration_report_rasters_weekly
infiltration_report_rasters = mcfg.infiltration_report_rasters
runoff_report_rasters = mcfg.runoff_report_rasters
subsurface_report_rasters = mcfg.subsurface_report_rasters
shading_report_rasters = mcfg.shading_report_rasters
surfacestore_report_rasters = mcfg.surfacestore_report_rasters
rainfalleventsfromgammadistribution_report_rasters = mcfg.rainfalleventsfromgammadistribution_report_rasters
exchange_report_rasters = mcfg.exchange_report_rasters
soilwashMMF_report_rasters = mcfg.soilwashMMF_report_rasters
regolith_report_rasters = mcfg.regolith_report_rasters
bedrockweathering_report_rasters = mcfg.bedrockweathering_report_rasters
evapotrans_report_rasters = mcfg.evapotrans_report_rasters
evapotranspirationsimple_report_rasters = mcfg.evapotranspirationsimple_report_rasters
biomassmodifiedmay_report_rasters = mcfg.biomassmodifiedmay_report_rasters
baselevel_report_rasters = mcfg.baselevel_report_rasters
creep_report_rasters = mcfg.creep_report_rasters
randomparameters_report_rasters = mcfg.randomparameters_report_rasters


# ==================================================================
# PLOT STYLING
# ==================================================================

"""
One configuration to rule them all
"""

# Default categorical colour cycle for EWS plots
EWS_colour_cycle = [
    "#1f77b4",  # muted blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
]

# Default line width for time series
EWS_linewidth = 1.8

# Default scatter style
EWS_scatter_size = 35
EWS_scatter_alpha = 0.85
EWS_scatter_edgewidth = 0.4
