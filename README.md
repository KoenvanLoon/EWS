# EWS

Copyright Koen van Loon & Tijmen Janssen

Early Warning Signal (EWS) calculations for modified pycatch models (original models can be found here:  https://github.com/computationalgeography/pycatch )

Short user manual
=================

Installation
-----------------

See https://github.com/computationalgeography/pycatch/blob/master/readme.txt

- Create a conda environment by running pcraster_pycatch.yaml


Configuring the models
-----------------

Settings for the modified pycatch models are found in EWS_configuration.py

In EWS_StateVariables.py one can specify parameters/variables (such as datatype, window size) for the state variables. Note that state variables used for EWS calculations should also be included in the configuration.

Running the models
-----------------

Two models are present:
- To run the model with 1 hour timesteps, run EWS_pycatch_hourly.py (erosion excluded)
- To run the model with 1 week timesteps, run EWS_pycatch_weekly.py (erosion included, but not documented)


Calculate EWS
-----------------

To calculate the EWS, run either EWS_weekly.py or EWS_hourly.py for the respective model. 

- EWS_weekly.py and EWS_hourly.py make use of EWSPy.py for the Early Warning Signal calculations (for both spatial and temporal datasets).
  See these files for more information on the methods present.

- Null model datasets can also be generated for 3 different methods with NULL_models_timeseries_weekly.py and NULL_models_spatial_weekly.py 
  ! - Note that EWS_weekly.py automatically runs the functions in these files through the configuration settings.
  See these files for more information on the methods present.
 
Plotting
-----------------
To plot the results, run EWS_weekly_plots.py or EWS_hourly_plots.py respectively. User input is needed for this step.
See these files for more information.

Statistical tests
-----------------

EWS_Tests.py contains multiple functions to test for statistical significance (Kendall's Tau) of trends found in EWS.

- A comparison between the Tau values of the dummy dataset and model run can be made in both a histogram plot and numbers.
- Window size, linear detrending, and Gaussian filter size can be optimized in terms of Tau value and it's p-value of a specific EWS trend.
