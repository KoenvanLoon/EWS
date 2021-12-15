# EWS

Copyright Koen van Loon & Tijmen Janssen

Early Warning Signal (EWS) calculations for modified pycatch models (originals can be found here:  https://github.com/computationalgeography/pycatch )

Short user manual
=================

Installation
-----------------

See https://github.com/computationalgeography/pycatch/blob/master/readme.txt

- Create a conda environment by running pcraster_pycatch.yaml


Configuring the models
-----------------

Settings for the modified pycatch models are in EWS_main_configuration.py

In EWS_StateVariables.py one can specify parameters/variables (such as window size) for the state variables. Note that this is linked to the configuration.

Running the models
-----------------

Two models are present:
- To run the model with 1 hour timesteps, run EWS_pycatch_hourly.py
- To run the model with 1 week timesteps, run EWS_pycatch_hourly.py (erosion included, but not documented)


Calculate EWS
-----------------

To calculate the EWS, run either EWS_weekly.py or EWS_hourly.py for the respective model. 

- EWS_weekly.py and EWS_hourly.py make use of EWSPy.py for the Early Warning Indicator calculations (for both spatial and temporal datasets).
  See these files for more information on the methods present.

- Null models can also be generated with 3 different methods with NULL_models_timeseries_weekly/hourly.py and NULL_models_spatial_weekly/hourly.py 
  See these files for more information on the methods present.
 
Plotting
-----------------
To plot the results, run EWS_weekly_plots.py or EWS_hourly_plots.py respectively. User input is needed for this step.
