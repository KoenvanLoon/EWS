# SPDX-License-Identifier: GPL-3.0-or-later
#
# Copyright (c) 2026 Koen van Loon
#
# See the LICENSE file in the repository root for full license text.

"""
EWS - Early Warning Signals
EWS State Variables

@authors: KoenvanLoon & TijmenJanssen
"""

import EWS_configuration as cfg

# Class StateVariable for Variable objects
"""
State variables used in all other EWS Python scripts are stored in the lists below. For each state variable, multiple
variables of these state variables, such as snapshot interval, window size and overlap, and datatype, can be defined
below as to guarantee the use of the same variables over the different EWS Python scripts.

    ! - Note that the 'full' sets of state variables are defined at the end of this file, and if state variables for EWS 
    are added, they also should be included here.
    
Args:
-----

name : string, short name for state variable (usually taken from model).

spatial : bool, selects whether data is spatial or not.

temporal : bool, selects whether data is temporal or not.

snapshot_interval : int, modelled time between saved spatial (map-based) data.

window_size : int, amount of datapoints, not physical time units, over which the early-warning signal is calculated (window).
                Interpretation depends on snapshot_interval and model (hourly/weekly).

window_overlap : int, amount of datapoints from previous window taken into the next window.

datatype : string, either 'map' for map-files or 'numpy' for numpy.txt-files.

full_name : string, full name of the state variable.

unit : string, unit of the state variable.
    

Example:
--------

Index Fund :

    INDF = StateVariable('INDF', temporal=True, datatype='numpy', window_size=30, window_overlap=0, 
        full_name='Index fund closing value', unit="Dollars ($)")
    
"""

variables_weekly = []
variables_hourly = []


class StateVariable:
    def __init__(self, name, spatial=False, temporal=False, snapshot_interval=cfg.interval_map_snapshots,          # Snapshot_interval defaults to cfg.interval_map_snapshots for consistency across variables
                 window_size=cfg.default_window_size, window_overlap=0, datatype=None, full_name='', unit='unit'):    # Default window size used for all variables unless overridden elsewhere
        self.name = name
        self.spatial = spatial
        self.temporal = temporal
        self.snapshot_interval = snapshot_interval
        self.window_size = window_size
        self.window_overlap = window_overlap
        self.datatype = datatype
        self.full_name = full_name
        self.unit = unit
        if not (spatial or temporal):
            raise ValueError(f"StateVariable '{name}' must be spatial or temporal.")

    def __repr__(self):
        kind = "spatial" if self.spatial else "temporal"
        return f"<StateVariable {self.name} ({kind})>"


# State variables for EWS #
"""
NAMING CONVENTION FOR STATE VARIABLES

All state variables follow the naming convention <base><suffix>
The suffix indicates the spatial-temporal representation of the data:

    M : Map-based spatial snapshot
        - PCRaster map
        - Represents the ful spatial field at a given timestep
        - Used for spatial EWS
        
    A : spatially Averaged temporal signal
        - 1D NumPy array
        - Each value represents the spatial mean at one timestep
        - Used for temporal EWS
        
    L : single-Location temporal signal
        - 1D NumPy array
        - Represents the value at a fixed reporting location
        - Used to detect local early warning signals
        
This is enforced consistently across the EWS code pipeline.

!Note on cumalative states vs. rate variables

Cumulative state variables such as growth and deposition, persist across timesteps and contain system memory,
    making them particularly suitable for EWS analysis. Rate-based variables represent per timestep fluxes or
    increments and are interpreted in EWS analysis as indicators of system sensitivity or response rather than stability
    of a stored state.
"""

# Maximum interception store
micM = StateVariable('micM', spatial=True, datatype='map', full_name='Maximum interception storage spatial', unit="m")
micA = StateVariable('micA', temporal=True, datatype='numpy', full_name='Maximum interception storage temporal', unit="m")
micL = StateVariable('micL', temporal=True, datatype='numpy', full_name='Maximum interception storage at location', unit="m")

# LAI
laiM = StateVariable('laiM', spatial=True, datatype='map', full_name='LAI spatial', unit="-")
laiA = StateVariable('laiA', temporal=True, datatype='numpy', full_name='LAI temporal', unit="-")
laiL = StateVariable('laiL', temporal=True, datatype='numpy', full_name='LAI at location', unit="-")

# Soil moisture
moiM = StateVariable('moiM', spatial=True, datatype='map', full_name='Soil moisture spatial', unit="- (fraction)")
moiA = StateVariable('moiA', temporal=True, datatype='numpy', full_name='Soil moisture temporal', unit="- (fraction)")
moiL = StateVariable('moiL', temporal=True, datatype='numpy', full_name='Soil moisture at location', unit="- (fraction)")

# Biomass
bioM = StateVariable('bioM', spatial=True, datatype='map', full_name='Biomass spatial', unit="kg m^-2")
bioA = StateVariable('bioA', temporal=True, datatype='numpy', full_name='Biomass temporal', unit="kg m^-2")
bioL = StateVariable('bioL', temporal=True, datatype='numpy', full_name='Biomass at location', unit="kg m^-2")

# Regolith thickness
regM = StateVariable('regM', spatial=True, datatype='map', full_name='Regolith thickness spatial', unit="m")
regA = StateVariable('regA', temporal=True, datatype='numpy', full_name='Regolith thickness temporal', unit="m")
regL = StateVariable('regL', temporal=True, datatype='numpy', full_name='Regolith thickness at location', unit="m")

# DEM
demM = StateVariable('demM', spatial=True, datatype='map', full_name='DEM spatial', unit="m")
demA = StateVariable('demA', temporal=True, datatype='numpy', full_name='DEM temporal', unit="m")
demL = StateVariable('demL', temporal=True, datatype='numpy', full_name='DEM at location', unit="m")

# Discharge
qA = StateVariable('qA', temporal=True, datatype='numpy', full_name='Discharge temporal', unit="m^3 per timestep")
Rq = StateVariable('Rq', temporal=True, datatype='numpy', full_name='Discharge', unit="m^3 per timestep")

# Grazing rate
gA = StateVariable('gA', temporal=True, datatype='numpy', full_name='Grazing rate temporal', unit="kg m^-2 Δt^-1")

# Growth part
gpM = StateVariable('gpM', spatial=True, datatype='map', full_name='Growth part spatial', unit="kg m^-2 Δt^-1")
gpA = StateVariable('gpA', temporal=True, datatype='numpy', full_name='Growth part temporal', unit="kg m^-2 Δt^-1")

# Grazing part
grM = StateVariable('grM', spatial=True, datatype='map', full_name='Grazing part spatial', unit="kg m^-2 Δt^-1")
grA = StateVariable('grA', temporal=True, datatype='numpy', full_name='Grazing part temporal', unit="kg m^-2 Δt^-1")

# Net growth (growth part + grazing)
grnM = StateVariable('grnM', spatial=True, datatype='map', full_name='Net growth spatial', unit="kg m^-2 Δt^-1")
grnA = StateVariable('grnA', temporal=True, datatype='numpy', full_name='Net growth temporal', unit="kg m^-2 Δt^-1")

# Net deposition
depM = StateVariable('depM', spatial=True, datatype='map', full_name='Net deposition spatial', unit="m h^-1 (model derived rate)")
depA = StateVariable('depA', temporal=True, datatype='numpy', full_name='Net deposition temporal', unit="m h^-1 (model derived rate)")
depL = StateVariable('depL', temporal=True, datatype='numpy', full_name='Net deposition at location', unit="m h^-1 (model derived rate)")

# Net weathering
weaM = StateVariable('weaM', spatial=True, datatype='map', full_name='Net weathering spatial', unit="m y^-1 (applied per model timestep)")
weaA = StateVariable('weaA', temporal=True, datatype='numpy', full_name='Net weathering temporal', unit="m y^-1 (applied per model timestep)")
weaL = StateVariable('weaL', temporal=True, datatype='numpy', full_name='Net weathering at location', unit="m y^-1 (applied per model timestep)")

# Net creep deposition
creM = StateVariable('creM', spatial=True, datatype='map', full_name='Net creep deposition spatial', unit="m (increment)")
creA = StateVariable('creA', temporal=True, datatype='numpy', full_name='Net creep deposition temporal', unit="m (increment)")
creL = StateVariable('creL', temporal=True, datatype='numpy', full_name='Net creep deposition at location', unit="m (increment)")


# Check which variables are present in the configuration and append these to the list of variables

full_set_of_variables_weekly = [micM, micA, micL, laiM, laiA, laiL, moiM, moiA, moiL, bioM, bioA, bioL, regM, regA,
                                regL, demM, demA, demL, qA, gA, gpM, gpA, grM, grA, grnM, grnA, depM, depA, depL, weaM,
                                weaA, weaL, creM, creA, creL]

# full_set_of_variables_weekly = [laiA, laiM, moiA, moiM, moiL, bioA, bioM, bioL, qA, grnA, grnM,
#                                 regA, regM, demA, demM, weaA, weaM, creA, creM, grA, gA]

full_set_of_variables_hourly = [Rq, moiA, moiM]


lookup_weekly = {v.name: v for v in full_set_of_variables_weekly}

if cfg.state_variables_for_ews_weekly == 'full':    # Magick String - Fragile
    variables_weekly = full_set_of_variables_weekly
else:
    variables_weekly = [lookup_weekly[name] for name in cfg.state_variables_for_ews_weekly if name in lookup_weekly]

    missing = [name for name in cfg.state_variables_for_ews_weekly if name not in lookup_weekly]

    if missing:
        raise ValueError(f"Unknown state variable(s): {missing}")

if not variables_weekly:
    raise ValueError("No weekly state variables selected.")


if cfg.state_variables_for_ews_hourly == 'full':    # Magick String - Fragile
    variables_hourly = full_set_of_variables_hourly
else:
    for state_variable in cfg.state_variables_for_ews_hourly:
        for variable in full_set_of_variables_hourly:
            if variable.name == state_variable:
                variables_hourly.append(variable)
