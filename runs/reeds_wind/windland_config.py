# -*- coding: utf-8 -*-
"""
We need to create a list of grid ids and associate each with their SAM config
file key.
"""

# Set path to repo root a get functions
import revruns

# Signal Setup
print("Setting up reV run 'Windland V2'...\n")

# Create the configuration generator object
cnfg = revruns.Config(technology="wind", verbose=True)

# Set years explicitly
years = [y for y in range(2007, 2014)]

# Set common parameters
cnfg.top_params["allocation"] = "rev"
cnfg.top_params["feature"] = "--qos=high"
cnfg.top_params["logdir"] = "./logs"
cnfg.top_params["multi_year"] = True
cnfg.top_params["nodes"] = 10
cnfg.top_params["outdir"] = "./outputs"
cnfg.top_params["outputs"] = ["cf_profile", "cf_mean", "lcoe_fcr", "ws_mean"]
cnfg.top_params["resource"] = "wind_conus_v1"
cnfg.top_params["set_tag"] = "windland"
cnfg.top_params["tech"] = "wind"
cnfg.top_params["walltime"] = 6.0
cnfg.top_params["years"] = years
cnfg.points = "all"

# SAM Config #1  ( This was used for defaults excepting the coordinates )
cnfg.sam_params["wind_farm_xCoordinates"] = [0, 600, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400, 0, 600, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400, 0, 600, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400, 0, 600, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400, 0, 600, 1200, 1800, 2400, 3000, 3600, 4200, 4800, 5400]
cnfg.sam_params["wind_farm_yCoordinates"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 2400, 2400, 2400, 2400, 2400, 2400, 2400, 2400, 2400, 2400 ]
sam_config = cnfg.config_sam(jobname="t205_100mHH")

# SAM Config #2
# It's possible to just start over if there are no shared sam parameters
cnfg.sam_params = {}
cnfg.sam_params["adjust:constant"] = 0.0
cnfg.sam_params["system_capacity"] = 4500
cnfg.sam_params["wind_farm_losses_percent"] = 12.8
cnfg.sam_params["wind_farm_wake_model"] = 0
cnfg.sam_params["wind_farm_xCoordinates"] = [0]
cnfg.sam_params["wind_farm_yCoordinates"] = [0]
cnfg.sam_params["wind_resource_model_choice"] = 0
cnfg.sam_params["wind_resource_shear"] = 0.140
cnfg.sam_params["wind_resource_turbulence_coeff"] = 0.10
cnfg.sam_params["wind_turbine_cutin"] = 0.0
cnfg.sam_params["wind_turbine_hub_ht"] = 110.0
wtpo = revruns.WIND_SAM_PARAMS["wind_turbine_powercurve_powerout"]  # Default
wtws = revruns.WIND_SAM_PARAMS["wind_turbine_powercurve_windspeeds"]
cnfg.sam_params["wind_turbine_powercurve_powerout"] = wtpo
cnfg.sam_params["wind_turbine_powercurve_windspeeds"] = wtws
cnfg.sam_params["wind_turbine_rotor_diameter"] = 167.0
cnfg.sam_params["capital_cost"] = 5512500
cnfg.sam_params["fixed_operating_cost"] = 175275
cnfg.sam_params["fixed_charge_rate"] = 0.052
cnfg.sam_params["variable_operating_cost"] = 0
sam_config = cnfg.config_sam(jobname="t205")

# reV Configs
file = cnfg.config_all(excl_pos_lon=True)
