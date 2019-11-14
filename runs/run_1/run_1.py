# -*- coding: utf-8 -*-
"""
We need to create a list of grid ids and associate each with their SAM config
file key.
"""

# Set path to repo root a get functions
from revruns import Config

# Signal Setup
print("Setting up reV run #1...\n")

# Create general config object
cnfg = Config()

# Set common parameters
cnfg.top_params["years"] = [2016, 2017]
cnfg.top_params["outdir"] = "./output"
cnfg.top_params["logdir"] = "./output/logs"
cnfg.top_params["outputs"] = ["cf_profile", "cf_mean", "poa"]
cnfg.top_params["allocation"] = "rev"
cnfg.top_params["walltime"] = 0.1
cnfg.top_params["nodes"] = 1
cnfg.sam_params["system_capacity"] = 5
cnfg.sam_params["dc_ac_ratio"] = 1.1

# Job #1
cnfg.sam_params["array_type"] = 0
cnfg.sam_params["tilt"] = "latitude"
sam_config = cnfg.config_sam(jobname="pvwattsv5_fixed", point_sample=1000)
gen_config = cnfg.config_gen(jobname="pvwattsv5_fixed", tech="pv")

# Job #2
cnfg.sam_params["array_type"] = 2
cnfg.sam_params["tilt"] = 0
sam_config = cnfg.config_sam(jobname="pvwattsv5_tracking", point_sample=1000)
gen_config = cnfg.config_gen(jobname="pvwattsv5_tracking", tech="pv")
