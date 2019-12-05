#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 09:37:26 2019

@author: twillia2
"""
# Import configuration about and point filter
from revruns import Config, conus_points

# Signal Setup
print("Setting up reV run 'reeds_solar'...\n")

# Create general config object
cnfg = Config()

# Points from CONUS
points = conus_points(resource="nsrdb_v3")

# Set years explicitly
years = [y for y in range(1998, 2019)]

# Set common parameters
cnfg.top_params["allocation"] = "rev"
cnfg.top_params["logdir"] = "./logs"
cnfg.top_params["keep_chunks"] = True
cnfg.top_params["memory"] = 192
cnfg.top_params["nodes"] = 15
cnfg.top_params["outdir"] = "./"
cnfg.top_params["outputs"] = ["cf_profile", "cf_mean", "ghi_mean", "lcoe_fcr"]
cnfg.top_params["resource"] = "nsrdb_v3"
cnfg.top_params["set_tag"] = "nsrdb"
cnfg.top_params["walltime"] = 4.0
cnfg.top_params["years"] = years
cnfg.points = points
cnfg.sam_params["dc_ac_ratio"] = 1.3
cnfg.sam_params["system_capacity"] = 20000

# SAM Config #1
cnfg.sam_params["array_type"] = 2
cnfg.sam_params["tilt"] = 0
sam_config = cnfg.config_sam(jobname="tracking")

# And this should trigger all of the other configuration files
gen_config = cnfg.config_all(excl_pos_lon=True)
