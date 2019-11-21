# -*- coding: utf-8 -*-
"""
Get POA profile for Los Angeles

Created on Mon Nov 18 09:07:00 2019

@author: twillia2
"""

# Set path to repo root a get functions
from revruns import Config, shape_points

# Signal Setup
print("Setting up reV run #2...\n")

# Create general config object
cnfg = Config()

# Local and remote city shapefiles
city_shpr = ("https://opendata.arcgis.com/datasets/" +
             "7b0998f4e2ea42bda0068afc8eeaf904_19.zip")

# Get a list of target grid IDS
points = shape_points(city_shpr)

# Set common parameters
cnfg.top_params["years"] = [2015]
cnfg.top_params["outdir"] = "./output"
cnfg.top_params["logdir"] = "./output/logs"
cnfg.top_params["outputs"] = ["poa", "cf_profile"]
cnfg.top_params["allocation"] = "rev"
cnfg.top_params["walltime"] = 0.1
cnfg.top_params["nodes"] = 1
cnfg.sam_params["system_capacity"] = 5
cnfg.sam_params["dc_ac_ratio"] = 1.1

# Job #1
cnfg.sam_params["array_type"] = 0
cnfg.sam_params["tilt"] = "latitude"
sam_config = cnfg.config_sam(jobname="pvwattsv5_fixed", points=points)
gen_config = cnfg.config_gen(jobname="pvwattsv5_fixed", tech="pv")

# Job #2
cnfg.sam_params["array_type"] = 2
cnfg.sam_params["tilt"] = 0
sam_config = cnfg.config_sam(jobname="pvwattsv5_tracking", points=points)
gen_config = cnfg.config_gen(jobname="pvwattsv5_tracking", tech="pv")
