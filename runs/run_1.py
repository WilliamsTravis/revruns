# -*- coding: utf-8 -*-
"""
We need to create a list of grid ids and associate each with their SAM config
file key.
"""

# Set path to repo root a get functions
import geopandas as gpd
from revruns import Config, CONUS, box_points, shape_points

# Signal Setup
print("Setting up reV run #1...\n")

# Create general config object
cnfg = Config()

# Points from a bounding box around Denver's portion of the Front Range
bbox = [-105.352679491, 39.4595438351, -104.9022400379, 40.3518303006]
points = box_points(bbox)

# Just points in CONUS
#shp_url = ("https://www2.census.gov/geo/tiger/TIGER2019/STATE/" + 
#          "tl_2019_us_state.zip")
#shp = gpd.read_file(shp_url)
#shp = shp[shp['STUSPS'].isin(CONUS)]
#points = shape_points(shp)

# Set years explicitly
years = [y for y in range(1998, 2019)]

# Set common parameters
cnfg.top_params["allocation"] = "rev"
cnfg.sam_params["dc_ac_ratio"] = 1.1
cnfg.top_params["logdir"] = "./"
cnfg.top_params["memory_utilization_limit"] = 0.2
cnfg.top_params["nodes"] = 10
cnfg.top_params["outdir"] = "./"
cnfg.top_params["outputs"] = ["cf_profile", "cf_mean", "poa"]
cnfg.top_params["resource"] = "nsrdb_v3"
cnfg.top_params["set_tag"] = "set1"
cnfg.top_params["tech"] = "pv"
cnfg.top_params["walltime"] = 2.0
cnfg.sam_params["system_capacity"] = 5
cnfg.top_params["years"] = years
cnfg.points = points

# SAM Config #1
cnfg.sam_params["array_type"] = 0
cnfg.sam_params["tilt"] = "latitude"
sam_config = cnfg.config_sam(jobname="fixed")

# SAM Config #2
cnfg.sam_params["array_type"] = 2
cnfg.sam_params["tilt"] = 0
sam_config = cnfg.config_sam(jobname="tracking")

# reV Configs
file = cnfg.config_all()
