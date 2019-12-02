# Import configuration about and point filter
from revruns import Config, box_points

# Signal Setup
print("Setting up reV run 'nsrdb_ensemble'...\n")

# Create general config object
cnfg = Config()

# Points from a bounding box around Denver's portion of the Front Range
bbox = [-105.352679491, 39.4595438351, -104.9022400379, 40.3518303006]
points = box_points(bbox)

# Set years explicitly
years = [y for y in range(2015, 2019)]

# Set common parameters
cnfg.top_params["allocation"] = "rev"
cnfg.top_params["logdir"] = "./"
cnfg.top_params["memory"] = 192
cnfg.top_params["nodes"] = 25
cnfg.top_params["outdir"] = "./"
cnfg.top_params["outputs"] = ["cf_profile", "cf_mean", "poa"]
cnfg.top_params["resource"] = "nsrdb_v3"
cnfg.top_params["set_tag"] = "nsrdb"
cnfg.top_params["walltime"] = 1.0
cnfg.top_params["years"] = years
cnfg.points = points
cnfg.sam_params["dc_ac_ratio"] = 1.1
cnfg.sam_params["system_capacity"] = 5

# SAM Config #1
cnfg.sam_params["array_type"] = 0
cnfg.sam_params["tilt"] = "latitude"
sam_config = cnfg.config_sam(jobname="fixed")

# SAM Config #2
cnfg.sam_params["array_type"] = 2
cnfg.sam_params["tilt"] = 0
sam_config = cnfg.config_sam(jobname="tracking")

# And this should trigger all of the other configuration files
gen_config = cnfg.config_all(excl_pos_lon=True)
