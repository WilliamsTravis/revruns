# -*- coding: utf-8 -*-
"""
We need to create a list of grid ids and associate each with their SAM config
file key.
"""

# Set path to repo root a get functions
from revruns import check_config, project_points

# Fixed
fpoints = project_points("i_pvwatts_fixed", sample=1000)
fpoints.to_csv("project_points/project_points_fixed.csv",
               index=False)

# tracking
tpoints = project_points("i_pvwatts_tracking", sample=1000)
fpoints.to_csv("project_points/project_points_tracking.csv",
               index=False)

# Check config files
check_config("config_gen.json")
check_config("sam_configs/i_pvwatts_fixed.json")
check_config("sam_configs/i_pvwatts_tracking.json")
    
