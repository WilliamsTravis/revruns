# -*- coding: utf-8 -*-
"""
We need to create a list of grid ids and associate each with their SAM config
file key.
"""

# Set path to repo root a get functions
import os
_root = os.path.abspath(os.path.dirname("__file__"))
os.chdir(os.path.join(_root, "../.."))
from functions import check_config, project_points

# Fixed
fpoints = project_points("i_pwatts_fixed", sample=1000)
fpoints.to_csv("runs/run_1/project_points/project_points_fixed.csv")

# tracking
tpoints = project_points("i_pwatts_tracking", sample=1000)
fpoints.to_csv("runs/run_1/project_points/project_points_tracking.csv")

# Check config files
check_config("runs/run_1/config_gen.json")
check_config("runs/run_1/sam_configs/i_pvwatts_fixed.json")
check_config("runs/run_1/sam_configs/i_pvwatts_tracking.json")
    
