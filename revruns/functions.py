
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for revruns/brainstorming intuitive and easy config building

Created on Tue Nov 12 13:15:55 2019

@author: twillia2
"""
import datetime as dt
import h5py as hp
import json
import os
import numpy as np
import pandas as pd
#import PySAM as sam
#from PySAM import Pvwattsv5 as pv
from reV.utilities.exceptions import JSONError


# Data path
_root = os.path.abspath(os.path.dirname(__file__))
def dp(path):
    return os.path.join(_root, 'data', path)

# Time check
NOW = dt.datetime.today().strftime("%Y-%m-%d %I:%M %p")

# More to add: e.g. the 2018 CONUS solar is 2,091,566, and full is 9,026,712
RESOURCE_DIMS = {
        "nsrdb_v3": 2018392,
        "wind_conus_v2": 2488136
        }

RESOURCE_LABELS = {
        "nsrdb_v3": "National Solar Radiation Database - v3.0.1",
        "wind_conus_v2": ("Wind Integration National Dataset (WIND) " +
                          "Toolkit - CONUS, v2.0.0")
        }

RESOURCE_DSETS = {
        "nsrdb_v3": "/datasets/NSRDB/v3.0.1/nsrdb_{}.h5",
        "wind_conus_v2": "/datasets/WIND/conus/v2.0.0/wtk_conus_2014.h5"
        }


# Default parameters
TOP_PARAMS = dict(logdir="./logs",
                  loglevel="INFO",
                  outdir="./",
                  outputs="cf_mean",
                  years="all",
                  pointdir="./project_points",
                  allocation="rev",
                  feature="--qos=normal",
                  nodes=1,
                  option="eagle",
                  sites_per_core=100,
                  walltime=0.5)

SAM_PARAMS = dict(system_capacity=5,
                  dc_ac_ratio=1.1,
                  inv_eff=96,
                  losses=14.0757,
                  adjust_constant=14.0757,
                  gcr=0.4,
                  tilt=20,
                  azimuth=180,
                  array_type=0,
                  module_type=0,
                  compute_module="pvwattsv5")
 

def box_points(bbox, crd_path=dp("nsrdb_v3_coords.csv")):
    """Filter grid ids by geographical bounding box"""
    # Resource coordinate data frame from get_coordinates
    grid = pd.read_csv(crd_path)

    # Filter the data frame for points within the bounding box
    sample = grid[(grid["lat"] > bbox[1]) &
                  (grid["lat"] < bbox[3]) &
                  (grid["lon"] > bbox[0]) &
                  (grid["lon"] < bbox[2])]

    # And get just the grid ids
    gids = sample.index.to_list()

    return gids


def check_config(config_file):
    """Check that a json file loads without error."""
    try:
        with open(config_file, "r") as file:
            json.load(file)
        print("'" + config_file + "' opens.")
    except json.decoder.JSONDecodeError as error:
        emsg = ('JSON Error:\n{}\nCannot read json file: '
                '"{}"'.format(error, config_file))
        raise JSONError(emsg)


def get_coordinates(file, savepath):
    """Get all of the coordintes and their grid ids from an hdf5 file"""
    # Get numpy array of coordinates
    with hp.File(file, mode="r") as f:
        crds = f["coordinates"][:]

    # Create a data frame and save it
    lats = crds[:, 0]
    lons = crds[:, 1]
    df = pd.DataFrame({"lat": lats, "lon": lons})
    df.to_csv(savepath, index=False)




def project_points(jobname, resource="nsrdb_v3", points=1000):
    """Generates a required point file for spatial querying in reV.

    Parameters:
        jobname (str): Job name assigned to SAM configuration.
        resource (str): Energy resource data set key. Set to None for options.
        points (int | str | list): Sample points to generate. Set to an
                                   integer, n, to use the first n grid IDs,
                                   set to a list of points to use those points,
                                   or set to the string "all" to use all
                                   available points in the chosen resource
                                   data set.
        coords (list): A list of geographic coordinates to be converted to grid
                       IDs. (not yet implemented)

    Returns:
        pandas.core.frame.DataFrame: A data frame of grid IDs and SAM config
                                     keys (job name).
    """
    # Create a project_points folder if it doesn't exist
    if not os.path.exists("project_points"):
        os.mkdir("project_points")

    # Print out options
    if not resource:
        print("Resource required...")
        print("Available resource datasets: ")
        for key, var in RESOURCE_LABELS.items():
            print("   '" + key + "': " + str(var))

    # Sample or full grid?
    if type(points) is int:
        gridids = np.arange(0, points)
    elif points == "all":
        gridids = np.arange(0, RESOURCE_DIMS[resource])
    else:
        gridids = points

    # Create data frame
    points = pd.DataFrame({"gid": gridids, "config": jobname})

    # Return data frame
    return points


class Config:
    """Sets reV model key values and generates configuration json files."""
    def __init__(self,
                 top_params=TOP_PARAMS.copy(),
                 sam_params=SAM_PARAMS.copy()):
        self.top_params = top_params
        self.sam_params = sam_params
        self._set_years()

    def config_sam(self, jobname="gen", resource="nsrdb_v3", points=1000):
        """Configure the System Advisor Model (SAM) portion of a reV model.

        Parameters:
            jobname (str): Job name assigned to SAM configuration.
            resource (str): Energy resource data set key. Set to None for
                            options.
            points (int | str | list): Sample points to generate. Set to an
                                       integer, n, to use the first n grid IDs,
                                       set to a list of points to use those
                                       points, or set to the string "all" to
                                       use all available points in the chosen
                                       resource data set.

        Returns:
            dict: A dictionary of default and user specified SAM parameters.
            file: A local json file
        """
        # Make sure there is a sam config folder
        if not os.path.exists("./sam_configs"):
            os.mkdir("./sam_configs")

        # Separate parameters for space
        params = self.sam_params

        # Create the dictionary from the current set of parameters
        config_dict = {
            "system_capacity" : params["system_capacity"],
            "dc_ac_ratio" : params["dc_ac_ratio"],
            "inv_eff" : params["inv_eff"],
            "losses" : params["losses"],
            "adjust:constant" : params["adjust_constant"],
            "gcr" : params["gcr"],
            "tilt" : params["tilt"],
            "azimuth" : params["azimuth"],
            "array_type" : params["array_type"],
            "module_type" : params["module_type"],
            "compute_module" : params["compute_module"]
        }

        # Create project points
        proj_points = project_points(jobname, resource, points)
        point_path = os.path.join("project_points", jobname + ".csv")
        proj_points.to_csv(point_path)
        print(jobname + " project points" + " saved to '" + point_path + "'.")

        # Save to json using jobname for file name
        config_path = os.path.join("sam_configs", jobname + ".json")
        with open(config_path, "w") as file:
            file.write(json.dumps(config_dict))
        print(jobname + " SAM config file saved to '" + config_path + "'.")

        # Check that the json as written correctly
        check_config(config_path)

        # Return configuration dictionary 
        return config_dict


    def config_gen(self, tech="pv", jobname="gen"):
        """
        create a generation config file.
        """
        # Separate parameters for space
        params = self.top_params

        # Create the dictionary from the current set of parameters
        config_dict = {
            "directories": {
                "logging_directory": params["logdir"],
                "output_directory": params["outdir"]
            },
            "execution_control": {
                "allocation": params["allocation"],
                "feature": params["feature"],
                "nodes": params["nodes"],
                "option": params["option"],
                "sites_per_core": params["sites_per_core"],
                "walltime": params["walltime"]
            },
            "project_control": {
                "analysis_years": params["years"],
                "logging_level": params["loglevel"],
                "name": jobname,
                "output_request": params["outputs"],
                "technology": tech
            },
            "project_points": os.path.join(params["pointdir"],
                                           jobname + ".csv"),
            "resource_file": "/datasets/NSRDB/v3.0.1/nsrdb_{}.h5",
            "sam_files": {
                jobname: "./sam_configs/" + jobname + ".json"
            }
        }

        # Save to json using jobname for file name
        config_path = os.path.join("config_gen_" + jobname + ".json")
        with open(config_path, "w") as file:
            file.write(json.dumps(config_dict))
        print(jobname + " GEN config file saved to '" + config_path + "'.")

        # Check that the json as written correctly
        check_config(config_path)

        # Return configuration dictionary
        return config_dict

    def _set_years(self):
        """Set years attribute to all available if not specified"""
        if self.top_params["years"] == "all":
            self.years = range(1998, 2016)