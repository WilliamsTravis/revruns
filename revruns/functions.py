#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for revruns,

Created on Tue Nov 12 13:15:55 2019

@author: twillia2
"""
import datetime as dt
import json
import os
import numpy as np
import pandas as pd
#import PySAM as sam
#from PySAM import Pvwattsv5 as pv
from reV.utilities.exceptions import JSONError

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


def check_config(config_file):
    '''
    Check that these jsons load properly
    '''
    try:
        with open(config_file, "r") as file:
            json.load(file)
        print(config_file + " opens")
    except json.decoder.JSONDecodeError as error:
        emsg = ('JSON Error:\n{}\nCannot read json file: '
                '"{}"'.format(error, config_file))
        raise JSONError(emsg)


def project_points(configid, resource="nsrdb_v3", sample=None):
    '''
    A way to generate either sample points or everypoint.
    It would also be useful to convert a list of coordinates into gridids.
    '''
    # create a project_points folder if it doesn't exist
    if not os.path.exists("project_points"):
        os.mkdir("project_points")

    # Print out options
    if not resource:
        print("Resource required...")
        print("Available resource datasets: ")
        for key, var in RESOURCE_LABELS.items():
            print("   '" + key + "': " + str(var))

    # Sample or full grid?
    if sample:
        gridids = np.arange(0, sample)
    else:
        gridids = np.arange(0, RESOURCE_DIMS[resource])

    # Create data frame
    points = pd.DataFrame({"gid": gridids, "config": configid})

    # Save or return data frame?
    return points


def test_function():
    '''
    Checking functionality
    '''
    print("revruns installed: " + NOW)
