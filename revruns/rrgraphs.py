#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A CLI to quickly generate some summary statistics/charts and graphs. For now
this is just use the supply curve tables.

Created on Thu Apr 23 10:06:50 2020

@author: twillia2
"""

import json
import multiprocessing as mp
import os
import sys
import warnings

from glob import glob
from osgeo import gdal

import click
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pandas.core.common import SettingWithCopyWarning
from revruns import VARIABLE_CHECKS
from revruns.rrshape import h5_to_shape, csv_to_shape
from tqdm import tqdm


# Turn off setting with copy warning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# Help printouts
DIR_HELP = ("The root directory of the reV project containing configuration"
            "files (Defaults to '.').")
SAVE_HELP = ("The path to use for the output graph images (defaults to " +
             "'./graphs').")
SUPPLY_UNITS = {'mean_cf': 'unitless',
                'mean_lcoe': '$/MWh',
                'mean_res': 'm/s',
                'capacity': 'MW',
                'area_sq_km': 'square km',
                'trans_capacity': 'MW',
                'trans_cap_cost': '$/MW',
                'lcot': '$/MWh',
                'total_lcoe': '$/MWh'}


def get_sc_path(project_dir):
    """Infer which outputs to check from the configuration files. Generation
    may or may not be present.

    Sample Argument
    --------------
    project_dir = "/shared-projects/rev/projects/india/forecast/pv"
    """

    # list configuration files
    config_files = glob(os.path.join(project_dir, "*.json"))

    # Try to find the supply-curve configuration
    try:
        config_file = [f for f in config_files if "supply-curve" in f][0]
    except:
        raise OSError("Supply Curve configuration not found.")

    # Read the configuration file in as a dictionary
    with open(config_file) as file:
        sc_config = json.load(file)

    # Use the configuration to find the file or output directory
    outdir = sc_config["directories"]["output_directory"]
    if "./" in outdir:
        outdir = os.path.join(project_dir, outdir.replace("./", ""))    
    csv_files = glob(os.path.join(outdir, "*.csv"))
    try:
        sc_path = [f for f in csv_files if "_sc.csv" in f][0]
    except:
        raise OSError("Supply Curve table not found.")

    return sc_path 


def supply_histograms(directory, savedir, show=False):
    """Create histograms of variables from supply curve tables.

    Sample Arguments
    ----------------
    directory = "/shared-projects/rev/projects/india/forecast/wind"
    savedir = "/shared-projects/rev/projects/india/forecast/wind/graphs"
    """

    print("Saving histograms of supply-curve variables to " + savedir)

    # No popups?
    plt.ioff()

    # Create paths
    sc_path = get_sc_path(directory)
    graph_dir = os.path.join(savedir, "sc_histograms")
    os.makedirs(graph_dir, exist_ok=True)

    # Now we can compare within each scenario
    df = pd.read_csv(sc_path, low_memory=False)

    # Loop through these variables and create a graph for each
    for variable, units in SUPPLY_UNITS.items():

        # Save path
        save_path = os.path.join(graph_dir, variable + ".png")  

        # Get data
        data = df[variable].copy()

        # Check for infinite values
        if data[np.isinf(data)].shape[0] > 0:
            ninf = data[np.isinf(data)].shape[0]
            total = data.shape[0]
            warnings.warn("Infinite values in " + variable + ". " + str(ninf) +
                          " out of " + str(total) + " values.")
            data[np.isinf(data)] = np.nan

        # Graph
        fig, ax = plt.subplots()
        n, bins, patches = ax.hist(data, 100)
        ax.set_xlabel(units)
        ax.set_ylabel("Count")
        ax.set_title(variable)
        fig.tight_layout()
        # plt.show()
        plt.savefig(save_path)
        plt.close(fig)


# The command
@click.command()
@click.option("--directory", "-d", default=".", help=DIR_HELP)
@click.option("--savedir", "-s", default="./graphs", help=SAVE_HELP)
def main(directory, savedir):
    """
    revruns Graphs

    Creates a set of graphs to quickly check a variety of reV outputs.

    directory = "/shared-projects/rev/projects/india/forecast/wind"
    savedir = "/shared-projects/rev/projects/india/forecast/wind/graphs"
    """

    # Go to that directory
    os.chdir(directory)

    # Expand paths for the csv
    directory = os.path.expanduser(directory)
    directory = os.path.abspath(directory)
    savedir = os.path.expanduser(savedir)
    savedir = os.path.abspath(savedir)

    # Make the save directory
    os.makedirs(savedir, exist_ok=True)

    # Generation map sample
    
    # Supply curve map sample
    
    # Supply curve histograms
    supply_histograms(directory, savedir)


if __name__ == "__main__":
    main()
