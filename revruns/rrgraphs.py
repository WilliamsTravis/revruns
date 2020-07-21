# -*- coding: utf-8 -*-
"""Quickly generate some summary statistics/charts and graphs. Incomplete.
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
from revruns.constants import VARIABLE_CHECKS
from revruns.rrshape import h5_to_shape, csv_to_shape
from revruns.rrlogs import status_dataframe
from tqdm import tqdm


# Turn off setting with copy warning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# Help printouts
DIR_HELP = ("The root directory of the reV project containing configuration"
            "files and outputs (Defaults to '.').")
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
MPATTERNS = ["_gen", "_multi-year", "_rep-profiles", "_agg", "_sc"]


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


def get_files(folder):
    """Get the output files and jobnames."""
    status = status_dataframe(folder)
    status = status[status["job_status"] == "successful"]
    join = lambda x: os.path.join(x["dirout"], x["fout"])
    files = status.apply(join, axis=1)
    return files


def gen_boxplots(files, savedir):
    """"Create boxplots for each generation variable in a yearly or multiyear
    generation output. ONly works for multi-year atm."""

    # Find the multiyear file
    try:
        file = files[files.index.str.contains("multi-year")].values[0]
    except:
        files = files.values   # <--------------------------------------------- not worked in yet

    # Outlier and non-outlier folders
    nout_folder = os.path.join(savedir, "gen_boxplots", "noutliers")
    out_folder = os.path.join(savedir, "gen_boxplots", "outliers")
    os.makedirs(nout_folder, exist_ok=True)
    os.makedirs(out_folder, exist_ok=True)

    # Infer datasetsmulti
    keys = list(h5py.File(file, "r").keys())
    exclude_chars = ["time", "meta", "profile"]
    datasets = [k for k in keys if not any([e in k for e in exclude_chars])]
    stats = np.unique([d.split("-")[0] for d in datasets])
    stat_dict = {s: [d for d in datasets if s in d] for s in stats} 

    with h5py.File(file, "r") as ds:
        for stat, datasets in stat_dict.items():
            stat_label = " ".join([s for s in stat.split("_")]).upper()
            data = []
            labels = []
            units = ds[datasets[0]].attrs["units"]
            scale = ds[datasets[0]].attrs["scale_factor"]
            values = [ds[d][:] / scale for d in datasets if "stdev" not in d]
            labels = [d.split("-")[1] for d in datasets if "std" not in d]

            # With outliers
            fig, ax = plt.subplots()
            ax.set_title(" - ".join([stat_label]))
            plt.xticks(rotation=45)
            plt.ylabel(units)
            ax.boxplot(values, labels=labels, showfliers=True)
            save_path = os.path.join(out_folder, "_".join([stat]))
            plt.savefig(save_path + ".png")

            # Without outliers
            fig, ax = plt.subplots()
            ax.set_title(" - ".join([stat_label, description]))
            plt.xticks(rotation=45)
            plt.ylabel(units)
            ax.boxplot(data, labels=labels, showfliers=False)
            save_path = os.path.join(nout_folder, "_".join([scenario, stat]))
            plt.savefig(save_path + ".png")


def supply_histogram(files, savedir, show=False):
    """Create histograms of variables from supply curve tables.

    Sample Arguments
    ----------------
    folder = "/shared-projects/rev/projects/india/forecast/wind"
    savedir = "/shared-projects/rev/projects/india/forecast/wind/graphs"
    """

    print("Saving histograms of supply-curve variables to " + savedir)

    # No popups?
    plt.ioff()

    # Create paths
    sc_path = files[files.index.str.contains("_sc")].values[0]
    graph_dir = os.path.join(savedir, "sc_histograms")
    os.makedirs(graph_dir, exist_ok=True)

    # Now we can compare within each scenario
    df = pd.read_csv(sc_path, low_memory=False)

    # Loop through these variables and create a graph for each
    for variable, units in SUPPLY_UNITS.items():

        # For simple supply curves, we might not have a few columns
        if variable in df.columns:
            
            # Save path
            save_path = os.path.join(graph_dir, variable + ".png")  
    
            # Get data
            data = df[variable].copy()
    
            # Check for infinite values
            if data[np.isinf(data)].shape[0] > 0:
                ninf = data[np.isinf(data)].shape[0]
                total = data.shape[0]
                warnings.warn("Infinite values in " + variable + ". "
                              + str(ninf) + " out of " + str(total)
                              + " values.")
                data[np.isinf(data)] = np.nan
    
            # Graph
            fig, ax = plt.subplots()
            n, bins, patches = ax.hist(data, 100)
            ax.set_xlabel(units)
            ax.set_ylabel("Count")
            ax.set_title(variable)
            fig.tight_layout()
    #        plt.show()
            plt.savefig(save_path)
            plt.close(fig)


# The command
@click.command()
@click.option("--folder", "-d", default=".", help=DIR_HELP)
@click.option("--savedir", "-s", default="./graphs", help=SAVE_HELP)
def main(folder, savedir):
    """
    revruns Graphs

    Creates a set of graphs to quickly check a variety of reV outputs.

    folder = "/shared-projects/rev/projects/iraq/rev/solar/aggregation/fixed"
    savedir = "/shared-projects/rev/projects/iraq/rev/solar/aggregation/fixed/wind/graphs"
    """

    # Go to that directory
    os.chdir(folder)

    # Expand paths for the csv
    folder = os.path.expanduser(folder)
    folder = os.path.abspath(folder)
    savedir = os.path.expanduser(savedir)
    savedir = os.path.join(folder, "graphs")

    # Make the save directory
    os.makedirs(savedir, exist_ok=True)

    # We need to run the appropriate methods for the available modules
    files = get_files(folder)
    jobs = files.index

    if any(["_gen" in j for j in jobs]) or any(["multi" in j for j in jobs]):

        # Generation boxplots
        gen_boxplots(files, savedir)

        # Generation map sample
        

    # Supply Curves    
    if any(["_sc" in j for j in jobs]):

        # Supply curve histograms
        supply_histogram(files, savedir)

        # Supply curve map sample


if __name__ == "__main__":
    main()
