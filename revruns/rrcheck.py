# -*- coding: utf-8 -*-
"""
A CLI to quickly check if HDF files have potentially anomalous values. This
will check all files with an 'h5' extension in the current directory.
The checks, so far, only include whether the minimum and maximum values are
within the ranges defined in the "VARIABLE_CHECKS" object.

Created on Mon Nov 25 08:52:00 2019

@author: twillia2
"""
import click
import h5py
import json
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import sys
from glob import glob
from osgeo import gdal
from revruns import VARIABLE_CHECKS
from tqdm import tqdm

# Help printouts
DIR_HELP = "The directory from which to read the hdf5 files (Defaults to '.')."
SAVE_HELP = ("The path to use for the csv output file (defaults to " +
             "'./checkvars.csv').")

def single_info(file):
    """Return summary statistics of all data sets in a single hdf5 file.

    So, GDAL handles the multi-dimesional data sets fairly quickly, but doesn't
    even detect the singular dimensional data sets. Actually, it doesn't always
    detect the multidimensional datasets!

    For the 1-D data sets, we could probably just use 5py, but definitely not
    for the 2-D ones. However, if it is inconsistently detecting even these,
    what to do?

    First, see if this has happened to anyone else...

    """
    # GDAL For multidimensional data sets
    pointer = gdal.Open(file)

    # Get the list of sub data sets in each file
    subds = pointer.GetSubDatasets()

    # If there was only one detectable data set, we'll need to use this instead
    if len(subds) == 0:
        subds = [(pointer.GetDescription(),)]

    # For each of these sub data sets, get an info dictionary
    stat_dicts = []
    for sub in subds:

        # Turn off most options to try and speed this up
        info_str = gdal.Info(sub[0],
                             stats=True,
                             showFileList=True,
                             format="json",
                             listMDD=False,
                             approxStats=False,
                             deserialize=False,
                             computeMinMax=False,
                             reportHistograms=False,
                             reportProj4=False,
                             computeChecksum=False,
                             showGCPs=False,
                             showMetadata=False,
                             showRAT=False,
                             showColorTable=False,
                             allMetadata=False,
                             extraMDDomains=None)

        # Read this as a dictionary
        info = json.loads(info_str)
        desc = info["description"]
        ds = desc[desc.index("//") + 2: ]  # data set name
        stats = info["bands"][0]
        max_threshold = VARIABLE_CHECKS[ds][1]
        min_threshold = VARIABLE_CHECKS[ds][0]

        # Return just these elements
        stat_dict = {"file": file,
                     "data_set": ds,
                     "min": stats["minimum"],
                     "max": stats["maximum"],
                     "mean": stats["mean"],
                     "std": stats["stdDev"],
                     "min_threshold": min_threshold,
                     "max_threshold": max_threshold}
        stat_dicts.append(stat_dict)

    # To better account for completed data sets, make a data frame
    gdal_data = pd.DataFrame(stat_dicts)

    # H5py for one-dimensional data sets
    with h5py.File(file) as data_set:
        keys = data_set.keys()
        keys = [k for k in keys if k not in gdal_data["data_set"].values]
        keys = [k for k in keys if k not in ["meta", "time_index"]]
        data_sets = {k: data_set[k][:] for k in keys}

    # Now we have to calculate these "manually" (lol)
    stat_dicts = []
    for ds in data_sets.keys():
        values = data_sets[ds]
        if ds in VARIABLE_CHECKS:
            max_threshold = VARIABLE_CHECKS[ds][1]  # Figure out the thresholds for these
            min_threshold = VARIABLE_CHECKS[ds][0]
        else:
            max_threshold = np.nan
            min_threshold = np.nan
        minv = np.min(values)
        maxv = np.max(values)
        meanv = np.mean(values)
        stdv = np.std(values)
        stat_dict = {"file": file,
                     "data_set": ds,
                     "min": minv,
                     "max": maxv,
                     "mean": meanv,
                     "std": stdv,
                     "min_threshold": min_threshold,
                     "max_threshold": max_threshold}
        stat_dicts.append(stat_dict)

    # Make another data frame with the 1-D data set statistics
    hdf_data = pd.DataFrame(stat_dicts)

    # Concatenate the two together
    summary_table = pd.concat([gdal_data, hdf_data]).reset_index(drop=True)

    return summary_table


# The command
@click.command()
@click.option("--directory", "-d", default=".", help=DIR_HELP)
@click.option("--savepath", "-p", default="./checkvars.csv", help=SAVE_HELP)
def main(directory, savepath):
    """Checks all hdf5 files in a current directory for threshold values in
    data sets. This uses GDALINFO and also otputs an XML file with summary
    statistics and attribute information for each hdf5 file.
    """
    # Expand path
    directory = os.path.abspath(".")

    # Get and open files.
    files = glob(os.path.join(directory, "*h5"))

    # How many cpus do we have?
    ncores = mp.cpu_count()

    # Create a multiprocessin pool
    pool = mp.Pool(ncores - 1)

    # Try to dl in parallel with progress bar
    info_dfs = []
    for info_df in tqdm(pool.imap(single_info, files),
                        total=len(files), position=0,
                        file=sys.stdout):
        info_dfs.append(info_df)

    # Close pool object
    pool.close()

    # Combine output into single data frame
    info_df = pd.concat(info_dfs)
    info_df = info_df.reset_index(drop=True)

    # Write to file
    info_df.to_csv(savepath, index=False)

if __name__ == "__main__":
    main()
