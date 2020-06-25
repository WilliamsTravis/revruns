# -*- coding: utf-8 -*-
"""
Write an HDF5 exclusions dataset to a geotiff, or write a stack of multiple
datasets to a geotiff according to an aggregating function.

Created on Mon Jun  8 13:42:12 2020

@author: twillia2
"""

import json
import multiprocessing as mp
import os

import click
import dask.array as da
import h5py
import numpy as np
import rasterio

from dask.distributed import Client
from osgeo import gdal
from shapely.geometry import Point
from tqdm import tqdm


# Use GDAL to catch GDAL exceptions
gdal.UseExceptions()

# Help printouts
FILE_HELP = "The file from which to create the shape file. (str)"
SAVE_HELP = ("The path to use for the output file. Defaults to current "
             "directory with the basename of the csv file. (str)")
DATASET_HELP = ("The HDF5 data set to render. (str)")
AGGFUN_HELP = ("The aggregating function to apply to the stack of rasters "
               "if multiple datasets are chosen. Most numpy summary "
               "functions will work. Defaults to 'max'. (str)")
CONFIG_HELP = ("A reV aggregation config. If this is provided all exclusion "
               "datasets in it will be aggregated together.")

# Different possible lat lon column names
OMISSIONS =  ["coordinates" ,"time_index", "meta", "latitude", "longitude"]
COORD_NAMES = {"lat": ["latitude", "lat", "y"],
               "lon": ["longitude", "lon", "long", "x"]}

# For multiprocess
def get_array(arg):
    """Get a single array out of an HDF5 dataset."""

    h5, dataset = arg
    with h5py.File(h5, "r") as excl:
        array = excl[dataset][:]

    return array


@click.command()
@click.option("--file", "-f", help=FILE_HELP)
@click.option("--savepath", "-s", default=None, help=SAVE_HELP)
@click.option("--dataset", "-ds", default=None, help=DATASET_HELP)
@click.option("--config", "-c", default=None, help=CONFIG_HELP)
@click.option("--aggfun", "-a", default=None, help=AGGFUN_HELP)
def main(file, savepath, dataset, config, aggfun="max"):
    """Write an HDF5 exclusions dataset to a geotiff, or write a stack of
    multiple datasets to a geotiff according to an aggregating function.

    Sample Parameters
    -----------------
    file = "/projects/rev/data/exclusions/ATB_Exclusions.h5"
    savepath = "/scratch/twillia2/16_tc_rl_mid_excl.tif"
    dataset = None
    config = "/shared-projects/rev/projects/weto/task_1/aggregation/16_tc_rl_mid/config_aggregation.json"
    aggfun = "max"
    """

    # If just one data set, write just that one
    if dataset:
        with h5py.File(file, "r") as excl:
            profile = json.loads(excl[dataset].attrs["profile"])
            array = excl[dataset][:]
        with rasterio.Env():
            with rasterio.open(savepath, 'w', **profile) as dst:
                dst.write(array)

    # If a config was provided use that to choose datasets
    if config:
        with open(config, "r") as cnfg:
            config = json.load(cnfg)
        datasets = config["excl_dict"].keys()

        # Collect arrays <----------------------------------------------------- 'OverflowError('cannot serialize a bytes object larger than 4 GiB')'
        # args = [(file, d) for d in datasets]
        # arrays = []
        # with mp.Pool(os.cpu_count()) as pool:
        #     for array in tqdm(pool.imap(get_array, args), total=len(args)):
        #         arrays.append(array)

        # Collect Arrays
        arrays = []
        navalue = 0
        with h5py.File(file, "r") as excl:
            for d in tqdm(datasets, position=0):
                profile = json.loads(excl[d].attrs["profile"])
                nodata = profile["nodata"]
                array = excl[d][0]  # These are shaped (1, y, x)
                array[array == nodata] = 0
                arrays.append(array)

                # Find the data type and the largest possible value for na
                try:
                    maxv = np.finfo(array.dtype).max
                except ValueError:
                    maxv = np.iinfo(array.dtype).max
                if maxv > navalue:
                    navalue = maxv

            # Find the function
            if "nan" not in aggfun:
                aggfun = "nan" + aggfun
            fun = np.__dict__[aggfun]

            # Make composite raster
            stack = np.stack(arrays)  # <-------------------------------------- Breaking here....need to standardize data types?
            composite = fun(stack, axis=0)

        # Write to target path
        with rasterio.Env():
            profile["no_data"] = 0
            profile["dtype"] = str(composite.dtype)
            profile["tiled"] = True
            with rasterio.open(savepath, 'w', **profile) as dst:
                dst.write(composite)


# if __name__ == "__main__":
#     main()

