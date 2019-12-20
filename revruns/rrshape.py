# -*- coding: utf-8 -*-
"""
It would be good to gbe able to make a quick shape file or geopackage out of
the csv outputs from rev (i.e., aggregations, supply-curves, and
representative profiles).

It would also be good to load the data into a data base that we could connect
to remotely via QGIS or something. That way (until we manage this x11
forwarding business) we can easily see maps of reV outputs.

Created on Tue Dec 17 2019

@author: twillia2
"""
import click
import geofeather as gfr
import geopandas as gpd
import json
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import sys
from glob import glob
from osgeo import gdal
from revruns import VARIABLE_CHECKS
from shapely.geometry import Point
from tqdm import tqdm

# Use GDAL to catch GDAL exceptions
gdal.UseExceptions()

# Help printouts
FILE_HELP = "The csv file from which to create the shape file. (str)"
SAVE_HELP = ("The path to use for the output file. Defaults to current " +
             "directory with the basename of the csv file. (str)")
DRIVER_HELP = ("Save as a Geopackage ('gpkg') or ESRI Shapefile ('shp'). " +
              "Defaults to 'gpkg'. (str).")
FEATHER_HELP = ("Use feather formatted data. This is much quicker but cannot" +
                "be used with GIS programs. (boolean)")

# Different possible lat lon column names
COORD_NAMES = {"lat": ["latitude", "lat", "y"],
               "lon": ["longitude", "lon", "x"]}

DRIVERS = {"shp": "ESRI Shapefile",
           "gpkg": "GPKG"}


# Guess what columns are lat and lons
def guess_coords(csv):
    """Use some common coordinate labels to infer which columns are lat and lon
    in a pandas data frame."""
    columns = csv.columns
    columns = ["lat" if c in COORD_NAMES["lat"] else c for c in columns]
    columns = ["lon" if c in COORD_NAMES["lon"] else c for c in columns]
    csv.columns = columns

    return csv


def to_point(row):
    """Create a point object from a row with 'lat' and 'lon' columns"""
    point = Point((row["lon"], row["lat"]))
    return point


def csv_to_shape(file, driver="gpkg", savepath=None, feather=False):
    """Take in a csv file path, a shapefile driver, and output a shapefile

    This will be able to create a shapefile out of the csv outputs, but
    what about the hdf5 outputs? Should it be two different functions?
    Probably so, but would it be terribly confusing to make this do either?
    If it is an HDF we would have to supply the data set in addition to the
    file. For reV outputs we'd have to access the meta data and the numpy
    arrays, and if it is a profile we'll have ...perhaps it would be better to
    have a separate function that outputs rasters for  hdf5.

    Let's just do the 2-D csv outputs for now.

    Sample args:

    file = "/projects/rev/new_projects/reeds_solar/reeds_solar_sc.csv"
    driver = "gpkg"
    savepath = None
    feather = False
    """
    # Select driver
    try:
        driver_str = DRIVERS[driver]
    except KeyError:
        print("KeyError: Please provide or check the spelling of the driver " +
              "input.")

    # If no save path
    if not savepath:
        name = os.path.splitext(file)[0]
        savepath = name + "." + driver

    # Use the save path to get the layer name
    layer = os.path.splitext(os.path.basename(savepath))[0]

    # Read csv
    csv = pd.read_csv(file)

    # Standardize the coordinate column names
    csv = guess_coords(csv)

    # Create a geometry columns and a geodataframe
    csv["geometry"] = csv.apply(to_point, axis=1)
    gdf = gpd.GeoDataFrame(csv, geometry="geometry", crs={"init": 4326})
    gdf.crs = {"init": "epsg:4326"}

    # Save file
    if feather:
        # Save as with feather data types. Much quicker, but limited in use
        savepath = savepath.replace(driver, "feather")
        gfr.to_geofeather(gdf, savepath)
    else:
        # Save to shapefile or geopackage
        gdf.to_file(savepath, layer=layer, driver=driver_str)


@click.command()
@click.option("--file", "-f", help=FILE_HELP)
@click.option("--savepath", "-s", default=None, help=SAVE_HELP)
@click.option("--driver", "-d", default="gpkg", help=DRIVER_HELP)
@click.option("--feather", is_flag=True, help=FEATHER_HELP)
def main(file, savepath, driver, feather):
    """ Take a csv output from reV and write a shape file, geopackage, or
    geofeather file.
    """
    csv_to_shape(file=file, savepath=savepath, driver=driver, feather=feather)


if "__name__" == "__main__":
    main()
