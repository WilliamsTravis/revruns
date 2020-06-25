# -*- coding: utf-8 -*-
"""Make a quick shape file or geopackage out of the outputs from reV.
"""

import os

import click
import geopandas as gpd
import h5py
import pandas as pd

from osgeo import gdal
from shapely.geometry import Point
from tqdm import tqdm

# Use GDAL to catch GDAL exceptions
gdal.UseExceptions()


# Help printouts
FILE_HELP = "The file from which to create the shape file. (str)"
SAVE_HELP = ("The path to use for the output file. Defaults to current "
             "directory with the basename of the csv file. (str)")
LAYER_HELP = ("For hdf5 time series, the time layer to render. Defaults to "
              " 0. (int)")
DATASET_HELP = ("For hdf5 time series, the data set to render. Defaults to "
                "all available data sets. (str)")
DRIVER_HELP = ("Save as a Geopackage ('gpkg') or ESRI Shapefile ('shp'). "
              "Defaults to 'gpkg'. (str).")
CRS_HELP = ("A proj4 string or epsg code associated with the file's "
            "coordinate reference system. (str | int)" )
OMISSIONS =  ["coordinates" ,"time_index", "meta", "latitude", "longitude"]

# Different possible lat lon column names
COORD_NAMES = {"lat": ["latitude", "lat", "y"],
               "lon": ["longitude", "lon", "long", "x"]}

DRIVERS = {"shp": "ESRI Shapefile",
           "gpkg": "GPKG"}


# Decode HDF columns
def decode_cols(df):
    """Decode byte columns in a pandas data frame."""

    for col in df.columns:
        if isinstance(df[col].iloc[0], bytes):
            df[col] = df[col].apply(lambda x: x.decode())

    return df


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


def csv_to_shape(file, driver="gpkg", savepath=None, epsg=4326):
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

    file = "/shared-projects/rev/projects/weto/aggregation/05_b_b_mid/05_b_b_mid_sc.csv"
    driver = "gpkg"
    savepath = "/shared-projects/rev/projects/weto/aggregation/05_b_b_mid/05_b_b_mid_sc.gpkg"
    """
    # Select driver
    driver_str = DRIVERS[driver]

    # If no save path
    if not savepath:
        name = os.path.splitext(file)[0]
        savepath = name + "." + driver

    # Use the save path to get the layer name
    layer = os.path.splitext(os.path.basename(savepath))[0]

    # Read csv
    csv = pd.read_csv(file, low_memory=False)

    # Standardize the coordinate column names
    csv = guess_coords(csv)

    # Create a geometry columns and a geodataframe
    csv["geometry"] = csv.apply(to_point, axis=1)
    gdf = gpd.GeoDataFrame(csv, geometry="geometry")
    gdf.crs = "epsg:{}".format(epsg)

    # Save file
    gdf.to_file(savepath, layer=layer, driver=driver_str)


def h5_to_shape(file, driver="gpkg", dataset=None, layer=0, savepath=None,
                epsg=4326):

    """For now, this will just take a single time period as an index position
    in the time series.
    

    Parameters
    ----------
    file : TYPE
        DESCRIPTION.
    driver : TYPE, optional
        DESCRIPTION. The default is "gpkg".
    dataset : TYPE, optional
        DESCRIPTION. The default is "cf_mean".
    layer : TYPE, optional
        DESCRIPTION. The default is 0.
    savepath : TYPE, optional
        DESCRIPTION. The default is None.
    epsg : TYPE, optional
        DESCRIPTION. The default is 4326.

    Sample Arguments
    ----------------
    
    file = "/shared-projects/rev/projects/heco/data/resource/comparisons/differences.h5"
    savepath = "/shared-projects/rev/projects/heco/data/resource/comparisons/mean_windspeed_differences.gpkg"
    driver = "gpkg"
    dataset = None
    layer = 0
    epsg = 4326
    """

    # Select driver
    driver_str = DRIVERS[driver]

    # If no save path
    if not savepath:
        name = os.path.splitext(file)[0]
        savepath = name + "." + driver

    # Read hdf5 file
    arrays = {}
    with h5py.File(file, "r") as ds:
        keys = list(ds.keys())
        if "meta" in keys:
            meta = pd.DataFrame(ds["meta"][:])

        # else:
        #     coord_check = any([lat in keys for lat in COORD_NAMES["lat"]])
        #     if coord_check:
        #         latcol = [lat for lat in COORD_NAMES["lat"] if lat in keys ][0]
        #         loncol = [lon for lon in COORD_NAMES["lon"] if lon in keys ][0]
        #         lats = pd.DataFrame(ds[latcol][:])
        #         lons = pd.DataFrame(ds[loncol][:])
        #         meta = lons.join(lats)
        else:
            raise KeyError("Meta data not found.")


        if dataset:
            scale = ds[dataset].attrs["scale_factor"]
            arrays[dataset] = ds[dataset][:] #/ scale
        else:
            datasets = [k for k in ds.keys() if k not in OMISSIONS]
            print("Rendering " + str(len(datasets)) + " datasets: \n  " + 
                  "\n  ".join(datasets))
            for k in tqdm(datasets, position=0):
                # scale = ds[k].attrs["scale_factor"]
                arrays[k] = ds[k][:] #/ scale

    # Standardize the coordinate column names
    meta = guess_coords(meta)
    meta = decode_cols(meta)
    for dataset, array in arrays.items():
        if len(array.shape) > 1:
            meta[dataset] = array[layer]
        else:
            meta[dataset] = array

    # Create a geometry columns and a geodataframe
    meta["geometry"] = meta.apply(to_point, axis=1)
    crs = "epsg:{}".format(epsg)
    gdf = gpd.GeoDataFrame(meta, geometry="geometry", crs=crs)

    # We'd need to decode some columns, let's just get the datasets
    gdf = gdf[["geometry", *arrays.keys()]]
    # gdf = gdf[["geometry", "offshore", *arrays.keys()]]

    # Save file
    gdf.to_file(savepath, driver=driver_str)


@click.command()
@click.option("--file", "-f", help=FILE_HELP)
@click.option("--savepath", "-s", default=None, help=SAVE_HELP)
@click.option("--dataset", "-ds", default=None, help=DATASET_HELP)
@click.option("--layer", "-l", default=0, help=LAYER_HELP)
@click.option("--driver", "-d", default="GPKG", help=DRIVER_HELP)
def main(file, savepath, dataset, layer, driver):
    """ Take a csv or hdf5 output from reV and write a shapefile or geopackage.
    """

    # Set driver to lower case if gpkg
    if driver.lower() == "gpkg":
        driver = driver.lower()

    # Make sure the driver is available
    try:
        DRIVERS[driver]
    except KeyError:
        print("KeyError: Please provide or check the spelling of the driver " +
              "input...only 'shp' and 'gpkg' available at the moment.")

    # Two cases, hdf5 or csv - find better way to check for file format
    ext = os.path.splitext(file)[1]
    if ext == ".h5":
        h5_to_shape(file, driver=driver, dataset=dataset, layer=layer,
                    savepath=savepath)
    elif ext == ".csv":
        csv_to_shape(file=file, savepath=savepath, driver=driver)
    else:
        print("Sorry, rrshape can't handle that file type yet.")
        raise KeyError

if "__name__" == "__main__":
    main()
