# -*- coding: utf-8 -*-
"""Make a quick shape file or geopackage out of the outputs from reV."""
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
             "directory with the basename of the csvh5 file. (str)")
LAYER_HELP = ("For HDF5 time series, the time index to render. Defaults to "
              " 0. (int)")
AG_HELP = ("For HDF5 time series, choose how to aggregate values. Use 'sum'"
           " for sum and 'mean' for mean. Defautls to '--layer' input.")
DATASET_HELP = ("For HDF5 files, the data set to render. Defaults to "
                "all available data sets. (str)")
DRIVER_HELP = ("Save as a Geopackage ('gpkg') or ESRI Shapefile ('shp'). "
               "Defaults to 'gpkg'. (str).")
CRS_HELP = ("A proj4 string or epsg code associated with the file's "
            "coordinate reference system. (str | int)")
OMISSIONS = ["coordinates", "time_index", "meta", "latitude", "longitude"]

# Different possible lat lon column names
COORD_NAMES = {"lat": ["latitude", "lat", "y", "ylat"],
               "lon": ["longitude", "lon", "long", "x", "xlong"]}
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
    in a pandas data frame.
    """
    columns = csv.columns
    columns = ["lat" if c in COORD_NAMES["lat"] else c for c in columns]
    columns = ["lon" if c in COORD_NAMES["lon"] else c for c in columns]
    csv.columns = columns

    return csv


def to_point(row):
    """Create a point object from a row with 'lat' and 'lon' columns."""
    point = Point((row["lon"], row["lat"]))
    return point


def csv_to_shape(src, dst=None, driver="gpkg", epsg=4326):
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

    src = "/shared-projects/rev/projects/weto/aggregation/05_b_b_mid/05_b_b_mid_sc.csv"
    driver = "gpkg"
    dst = "/shared-projects/rev/projects/weto/aggregation/05_b_b_mid/05_b_b_mid_sc.gpkg"
    """
    # Select driver
    driver_str = DRIVERS[driver]

    # If no save path
    if not dst:
        dst = src.replace(".csv", "." + driver)

    # Make sure the dst folder exists
    if os.path.dirname(dst) != "":
        os.makedirs(os.path.dirname(dst), exist_ok=True)

    # Use the save path to get the layer name
    layer = os.path.splitext(os.path.basename(dst))[0]

    # Read csv
    csv = pd.read_csv(src, low_memory=False)

    # Standardize the coordinate column names
    csv = guess_coords(csv)

    # Create a geometry columns and a geodataframe
    csv["geometry"] = csv.apply(to_point, axis=1)
    gdf = gpd.GeoDataFrame(csv, geometry="geometry")
    gdf.crs = "epsg:{}".format(epsg)

    # Save file
    gdf.to_file(dst, layer=layer, driver=driver_str)


def h5_to_shape(src, dst=None, driver="gpkg", dataset=None, layer=0, ag=None,
                epsg=4326):
    """For now, this will just take a single time period as an index position
    in the time series.

    Parameters
    ----------
    src : str
        reV HDF5 output file path.
    dst : str, optional
        Path to output shapefile. If not specified the output shapefile will be
        named using the input src path. The default is None.
    driver : str, optional
        The shapefile driver to use. The default is "GPKG".
    dataset : str
        The dataset to write, optional. The default is "cf_mean".
    layer : The , optional
        DESCRIPTION. The default is 0.
    ag : str
        How to aggregate time-series. 'mean' or 'sum'.
    epsg : TYPE, optional
        DESCRIPTION. The default is 4326.

    Sample Arguments
    ----------------

    src = "/shared-projects/rev/projects/heco/data/resource/era5/era5_2014.h5"
    dst = None
    driver = "gpkg"
    dataset = None
    layer = 1
    epsg = 4326
    """
    # Select driver
    driver_str = DRIVERS[driver]

    # If no save path
    if not dst:
        name = os.path.splitext(src)[0]
        dst = name + "." + driver

    # Make sure the dst folder exists
    if os.path.dirname(dst) != "":
        os.makedirs(os.path.dirname(dst), exist_ok=True)

    # Read hdf5 file
    arrays = {}
    with h5py.File(src, "r") as file:

        # This requires a meta object
        keys = list(file.keys())
        if "meta" in keys:
            meta = pd.DataFrame(file["meta"][:])
        else:
            raise KeyError("Meta data not found.")

        # Get our dataset if one is specified
        if dataset:
            ds = file[dataset]
            scale = ds.attrs["scale_factor"]

            if len(ds.shape) == 1:
                arrays[dataset] = ds[:] / scale
            else:
                arrays[dataset] = ds[layer] / scale

        # Get all available datasets if none is specified
        else:
            datasets = [k for k in keys if k not in OMISSIONS]
            print("Rendering " + str(len(datasets)) + " datasets: \n  " +
                  "\n  ".join(datasets))
            for k in tqdm(datasets, position=0):
                if k == "windspeed_100m":
                    break
                ds = file[k]
                scale = ds.attrs["scale_factor"]
                if len(ds.shape) == 1:
                    arrays[k] = ds[:] / scale
                else:
                    arrays[k] = ds[layer] / scale

    # Build our data frame from the meta file
    meta = guess_coords(meta)
    meta = decode_cols(meta)
    for dataset, array in arrays.items():
        meta[dataset] = array

    # Create a geometry columns and a geodataframe
    meta["geometry"] = meta.apply(to_point, axis=1)
    crs = "epsg:{}".format(epsg)
    gdf = gpd.GeoDataFrame(meta, geometry="geometry", crs=crs)

    # We'd need to decode some columns, let's just get the datasets
    gdf = gdf[["geometry", *arrays.keys()]]

    # Save file
    gdf.to_file(dst, driver=driver_str)


@click.command()
@click.argument("src")
@click.argument("dst", required=False, default=None)
@click.option("--dataset", "-ds", default=None, help=DATASET_HELP)
@click.option("--layer", "-l", default=0, help=LAYER_HELP)
@click.option("--agg", "-a", default=None, help=AG_HELP)
@click.option("--driver", "-d", default="GPKG", help=DRIVER_HELP)
def main(src, dst, dataset, layer, ag, driver):
    """Take a csv or hdf5 from reV and write a shapefile or geopackage."""
    # Expand this path in case we need to set dst
    src = os.path.expanduser(src)
    src = os.path.abspath(src)

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
    ext = os.path.splitext(src)[1]
    if ext == ".h5":
        h5_to_shape(src, dst, driver=driver, dataset=dataset, layer=layer,
                    ag=ag)
    elif ext == ".csv":
        csv_to_shape(src, dst, driver=driver)
    else:
        print("Sorry, rrshape can't handle that file type yet.")
        raise KeyError


if "__name__" == "__main__":
    main()
