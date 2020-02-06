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
import h5py
import os
import pandas as pd
from osgeo import gdal
from shapely.geometry import Point
import fiona

# Use GDAL to catch GDAL exceptions
gdal.UseExceptions()

# Help printouts
FILE_HELP = "The file from which to create the shape file. (str)"
SAVE_HELP = ("The path to use for the output file. Defaults to current " +
             "directory with the basename of the csv file. (str)")
LAYER_HELP = ("For hdf5 time series, the time layer to render. Defaults to " +
              " 0. (int)")
DATASET_HELP = ("For hdf5 time series, the data set to render. Defaults to " +
                "'cf_mean' (str)")
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
    driver_str = DRIVERS[driver]

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
    gdf = gpd.GeoDataFrame(csv, geometry="geometry")
    gdf.crs = fiona.crs.from_epsg(4326)

    # Save file
    if feather:
        # Save as with feather data types. Much quicker, but limited in use
        savepath = savepath.replace(driver, "feather")
        gfr.to_geofeather(gdf, savepath)
    else:
        # Save to shapefile or geopackage
        gdf.to_file(savepath, layer=layer, driver=driver_str)


def h5_to_shape(file, driver="gpkg", dataset="cf_mean", layer=0, savepath=None,
                feather=False):
    """
    For now, this will just take a single time period as an index position
    in the time series.

    Parameters
    ----------
    file : TYPE
        DESCRIPTION.
    driver : TYPE, optional
        DESCRIPTION. The default is "gpkg".
    savepath : TYPE, optional
        DESCRIPTION. The default is None.
    feather : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.


    SAMPLES:
    
        file = "/projects/rev/new_projects/lopez_wind/outputs/outputs_gen_2009.h5"
        savepath = "/projects/rev/new_projects/lopez_wind/outputs/outputs_gen_2009_1.gpkg"
        driver="gpkg"
        dataset="cf_mean"
        layer=0
    """
    # Select driver
    driver_str = DRIVERS[driver]

    # If no save path
    if not savepath:
        name = os.path.splitext(file)[0]
        savepath = name + "." + driver

    # Read hdf5 file
    with h5py.File(file) as ds:
        meta = pd.DataFrame(ds["meta"][:])
        array = ds[dataset][:]

    # Standardize the coordinate column names
    meta = guess_coords(meta)
    meta[dataset] = array


    # Create a geometry columns and a geodataframe
    meta["geometry"] = meta.apply(to_point, axis=1)
    gdf = gpd.GeoDataFrame(meta, geometry="geometry")
    gdf.crs = fiona.crs.from_epsg(4326)

    # We'd need to decode some columns, let's just get the one
    gdf = gdf[["geometry", dataset]]

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
@click.option("--dataset", "-ds", required=True, help=LAYER_HELP)
@click.option("--layer", "-l", default=0, help=LAYER_HELP)
@click.option("--driver", "-d", default="gpkg", help=DRIVER_HELP)
@click.option("--feather", is_flag=True, help=FEATHER_HELP)
def main(file, savepath, dataset, layer, driver, feather):
    """ Take a csv output from reV and write a shape file, geopackage, or
    geofeather file.


    sample args:
    file = "/projects/rev/new_projects/sergei_doubleday/final_outputs/5min_2018.h5"
    savepath = None
    dataset = "cf_mean"
    layer = 0
    driver = "gpkg"
    feather = False
    """
    # Make sure the driver is available
    try:
        DRIVERS[driver]
    except KeyError:
        print("KeyError: Please provide or check the spelling of the driver " +
              "input...only 'shp' and 'gpkg' available at the moment.")

    # Two cases, hdf5 or csv
    ext = os.path.splitext(file)[1]
    if ext == ".h5":
        h5_to_shape(file, driver=driver, dataset=dataset, layer=layer,
                    savepath=savepath, feather=feather)
    elif ext == ".csv":
        csv_to_shape(file=file, savepath=savepath, driver=driver,
                     feather=feather)
    else:
        print("Sorry, rrshape can't handle that file type yet.")
        raise KeyError

if "__name__" == "__main__":
    main()
