# -*- coding: utf-8 -*-
"""Create a raster out of an HDF point file.
"""

import os
import subprocess as sp

import click
import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import rasterio as rio
import revruns as rr


FILE_HELP = "The file from which to create the shape geotiff. (str)"
SAVE_HELP = ("The path to use for the output file. Defaults to current "
             "directory with the basename of the csv/h5 file. (str)")
DATASET_HELP = ("For HDF5 files, the data set to rasterize. For CSV files, the "
                "field to rasterize. (str)")
CRS_HELP = ("Coordinate reference system. Pass as <'authority':'code'>. (str)")
RES_HELP = ("Target resolution, in the same units as the CRS. (numeric)")
AGG_HELP = ("For HDF5 time series, the aggregation function to use to render "
            " the raster layer. Any appropriate numpy method. If 'layer' is "
            "provided, this will be ignored."
            " defaults to mean. (str)")
MASK_HELP = ("Path to a binary raster of equal dimension to the output raster "
             "used to mask out cells of the output raster. (str)")
LAYER_HELP = ("For HDF5 time series, the time index to render. If attempting "
              "to rasterize a time series and this isn't provided, an "
              "aggregation function will be used. Defaults to 0. (int)")
FILTER_HELP = ("A column name, value pair to use to filter the data before "
               "rasterizing (e.g. rrraster -f state -f Georgia ...). (list)")
FILL_HELP = ("Fill na values by interpolation. (boolen)")

def get_scale(ds, dataset):
    attrs = ds[dataset].attrs.keys()
    scale_key = [k for k in attrs if "scale_factor" in k][0]
    if scale_key:
        scale = ds[dataset].attrs[scale_key]
    else:
        scale = 1
    return scale


def h5(file, dst, dataset, res, crs, agg_fun, layer, fltr, mask, fillna):

    # Open the file
    with h5py.File(file, "r") as ds:
        meta = pd.DataFrame(ds["meta"][:])
        meta.rr.decode()

        # Is this a time series or single layer
        scale = get_scale(ds, dataset)
        if len(ds[dataset].shape) > 1:
            data = h5_timeseries(ds, dataset, agg_fun, layer)
            field = "{}_{}".format(dataset, agg_fun)
        else:
            data = ds[dataset][:]
            field = dataset

    # Append the data to the meta object and create geodataframe
    meta[field] = data / scale

    # Do we wnt to apply a filter? How to parameterize that?
    if fltr:
        meta = meta[meta[fltr[0]] == fltr[1]]

    # Create a GeoDataFrame
    gdf = meta.rr.to_geo()
    gdf = gdf[["geometry", field]]

    # We need to reproject to the specified projection system
    gdf = gdf.to_crs(crs)

    # And finally rasterize
    rasterize(gdf, res, dst, mask, fillna)


def gpkg(file, dst, dataset, res, crs, mask, fillna):

    # This is inefficient
    gdf = gpd.read_file(file)
    gdf = gdf[["geometry", dataset]]    

    # We need to reproject to the specified projection system
    gdf = gdf.to_crs(crs)

    # And finally rasterize
    rasterize(gdf, res, dst, mask, fillna)


def csv(file, dst, dataset, res, crs, mask, fillna):
    
    # This is inefficient
    df = pd.read_csv(file)
    gdf = df.rr.to_geo()
    gdf = gdf[["geometry", dataset]]    

    # We need to reproject to the specified projection system
    gdf = gdf.to_crs(crs)

    # And finally rasterize
    rasterize(gdf, res, dst, mask, fillna)
    

def h5_timeseries(ds, dataset, agg_fun, layer):

    import numpy as np    

    # Specifying a layer overrides the aggregation
    if layer:
        data = ds[dataset][layer]
    else:
        fun = getattr(np, agg_fun)
        data = fun(ds[dataset][:], axis=0)
    return data


def rasterize(gdf, res, dst, mask, fillna):

    # Make sure we have the raget directory
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)

    # Not doing this in memory in case it's big
    tmp_src = dst.replace(".tif" ,".gpkg")
    layer_name = os.path.basename(dst).replace(".tif", "")
    gdf.to_file(tmp_src, driver="GPKG")

    # There will only be two columns
    attribute = gdf.columns[1]

    # Write to dst
    sp.call(["gdal_rasterize",
             tmp_src, dst,
             "-a_nodata", "-9999",
             "-l", layer_name,
             "-a", attribute,
             "-at",
             "-tr", str(res), str(res)])

    # Fill na values
    if fillna:
        sp.call(["gdal_fillnodata.py", dst])

    # If mask is provided
    if mask:
        with rio.open(dst) as raster:
            with rio.open(mask) as rmask:
                r = raster.read(1)
                m = rmask.read(1)
                final = r * m
                profile = raster.profile
        profile["nodata"] = 0
        with rio.Env():
            with rio.open(dst, "w", **profile) as file:
                file.write(final, 1)

    # Get rid of temporary shapefile
    os.remove(tmp_src)


@click.command()
@click.argument("src")
@click.argument("dst")
@click.option("--dataset", "-d", required=True, help=DATASET_HELP)
@click.option("--resolution", "-r", required=True, help=RES_HELP)
@click.option("--crs", "-c", default="epsg:4326", help=CRS_HELP)
@click.option("--agg_fun", "-a", default="mean", help=AGG_HELP)
@click.option("--layer", "-l", default=None, help=LAYER_HELP)
@click.option("--filter", "-f",default=None, multiple=True, help=FILTER_HELP)
@click.option("--mask", "-m", default=None, help=MASK_HELP)
@click.option("--fillna", "-fn", is_flag=True, help=FILL_HELP)
def main(src, dst, dataset, resolution, crs, agg_fun, layer, filter, mask, fillna):
    """
    file = "20ps_sc.csv"
    dst = "120hh_total_lcoe.tif"
    dataset = "total_lcoe"
    res = 5760
    crs = "EPSG:3466"
    agg_fun = "mean"
    layer = None
    filter = None
    mask = "/shared-projects/rev/projects/soco/data/rasters/se_mask.tif"
    """

    extension = os.path.splitext(src)[1]        

    if extension == ".h5":
        h5(src, dst, dataset, resolution, crs, agg_fun, layer, filter, mask, fillna)

    elif extension == ".csv":
        csv(src, dst, dataset, resolution, crs, mask, fillna)

    elif extension == ".gpkg":
        gpkg(src, dst, dataset, resolution, crs, mask, fillna)


if __name__ == "__main__":    
    main()
