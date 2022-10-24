# -*- coding: utf-8 -*-
"""Create a raster out of an HDF point file."""
import click
import os
import shutil
import subprocess as sp

import fiona
import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import rasterio as rio

from revruns import rr

from scipy.spatial import cKDTree

os.environ['PROJ_NETWORK'] = 'OFF'


FILE_HELP = "The file from which to create the shape geotiff. (str)"
SAVE_HELP = ("The path to use for the output file. Defaults to current "
             "directory with the basename of the csv/h5 file. (str)")
VARIABLE_HELP = ("For HDF5 files, the data set to rasterize. For CSV files, the"
                " field to rasterize. (str)")
CRS_HELP = ("Coordinate reference system. Pass as <'authority':'code'>. (str)")
RES_HELP = ("Target resolution, in the same units as the CRS. (numeric)")
AGG_HELP = ("For HDF5 time series, the aggregation function to use to render "
            " the raster layer. Any appropriate numpy method. If 'layer' is "
            "provided, this will be ignored."
            " defaults to mean. (str)")
LAYER_HELP = ("For HDF5 time series, the time index to render. If attempting "
              "to rasterize a time series and this isn't provided, an "
              "aggregation function will be used. Defaults to 0. (int)")
FILTER_HELP = ("A column name, value pair to use to filter the data before "
               "rasterizing (e.g. rrraster -f state -f Georgia ...). (list)")
FILL_HELP = ("Fill na values by interpolation. (boolen)")
CUT_HELP = ("Path to vector file to use to clip output. (str)")


def csv(src, dst, variable, res, crs, fillna, cutline):
    # This is inefficient
    df = pd.read_csv(src)
    gdf = df.rr.to_geo()
    gdf = gdf[["geometry", variable]]

    # We need to reproject to the specified projection system
    gdf = gdf.to_crs(crs)

    # And finally rasterize
    rasterize(gdf, res, dst, fillna, cutline)


def get_scale(ds, variable):
    attrs = ds[variable].attrs.keys()
    scale_key = [k for k in attrs if "scale_factor" in k][0]
    if scale_key:
        scale = ds[variable].attrs[scale_key]
    else:
        scale = 1
    return scale


def gpkg(src, dst, variable, resolution, crs, fillna, cutline):
    # This is inefficient
    gdf = gpd.read_file(src)
    gdf = gdf[["geometry", variable]]

    # We need to reproject to the specified projection system
    gdf = gdf.to_crs(crs)

    # Convert to true grid

    # And finally rasterize
    rasterize(gdf, resolution, dst, fillna, cutline)


def h5(src, dst, variable, resolution, crs, agg_fun, layer, fltr, fillna,
       cutline):
    """
    Rasterize dataset in HDF5 file.

    Parameters
    ----------
    src : TYPE
        DESCRIPTION.
    dst : TYPE
        DESCRIPTION.
    variable : TYPE
        DESCRIPTION.
    resolution : TYPE
        DESCRIPTION.
    crs : TYPE
        DESCRIPTION.
    agg_fun : TYPE
        DESCRIPTION.
    layer : TYPE
        DESCRIPTION.
    fltr : TYPE
        DESCRIPTION.
    fillna : TYPE
        DESCRIPTION.
    cutline : TYPE
        DESCRIPTION.

    Returns
    -------
    """
    # Open the file
    with h5py.File(src, "r") as ds:
        meta = pd.DataFrame(ds["meta"][:])
        meta.rr.decode()

        # Is this a time series or single layer
        scale = get_scale(ds, variable)
        if len(ds[variable].shape) > 1:
            data = h5_timeseries(ds, variable, agg_fun, layer)
            field = "{}_{}".format(variable, agg_fun)
        else:
            data = ds[variable][:]
            field = variable

    # Append the data to the meta object and create geodataframe
    meta[field] = data / scale

    # Do we wnt to apply a filter? How to parameterize that?  # <-------------- Not ready
    if fltr:
        meta = meta[meta[fltr[0]] == fltr[1]]

    # Create a GeoDataFrame
    gdf = meta.rr.to_geo()
    gdf = gdf[["geometry", field]]

    # We need to reproject to the specified projection system
    gdf = gdf.to_crs(crs)

    # Create a consistent grid out of this
    gdf = to_grid(gdf, variable, resolution)

    # And finally rasterize
    rasterize(gdf, resolution, dst, fillna, cutline)

    return dst


def h5_timeseries(ds, dataset, agg_fun, layer):
    # Specifying a layer overrides the aggregation
    if layer:
        data = ds[dataset][layer]
    else:
        fun = getattr(np, agg_fun)
        data = fun(ds[dataset][:], axis=0)
    return data


def rasterize(gdf, resolution, dst, fillna=False, cutline=None, variable=None):
    # Make sure we have the target directory
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)

    # Not doing this in memory in case it's big
    tmp_src = dst.replace(".tif", "_tmp.gpkg")
    gdf.to_file(tmp_src, driver="GPKG")
    layer_name = fiona.listlayers(tmp_src)[0]

    # There will only be two columns
    if not variable:
        variable = gdf.columns[1]

    # Set no data value
    dtype = str(gdf[variable].dtype)
    if "float" in dtype:
        na = np.finfo(dtype).max
    else:
        na = np.iinfo(dtype).max

    # Get extent
    te = [str(b) for b in gdf.total_bounds]

    # Write to dst
    sp.call(
        [
            "gdal_rasterize",
            tmp_src, dst,
            "-l", layer_name,
            "-a", variable,
            "-a_nodata", str(na),
            "-at",
            "-te", *te,
            "-tr", str(resolution), str(-resolution)
        ]
    )

    # Fill na values
    if fillna:
        tmp_dst = dst.replace(".tif", "_tmp.tif")
        sp.call(["gdal_fillnodata.py", dst, tmp_dst])
        os.remove(dst)
        shutil.move(tmp_dst, dst)

    # Cut to vector
    if cutline:
        tmp_dst = dst.replace(".tif", "_tmp.tif")
        sp.call(["gdalwarp", dst, tmp_dst, "-cutline", cutline])
        os.remove(dst)
        shutil.move(tmp_dst, dst)

    # Get rid of temporary shapefile
    os.remove(tmp_src)


def to_grid(gdf, variable, resolution):
    """Convert coordinates from an irregular point dataset into an even grid.

    Parameters
    ----------
    gdf: geopandas.geodataframe.GeoDataFrame
        A geopandas data frame
    resolution: int | float
        The resolution of the target grid.
    Returns
    -------
    numpy.ndarray, numpy.ndarray
        Returns a 3D array (y, x, time) of data values a 2D array of coordinate
        values (nxy, 2).
    Notes
    -----
    - This only takes about a minute for a ~500 X 500 X 8760 dim dataset, but
    it eats up memory. If we saved the df to file, opened it as a dask data
    frame, and generated the arrays as dask arrays we might be able to save
    space.
    - At the moment it is a little awkardly shaped, just because I haven't
    gotten to it yet.
    """
    # Only one variable at a time here
    gdf = gdf[[variable, "geometry"]]

    # At the end of all this the actual data will be inbetween these columns
    non_values = ["geometry", "gx", "gy", "ix", "iy"]

    # Get the extent
    minx, miny, maxx, maxy = gdf.total_bounds

    # Estimate target grid coordinates
    gridx = np.arange(minx, maxx + resolution, resolution)
    gridy = np.arange(miny, maxy + resolution, resolution)
    grid_points = np.array(np.meshgrid(gridy, gridx)).T.reshape(-1, 2)

    # Go ahead and make the geotransform
    geotransform = [resolution, 0, minx, 0, resolution, miny]

    # Get source point coordinates
    gdf["y"] = gdf["geometry"].apply(lambda p: p.y)
    gdf["x"] = gdf["geometry"].apply(lambda p: p.x)
    points = gdf[["y", "x"]].values

    # Build kdtree
    ktree = cKDTree(grid_points)
    dist, indices = ktree.query(points)

    # Those indices associate grid point coordinates with the original points
    gdf["gy"] = grid_points[indices, 0]
    gdf["gx"] = grid_points[indices, 1]

    # And these indices indicate the 2D cartesion positions of the grid
    gdf["iy"] = gdf["gy"].apply(lambda y: np.where(gridy == y)[0][0])
    gdf["ix"] = gdf["gx"].apply(lambda x: np.where(gridx == x)[0][0])

    # Now we want just the values from the data frame, no coordinates
    values = gdf[variable].T.values

    # Okay, now use this to create our 2D empty target grid
    grid = np.zeros((gridy.shape[0], gridx.shape[0]))

    # Now, use the cartesian indices to add the values to the new grid
    grid[gdf["iy"].values, gdf["ix"].values] = values  # <--------------------- Check these values against the original dataset

    # Holy cow, did that work?
    return grid, geotransform


@click.command()
@click.argument("src")
@click.argument("dst")
@click.option("--variable", "-v", required=True, help=VARIABLE_HELP)
@click.option("--resolution", "-r", required=True, help=RES_HELP)
@click.option("--crs", "-c", default="epsg:4326", help=CRS_HELP)
@click.option("--agg_fun", "-a", default="mean", help=AGG_HELP)
@click.option("--layer", "-l", default=None, help=LAYER_HELP)
@click.option("--fltr", "-f", default=None, multiple=True, help=FILTER_HELP)
@click.option("--fillna", "-fn", is_flag=True, help=FILL_HELP)
@click.option("--cutline", "-cl", default=None, help=CUT_HELP)
def main(src, dst, variable, resolution, crs, agg_fun, layer, fltr, fillna,
         cutline):
    """REVRUNS - RRASTER - Rasterize a reV output.

    src = "/home/twillia2/transfer/review_01_na_n_mid_sc_vs_01_na_n_mid_old_sc_diff_capacity.csv"
    dst = "/home/twillia2/transfer/review_01_na_n_mid_sc_vs_01_na_n_mid_old_sc_diff_capacity.tif"
    variable = "capacity_difference_percent"
    resolution = 11_520
    crs = "epsg:5070"
    agg_fun = "mean"
    layer = None
    fltr = None
    cutline = None
    fillna = False
    """
    # Get the file extension and call the appropriate function
    extension = os.path.splitext(src)[-1]
    if extension == ".h5":
        h5(src, dst, variable, resolution, crs, agg_fun, layer, fltr, fillna,
           cutline)
    elif extension == ".csv":
        csv(src, dst, variable, resolution, crs, fillna, cutline)
    elif extension == ".gpkg":
        gpkg(src, dst, variable, resolution, crs, fillna, cutline)


# if __name__ == "__main__":
    # main()
