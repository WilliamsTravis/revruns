# -*- coding: utf-8 -*-
"""
A CLI to georeference an HDF5 files for compatibility with GIS and GDAL tools.

Created on Wed Jun  3 09:48:49 2020

@author: twillia2
"""

import json
import os

import click
import h5py

from osgeo import osr

# Help printouts
FILE_HELP = ("The path to the target HDF5 file.")
PROFILE_HELP = ("A dictionary containing a `crs` and `geotransform` "
                "(or just `transform`) entry representing the cooridnate "
                "geometries of the target HDF5 datasets.")
CRS_HELP = ("The coordinate reference system of the target HDF5 file. Can be "
            "an EPSG code or proj4 string. By default, this will search the "
            "file's attributes for a profile dictionary with a `crs` entry.")
GEOTRANSFORM_HELP = ("The geotransformation values of the target HDF5 "
                     "file datasets in rasterio format. By default, this will "
                     "search the files' attributes for a dictionary with "
                     "`transform` or `geotransform` entries:\n    "
                     "(x-resolution, x-rotation, x-min, y-rotation, "
                     "y-resolution, y-min)")

@click.command()
@click.option("--file", "-f", required=True, help=FILE_HELP)
@click.option("--profile", "-p", default=None, help=PROFILE_HELP)
@click.option("--crs", "-c", default=None, help=CRS_HELP)
@click.option("--geotransform", "-g", default=None, help=GEOTRANSFORM_HELP)
def main(file, profile, crs, geotransform):
    """Georeference an HDF5 files for compatibility with GIS and GDAL tools.

    file = "/shared-projects/rev/projects/duke/data/exclusions/Duke_Exclusions.h5"
    profile = None
    crs = None
    geotransform = None
    """

    # Expand file path
    os.path.expanduser(file)

    # Check for existing georeferencing information if profile not given
    if not profile:
        if not crs or not geotransform:
            with h5py.File(file, "r") as h5:
                keys = h5.keys()
                akeys = h5.attrs.keys()
                if "profile" in akeys:
                    profile = h5.attrs["profile"]
                else:
                    for ds in keys:
                        if "profile" in h5[key].attrs.keys():
                            profile = h5[key].attrs["profile"]
                        else:
                            profile = None
        else:
            if not geotransform and crs:
                raise ValueError("Not enough information: need Geotransform.")
            if geotransform and not crs:
                raise ValueError("Not enough information: need CRS.")
            if not geotransform and not crs:
                raise ValueError("Not enough information: need CRS and "
                                 "geotransform or a profile dictionary "
                                 "containing both values.")
            profile = {"crs": crs, "transform": geotransform}


    # The profile might be a json
    if isinstance(profile, str):
        profile = json.loads(profile)

    # Get the geotransform the profile
    if "geotransform" in profile:
        transform = profile["geotransform"]
    elif "transform" in profile:
        transform = profile["transform"]
    else:
        raise KeyError("Geotransformation not found.")

    # Now reformat the crs
    crs = profile["crs"]
    spatial_ref = osr.SpatialReference()
    try:
        spatial_ref.ImportFromEPSG(crs)
    except TypeError:
        try: 
            code = spatial_ref.ImportFromProj4(crs)
            assert code == 0
        except AssertionError:
            try:
                code = spatial_ref.ImportFromWkt(crs)
                assert code == 0
            except AssertionError:
                print("CRS not properly formatted for proj4, epsg, or "
                      "wkt formats.")
                raise
    crs = spatial_ref.ExportToWkt()

    # Now set these attributes  # <-------------------------------------------- In the case of (cannot lock file, resource temporarily unavailable?)
    with h5py.File(file, "r+") as h5:
        h5.attrs["Projection"] = crs
        h5.attrs["GeoTransform"] = transform

if __name__ == "__main__":
    main()
