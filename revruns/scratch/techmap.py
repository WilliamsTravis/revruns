"""Explore alternate techmap strategy.

TechMapping has been a struggle lately. It might be possible to
recreate this grid with standard gdal/rasterio methods.

If we have two grids in different extents, crss, and resolutions
we should still be able to warp one to the others geometry to associate
grid cells. We'll need to resample the coarser grid to the finer grid.
"""
import json

import h5py
import rasterio as rio

from pyproj import CRS, Transformer


def main():
    """Associate fine grid ids with coarse grid ids."""
    # Read in fine and coarse grids
    fine = h5py.File("/projects/rev/data/exclusions/Offshore_Exclusions.h5", "r")
    coarse = h5py.File("/datasets/WIND/conus/v1.0.0/wtk_conus_2007.h5", "r")

    # Find georeferencing information for both

    # Extract/build the coarser grid

    # Coarser grid to grid id as values

    # Coarser grid to finer crs

    # Coarser grid to finer extent

    # Coarser grid to finer resolution

    # Upload new grid to finer hdf file as "techmap_{resource}"
