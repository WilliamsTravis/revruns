# -*- coding: utf-8 -*-
"""Reformat a shapefile or raster into a reV raster using a template.

Ideally, any raster or vector format will work with this.

Created on Tue Apr 27 16:47:10 2021

@author: twillia2
"""
TEMPLATE = "/projects/rev/data/conus/wind_deployment_potential/albers.tif"
FILE = ("/projects/rev/data/conus/offshore/dist_to_coast_offshore.tif")
DST = ("/projects/rev/data/conus/dist_to_coast.tif")


class Reformatter:
    """Reformat any file into a reV-shaped raster."""
    import os
    import subprocess as sp

    import geopandas as gpd
    import rasterio as rio

    from rasterio import features

    def __init__(self, template):
        """Initialize Reformatter object.

        Parameters
        ----------
        template : str
            Path to a raster to use as a template.
        """
        self.template = template
        self.string_lookup = {}

    def __repr__(self):
        """Return object representation string."""
        name = self.__module__
        msg = f"<{name}: template={self.template}>"
        return msg

    def raster(self, file, dst):
        """Resample and reproject a raster."""
        self.sp.call(["rio", "warp", file, dst,
                      "--like", self.template,
                      "--co", "blockysize=128",
                      "--co", "blockxsize=128",
                      "--co", "tiled=yes"])

    def shapefile(self, file, dst, field=None, buffer=None):
        """Preprocess, reproject, and rasterize a vector."""
        # Read and process file
        gdf = self._process_shapefile(file, field, buffer)

        # Rasterize
        shapes = [(g, r) for r, g in gdf[["raster_value", "geometry"]].values]
        shape = [self.meta["height"], self.meta["width"]]
        transform = self.meta["transform"]
        with self.rio.Env():
            array = self.features.rasterize(shapes, shape, transform=transform)

        # Save
        with self.rio.Env():
            with self.rio.open(dst, "w", **self.meta) as file:
                file.write(array, 1)

        return dst

    @property
    def meta(self):
        """Return the meta information from the template file."""
        with self.rio.open(self.template) as raster:
            meta = raster.meta
        return meta

    def _process_shapefile(self, file, field=None, buffer=None):
        """Process a single file."""
        # Read in file
        gdf = self.gpd.read_file(file)

        # Assign raster value
        if field:
            gdf["raster_value"] = gdf[field]
        else:
            gdf["raster_value"] = 1

        # Account for string values
        if isinstance(gdf["raster_value"].iloc[0], str):
            gdf = self._map_strings(file, gdf, field)

        # Reduce to two fields
        gdf = gdf[["raster_value", "geometry"]]

        # Reproject before buffering
        gdf = gdf.to_crs(self.meta["crs"])
        if buffer:
            gdf["geometry"] = gdf["geometry"].buffer(buffer)

        return gdf

    def _map_strings(self, file, gdf, field):
        """Map string values to integers and save a lookup dictionary."""
        # Assing integers to unique string values
        strings = gdf["raster_value"].unique()
        string_map = {i + 1: v for i, v in enumerate(strings)}
        value_map = {v: k for k, v in string_map.items()}

        # Replace strings with integers
        gdf["raster_value"] = gdf["raster_value"].map(value_map)

        # Update the string lookup dictionary
        name = self.os.path.basename(file).split(".")[0]
        self.string_lookup[name] = string_map

        return gdf


if __name__ == "__main__":
    file = FILE
    dst = DST
    # self = Reformatter(TEMPLATE)
    # self.shapefile(FILE, DST)
    # self.raster(FILE, DST)
