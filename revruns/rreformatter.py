# -*- coding: utf-8 -*-
"""Reformat a shapefile or raster into a reV raster using a template.

Ideally, any raster or vector format will work with this.

Created on Tue Apr 27 16:47:10 2021

@author: twillia2
"""
TEMPLATE = "/projects/rev/data/conus/wind_deployment_potential/albers.tif"
FILE = ("/projects/rev/data/conus/offshore/dist_to_coast_offshore.tif")
DST = ("/projects/rev/data/conus/dist_to_coast.tif")


class Exclusions:
    """Build or add to an HDF5 Exclusions dataset."""

    import h5py
    import numpy as np
    import rasterio as rio

    from pyproj import Transformer
    from rasterio.errors import RasterioIOError

    def __init__(self, excl_fpath, string_values={}):
        """Initialize Exclusions object.

        Parameters
        ----------
            excl_fpath : str
                Path to target HDF5 reV exclusion file.
            string_values : str | dict
                Dictionary or path dictionary of raster value, key pairs
                derived from shapefiles containing string values (optional).
        """
        super().__init__(self, **kwargs)
        self.excl_fpath = excl_fpath
        self.string_values = string_values
        self._preflight()
        self._initialize_h5()

    def __repr__(self):
        """Print the object representation string."""
        msg = "<Exclusions Object:  excl_fpath={}>".format(self.excl_fpath)
        return msg

    def add_layer(self, dname, file, description=None, overwrite=False):
        """Add a raster file and its description to the HDF5 exclusion file."""
        # Open raster object
        try:
            raster = self.rio.open(file)
        except Exception:
            raise self.RasterioIOError("file " + file + " does not exist")

        # Get profile information
        profile = raster.profile
        profile["crs"] = profile["crs"].to_proj4()
        dtype = profile["dtype"]
        profile = dict(profile)

        # We need a 6 element geotransform, sometimes we recieve three extra
        profile["transform"] = profile["transform"][:6]

        # Add coordinates and else check that the new file matches everything
        self._set_coords(profile)
        self._check_dims(raster, profile, dname)

        # Add everything to target exclusion HDF
        array = raster.read()
        with self.h5py.File(self.excl_fpath, "r+") as hdf:
            keys = list(hdf.keys())
            if dname in keys:
                if overwrite:
                    del hdf[dname]
                    keys.remove(dname)

            if dname not in keys:
                hdf.create_dataset(name=dname, data=array, dtype=dtype,
                                   chunks=(1, 128, 128))
                hdf[dname].attrs["file"] = os.path.abspath(file)
                hdf[dname].attrs["profile"] = json.dumps(profile)
                if description:
                    hdf[dname].attrs["description"] = description
                if dname in self.string_values:
                    string_value = json.dumps(self.string_values[dname])
                    hdf[dname].attrs["string_values"] = string_value

    def add_layers(self, file_dict, desc_dict=None, overwrite=False):
        """Add multiple raster files and their descriptions."""
        from tqdm import tqdm

        # Make copies of these dictionaries?
        file_dict = file_dict.copy()
        if desc_dict:
            desc_dict = desc_dict.copy()

        # If descriptions are provided make sure they match the files
        if desc_dict:
            try:
                dninf = [k for k in desc_dict if k not in file_dict]
                fnind = [k for k in file_dict if k not in desc_dict]
                assert not dninf
                assert not fnind
            except Exception:
                mismatches = self.np.unique(dninf + fnind)
                msg = ("File and description keys do not match. "
                       "Problematic keys: " + ", ".join(mismatches))
                raise AssertionError(msg)
        else:
            desc_dict = {key: None for key in file_dict.keys()}

        # Let's remove existing keys here
        if not overwrite:
            with self.h5py.File(self.excl_fpath, "r") as h5:
                keys = list(h5.keys())
                for key in keys:
                    if key in file_dict:
                        del file_dict[key]
                        del desc_dict[key]

        # Should we parallelize this?
        for dname, file in tqdm(file_dict.items(), total=len(file_dict)):
            description = desc_dict[dname]
            self.add_layer(dname, file, description, overwrite=overwrite)

    def techmap(self, res_fpath, dname, max_workers=None, map_chunk=2560,
                distance_upper_bound=None, save_flag=True):
        """Build a mapping grid between exclusion resource data.

        Parameters
        ----------
        res_fpath : str
            Filepath to HDF5 resource file.
        dname : str
            Dataset name in excl_fpath to save mapping results to.
        max_workers : int, optional
            Number of cores to run mapping on. None uses all available cpus.
            The default is None.
        distance_upper_bound : float, optional
            Upper boundary distance for KNN lookup between exclusion points and
            resource points. None will calculate a good distance based on the
            resource meta data coordinates. 0.03 is a good value for a 4km
            resource grid and finer. The default is None.
        map_chunk : TYPE, optional
          Calculation chunk used for the tech mapping calc. The default is
            2560.
        save_flag : boolean, optional
            Save the techmap in the excl_fpath. The default is True.
        """
        from reV.supply_curve.tech_mapping import TechMapping

        # If saving, does it return an object?
        arrays = TechMapping.run(self.excl_fpath, res_fpath, dname,
                                 max_workers=None, sc_resolution=2560)
        return arrays

    def _preflight(self):
        """More initializing steps."""
        if self.string_values:
            if not isinstance(self.string_values, dict):
                with open(self.string_values, "r") as file:
                    self.string_values = json.load(file)

    def _check_dims(self, raster, profile, dname):
        # Check new layers against the first added raster
        old = self.profile
        new = profile

        # Check the CRS
        if not crs_match(old["crs"], new["crs"]):
            raise AssertionError("CRS for " + dname + " does not match"
                                 " exisitng CRS.")

        # Check the transform
        try:
            # Standardize these
            old_trans = old["transform"][:6]
            new_trans = new["transform"][:6]
            assert old_trans == new_trans
        except Exception:
            raise AssertionError("Geotransform for " + dname + " does "
                                 "not match geotransform.")

        # Check the dimesions
        try:
            assert old["width"] == new["width"]
            assert old["height"] == new["height"]
        except Exception:
            raise AssertionError("Width and/or height for " + dname +
                                 " does not match exisitng " +
                                 "dimensions.")

    def _convert_coords(self, xs, ys):
        # Convert projected coordinates into WGS84
        mx, my = self.np.meshgrid(xs, ys)
        transformer = self.Transformer.from_crs(self.profile["crs"],
                                                "epsg:4326", always_xy=True)
        lons, lats = transformer.transform(mx, my)
        return lons, lats

    def _get_coords(self):
        # Get x and y coordinates (One day we'll have one transform order!)
        geom = self.profile["transform"]  # Ensure its in the right order
        xres = geom[0]
        ulx = geom[2]
        yres = geom[4]
        uly = geom[5]

        # Not doing rotations here
        xs = [ulx + col * xres for col in range(self.profile["width"])]
        ys = [uly + row * yres for row in range(self.profile["height"])]

        # Let's not use float 64
        xs = self.np.array(xs).astype("float32")
        ys = self.np.array(ys).astype("float32")

        return xs, ys

    def _initialize_h5(self):
        import datetime as dt

        # Create an empty hdf file if one doesn't exist
        date = format(dt.datetime.today(), "%Y-%m-%d %H:%M")
        self.excl_fpath = os.path.expanduser(self.excl_fpath)
        self.excl_fpath = os.path.abspath(self.excl_fpath)
        if not os.path.exists(self.excl_fpath):
            os.makedirs(os.path.dirname(self.excl_fpath), exist_ok=True)
            with self.h5py.File(self.excl_fpath, "w") as ds:
                ds.attrs["creation_date"] = date

    def _set_coords(self, profile):
        # Add the lat and lon meshgrids if they aren't already present
        with self.h5py.File(self.excl_fpath, "r+") as hdf:
            keys = list(hdf.keys())
            attrs = hdf.attrs.keys()

            # Set profile if needed
            if "profile" not in attrs:
                hdf.attrs["profile"] = json.dumps(profile)
            self.profile = profile

            # Set coordinates if needed
            if "latitude" not in keys or "longitude" not in keys:
                # Get the original crs coordinates
                xs, ys = self._get_coords()

                # Convert to geographic coordinates
                lons, lats = self._convert_coords(xs, ys)

                # Create grid and upload
                hdf.create_dataset(name="longitude", data=lons)
                hdf.create_dataset(name="latitude", data=lats)

class Reformatter(Exclusions):
    """Reformat any file into a reV-shaped raster."""
    import os
    import subprocess as sp

    import numpy as np
    import geopandas as gpd
    import rasterio as rio

    from rasterio import features

    def __init__(self, template, inputs):
        """Initialize Reformatter object.

        Parameters
        ----------
        template : str
            Path to a raster to use as a template.
        inputs: dict
            Dictionary containing with 
        """
        self.template = template
        self.shapefile_fields = shapefile_fields
        self.string_lookup = {}

    def __repr__(self):
        """Return object representation string."""
        return f"<Reformatter: template={self.template}>"

    def raster(self, file, dst):  # <------------------------------------------ Use rio python bindings here, the compression isnt' working
        """Resample and reproject a raster."""
        self.sp.call(["rio", "warp", file, dst,
                      "--like", self.template,
                      "--co", "compress=lzw",  # <----------------------------- Might not be working?
                      "--co", "blockysize=128",
                      "--co", "blockxsize=128",
                      "--co", "tiled=yes"])

    def shapefile(self, file, dst, field=None, buffer=None, overwrite=False):
        """Preprocess, reproject, and rasterize a vector."""
        # Read and process file
        gdf = self._process_shapefile(file, field, buffer)
        meta = self.meta

        # Skip if overwrite
        if not overwrite and self.os.path.exists(dst):
            return
        else:
            # Rasterize
            field = self.shapefile_fields[file]
            elements = gdf[[field, "geometry"]].values
            if isinstance(gdf[field].iloc[0], str):
                elements = self._map_strings(file, gdf, field)

            shapes = [(g, r) for r, g in elements]
            shape = [meta["height"], meta["width"]]
            transform = meta["transform"]
            with self.rio.Env():
                array = self.features.rasterize(shapes, shape,
                                                transform=transform)

            # Update meta
            dtype = str(array.dtype)
            if "int" in dtype:
                nodata = self.np.iinfo(dtype).max
            else:
                nodata = self.np.finfo(dtype).max
            meta["dtype"] = dtype
            meta["nodata"] = nodata

            # Write
            with self.rio.Env():
                with self.rio.open(dst, "w", **meta) as file:
                    file.write(array, 1)

    def guided(self, dirname="."):
        """Guide the processing of shapefiles with prompts and user inputs."""

    @property
    def meta(self):
        """Return the meta information from the template file."""
        with self.rio.open(self.template) as raster:
            meta = raster.meta
        return meta

    def _map_strings(self, file, gdf, field):
        """Map string values to integers and save a lookup dictionary."""
        # Assing integers to unique string values
        strings = gdf[field].unique()
        string_map = {i + 1: v for i, v in enumerate(strings)}
        value_map = {v: k for k, v in string_map.items()}

        # Replace strings with integers
        gdf[field] = gdf[field].map(value_map)

        # Update the string lookup dictionary
        name = self.os.path.basename(file).split(".")[0]  # <------------------ Adjust incase people use .'s in there file names
        self.string_lookup[name] = string_map

        return gdf

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


if __name__ == "__main__":
    shapefile_fields = {
        "2019-2024dppexclusionoptionareas_atlantic.gpkg": "TEXT_LABEL",
        "2019-2024dppexclusionoptionareas_gomr.gpkg": "TEXT_LABEL",
        "conservation_areas.gpkg": "Design"
    }

    file = FILE
    dst = DST
    self = Reformatter(TEMPLATE, shapefile_fields=shapefile_fields)
    # self.shapefile(FILE, DST)
    # self.raster(FILE, DST)
    # smap = self.string_lookup