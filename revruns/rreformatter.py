# -*- coding: utf-8 -*-
"""Reformat a shapefile or raster into a reV raster using a template.

Ideally, any raster or vector format will work with this.

Created on Tue Apr 27 16:47:10 2021

Updated in Feb, 2,2022 

@author: twillia2
"""
import datetime as dt
import json
import subprocess as sp
import os
import warnings

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import rasterio as rio

from rasterio import features
from pyproj import Transformer
from revruns.rr import crs_match, isint, isfloat
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)


NEEDED_FIELDS = ["name", "path"]


class Exclusions:
    """Build or add to an HDF5 Exclusions dataset."""

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
        raster = rio.open(file)

        # Get profile information
        profile = raster.profile
        profile["crs"] = profile["crs"].to_proj4()
        dtype = profile["dtype"]
        profile = dict(profile)

        # We need a 6 element geotransform, sometimes we receive three extra <- why?
        profile["transform"] = profile["transform"][:6]

        # Add coordinates and else check that the new file matches everything
        self._set_coords(profile)
        self._check_dims(file, self.profile)

        # Add everything to target exclusion HDF
        array = raster.read()
        with h5py.File(self.excl_fpath, "r+") as hdf:
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
                mismatches = np.unique(dninf + fnind)
                msg = ("File and description keys do not match. "
                       "Problematic keys: " + ", ".join(mismatches))
                raise AssertionError(msg)
        else:
            desc_dict = {key: None for key in file_dict.keys()}

        # Let's remove existing keys here
        if not overwrite:
            with h5py.File(self.excl_fpath, "r") as h5:
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

    def _check_dims(self, file, profile):
        # Check new layers against the first added raster
        old = profile
        dname = os.path.basename(file)
        with rio.open(file, "r") as r:    
            new = r.profile

        # Check the CRS
        if not crs_match(old["crs"], new["crs"]):
            raise AssertionError(f"CRS for {dname} does not match exisitng "
                                 "CRS.")

        # Check the transform
        try:
            # Standardize these
            old_trans = old["transform"][:6]
            new_trans = new["transform"][:6]
            assert old_trans == new_trans
        except AssertionError:
            print(f"Geotransform for {dname} does not match geotransform.")
            raise

        # Check the dimesions
        try:
            assert old["width"] == new["width"]
            assert old["height"] == new["height"]
        except AssertionError:
            print(f"Width and/or height for {dname} does not match existing "
                  "dimensions.")
            raise

    def _convert_coords(self, xs, ys):
        # Convert projected coordinates into WGS84
        mx, my = np.meshgrid(xs, ys)
        transformer = Transformer.from_crs(self.profile["crs"],
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
        xs = np.array(xs).astype("float32")
        ys = np.array(ys).astype("float32")

        return xs, ys

    def _initialize_h5(self):
        # Create an empty hdf file if one doesn't exist
        date = format(dt.datetime.today(), "%Y-%m-%d %H:%M")
        self.excl_fpath = os.path.expanduser(self.excl_fpath)
        self.excl_fpath = os.path.abspath(self.excl_fpath)
        if not os.path.exists(self.excl_fpath):
            os.makedirs(os.path.dirname(self.excl_fpath), exist_ok=True)
            with h5py.File(self.excl_fpath, "w") as ds:
                ds.attrs["creation_date"] = date

    def _set_coords(self, profile):
        # Add the lat and lon meshgrids if they aren't already present
        with h5py.File(self.excl_fpath, "r+") as hdf:
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
    """Reformat any file or set of files into a reV-shaped raster."""

    def __init__(self, inputs, out_dir, template=None, excl_fpath=None, 
                 overwrite_tif=False, overwrite_dset=False):
        """Initialize Reformatter object.

        Parameters
        ----------
        inputs : str | dict | pd.core.frame.DataFrame
            Input data information. If dictionary, top-level keys provide
            the target name of reformatted dataset, second-level keys are
            'path' (required path to file'), 'field' (optional field name for
            vectors), 'buffer' (optional buffer distance), and 'layer' 
            (required only for FileGeoDatabases). Inputs can also include any
            additional field and it will be included in the attributes of the
            HDF5 dataset if writing to and HDF5 file. If pandas data frame, the
            secondary keys are required as columns, and the top-level key
            is stored in a 'name' column. A path to a CSV of this table is
            also acceptable.
        out_dir : str
            Path to a directory where reformatted data will be written as
            GeoTiffs.
        template : str
            Path to a raster with target georeferencing attributes. If 'None'
            the 'excl_fpath' must point to an HDF5 file with target
            georeferencing information as a top-level attribute.
        excl_fpath : str
            Path to existing or target HDF5 exclusion file. If provided,
            reformatted datasets will be added to this file.
        overwrite_tif : boolean
            If `True`, this will overwrite rasters in `out_dir`.
        overwrite_dset : boolean
            If `True`, this will overwrite datasets in `excl_fpath`.
        """
        super().__init__(str(excl_fpath))
        os.makedirs(out_dir, exist_ok=True)
        if excl_fpath is not None:
            excl_fpath = os.path.abspath(os.path.expanduser(excl_fpath))
        self._parse_inputs(inputs)
        if template is not None:
            self.template = os.path.abspath(os.path.expanduser(template))
        else:
            self.template = template
        self.out_dir = os.path.abspath(os.path.expanduser(out_dir))
        self.excl_fpath = excl_fpath
        self.overwrite_tif = overwrite_tif
        self.overwrite_dset = overwrite_dset
        self.lookup = {}

    def __repr__(self):
        """Return object representation string."""
        attrs = [f"{k}={v}" for k, v in self.__dict__.items() if k != "inputs"]
        pattrs = ", ".join(attrs)
        return f"<Reformatter: {pattrs}>"

    @property
    def meta(self):
        """Return the meta information from the template file."""
        meta = None
        if self.template is not None:
            with rio.open(self.template) as raster:
                meta = raster.meta
        else:
            if not os.path.exists(self.excl_fpath):
                raise FileNotFoundError("If not template is provided, an "
                                        "existing `excl_fpath` is required.")
            with h5py.File(self.excl_fpath, "r") as ds:
                if "profile" in ds.attrs:
                    meta = json.loads(ds.attrs["profile"])
                else:
                    for key, data in ds.items():
                        if len(data.shape) == 3:
                            if "profile" in data.attrs:
                                meta = json.loads(data.attrs["profile"])
                                break

        if not meta:
            raise ValueError("No meta object found.")

        return meta

    @property
    def rasters(self):
        """Return list of all rasters in inputs."""
        rasters = {}
        for name, attrs in self.inputs.items():
            if attrs["path"].split(".")[-1] == "tif":
                rasters[name] = attrs

        return rasters

    def reformat_rasters(self):
        """Reformat all raster files in inputs."""
        # Sequential processing: transform vectors into rasters
        dsts = []
        for name, attrs in tqdm(self.rasters.items()):
            # Unpack attributes
            path = attrs["path"]

            # Create dst path
            dst_name = name + ".tif"
            dst = os.path.join(self.out_dir, dst_name)
            dsts.append(dst)

            # Reformat vector
            self.reformat_raster(
                path=path,
                dst=dst
            )

            # Add formatted path to input
            self.inputs[name]["formatted_path"] = dst

        return dsts

    def reformat_raster(self, path, dst):
        """Resample and re-project a raster."""
        if not os.path.exists(path):
            raise OSError(f"{path} does not exist.")

        if self.template is not None:
            try:
                self._check_dims(path)
                return
            except:
                pass

        if os.path.exists(dst) and not self.overwrite_tif:
            return
        else:
            # If the tiff is too big make adjustments to avoid big tiff errors
            # Reprojecting separately might do the trick
            sp.call([
                "rio", "warp", path, dst,
                "--like", self.template,
                "--co", "blockysize=128",
                "--co", "blockxsize=128",
                "--co", "compress=lzw",  # Check this
                "--co", "tiled=yes",
                "--overwrite"
            ])

    def reformat_vectors(self):
        """Reformat all vector files in inputs."""
        # Sequential processing: transform vectors into rasters
        dsts = []
        for name, attrs in tqdm(self.vectors.items()):
            # Unpack attributes
            path = attrs["path"]
            if "field" in attrs:
                field = attrs["field"]
            else:
                field = None

            if "buffer" in attrs:
                buffer = attrs["buffer"]
            else:
                buffer = None

            # Create dst path
            dst_name = name + ".tif"
            dst = os.path.join(self.out_dir, dst_name)
            dsts.append(dst)

            # Reformat vector
            self.reformat_vector(
                name=name,
                path=path,
                dst=dst,
                field=field,
                buffer=buffer
            )

            # Add formatted path to input
            self.inputs[name]["formatted_path"] = dst

        return dsts

    def reformat_vector(self, name, path, dst, field=None,  buffer=None):
        """Preprocess, re-project, and rasterize a vector."""
        # Check that path exists
        if not os.path.exists(path):
            raise OSError(f"{path} does not exist.")

        # Skip if overwrite
        if not self.overwrite_tif and os.path.exists(dst):
            return
        else:
            # Read and process file
            gdf = self._process_vector(name, path, field, buffer)
            meta = self.meta

            # Rasterize
            elements = gdf.values
            shapes = [(g, r) for r, g in elements]
            out_shape = [meta["height"], meta["width"]]
            transform = meta["transform"]
            with rio.Env():
                array = features.rasterize(shapes, out_shape, all_touched=True,
                                           transform=transform)

            dtype = str(array.dtype)
            if "int" in dtype:
                nodata = np.iinfo(dtype).max
            else:
                nodata = np.finfo(dtype).max
            meta["dtype"] = dtype
            meta["nodata"] = nodata

            # Write to a raster
            with rio.Env():
                with rio.open(dst, "w", **meta) as rio_dst:
                    rio_dst.write(array, 1)

    def to_h5(self):
        """transform all formatted rasters into a h5 file"""
        exclusions = Exclusions(self.excl_fpath, self.lookup)
        for name, attrs in tqdm(self.inputs.items(), total=len(self.inputs)):
            if "formatted_path" in attrs:
                file = attrs["formatted_path"]
                exclusions.add_layer(name, file, overwrite=self.overwrite_dset)

                # Add attributes
                with h5py.File(self.excl_fpath, "r+") as ds:
                    for akey, attr in attrs.items():
                        if attr:
                            ds[name].attrs[akey] = attr

    @property
    def vectors(self):
        """Return list of all shapefiles in inputs."""
        vectors = {}
        for name, attrs in self.inputs.items():
            if not os.path.exists(attrs["path"]):
                raise OSError(f"{attrs['path']} does not exist.")
            if attrs["path"].split(".")[-1] in ["gpkg", "shp", "geojson"]:
                vectors[name] = attrs
        return vectors

    def _check_input_fields(self, inputs):
        """Check that an input table or CSV contains all needed fields."""
        check = all([field in inputs.columns for field in NEEDED_FIELDS])
        assert check, ("Not all required fields present in inputs: "
                       f"{NEEDED_FIELDS}")

    def _parse_inputs(self, inputs):
        """Take input values and create standard dictionary."""
        # Create dictionary
        if isinstance(inputs, dict):
            self.inputs = inputs
        else:
            self._check_input_fields(inputs)
            self.inputs = {}
            if isinstance(inputs, str):
                inputs = pd.read_csv(inputs)
            inputs = inputs[~pd.isnull(inputs["name"])]
            inputs = inputs[~pd.isnull(inputs["path"])]
            inputs = inputs[inputs["name"] != "\xa0"]
            inputs = inputs[inputs["path"] != "\xa0"]

            cols = [c for c in inputs.columns if c != "name"]
            for i, row in inputs.iterrows():
                self.inputs[row["name"]] = {}
                for col in cols:
                    self.inputs[row["name"]][col] = row[col]

        # Convert NaNs to None
        for key, attrs in self.inputs.items():
            for akey, attr in attrs.items():
                if not isinstance(attr, str):
                    if attr is not None:
                        if np.isnan(attr):
                            self.inputs[key][akey] = None

    def _map_strings(self, layer_name, gdf, field):
        """Map string values to integers and save a lookup dictionary."""
        # Assing integers to unique string values
        strings = gdf["raster_value"].unique()
        string_map = {i + 1: v for i, v in enumerate(strings)}
        value_map = {v: k for k, v in string_map.items()}

        # Replace strings with integers
        gdf["raster_value"] = gdf[field].map(value_map)

        # Update the string lookup dictionary
        self.lookup[layer_name] = string_map
    
        return gdf

    def _process_vector(self, name, path, field=None, buffer=None):
        """Process a single vector file."""
        # Read in file and check the path
        gdf = gpd.read_file(path)

        if gdf.shape[0] == 0:
            raise IndexError(f"{path} is an empty file.")

        # Check the projection;
        if not crs_match(gdf.crs, self.meta["crs"]):
            gdf = gdf.to_crs(self.meta["crs"])

        # Check if the field value in the shapefile and Assign raster value
        if field:
            if field not in gdf.columns:
                raise Exception(f"Field '{field}' not in '{path}'")
            gdf["raster_value"] = gdf[field]
        else:
            gdf["raster_value"] = 1

        # Account for string values
        sample = gdf["raster_value"].iloc[0]
        if isinstance(sample, str):
            if isint(sample):
                gdf["raster_value"] = gdf["raster_value"].astype(int)
            elif isfloat(sample):
                gdf["raster_value"] = gdf["raster_value"].astype(float)
            else:
                gdf = self._map_strings(name, gdf, field)

        # Reduce to two fields
        gdf = gdf[["raster_value", "geometry"]]

        # Buffering
        if buffer:
            gdf["geometry"] = gdf["geometry"].buffer(buffer)

        return gdf

    def main(self):
        """Reformat all vectors and rasters listed in the inputs."""
        print("Formatting rasters...")
        _ = self.reformat_rasters()

        print("Formatting vectors...")
        _ = self.reformat_vectors()
        if self.excl_fpath:
            print(f"Building/updating exclusion {self.excl_fpath}...")
            self.to_h5()

