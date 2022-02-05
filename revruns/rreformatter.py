# -*- coding: utf-8 -*-
"""Reformat a shapefile or raster into a reV raster using a template.

Ideally, any raster or vector format will work with this.

Created on Tue Apr 27 16:47:10 2021

Updated in Feb, 2,2022 

@author: twillia2
"""
import glob
import json
import subprocess as sp
import os

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import rasterio as rio

from pyproj import CRS, Transformer
from rasterio.errors import RasterioIOError

from tqdm import tqdm
from pathos import multiprocessing as mp
from rasterio import features


def crs_match(crs1, crs2):
    """Check if two coordinate reference systems match."""
    # Using strings and CRS objects directly is not consistent enough
    check = False
    crs1 = CRS(crs1).to_dict()
    crs2 = CRS(crs2).to_dict()
    for key, value in crs1.items():
        if key in crs2:
            try:
                assert value == crs2[key]
                check = True
            except AssertionError:
                print(".")
        else:
            check = True
            print(f"crs2 is missing the {key} key...")

    return check


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
        try:
            raster = rio.open(file)
        except Exception:
            raise RasterioIOError("file " + file + " does not exist")

        # Get profile information
        profile = raster.profile
        
        profile["crs"] = profile["crs"].to_proj4()
        dtype = profile["dtype"]
        profile = dict(profile)

        # We need a 6 element geotransform, sometimes we receive three extra
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
    """Reformat any file into a reV-shaped raster."""

    def __init__(self, template, input_table_path,raster_dir,exclusion_file,
                sheet_name=None,overwrite=False):
        """Initialize Reformatter object.

        Parameters
        ----------
        template : str
            Path to a raster to use as a template.
        input_table_path: str
            path to a raster 
        """
        self.template = template
        self.string_lookup = {}
        self.vectorfile_fields = self._format_input(
            input_table_path,
            sheet_name
        )
        self.raster_dir = raster_dir
        self.overwrite = overwrite
        super().__init__(exclusion_file, self.string_lookup)

    def __repr__(self):
        """Return object representation string."""
        return f"<Reformatter: template={self.template}>"

    def reformat_rasters(self):
        """Resample and re-project rasters."""
        files = self.rasters
        n_cpu = self.os.cpu_count()
        dsts = []
        with mp.Pool(n_cpu) as pool:
            for dst in self.tqdm(pool.imap(self.reformat_raster, files),
                            total=n_cpu):
                dsts.append(dst)
        return dsts
    
    def reformat_raster(self, file, dst):
        """Resample and re-project a raster."""
        dst = os.path.join(dst_dir, os.path.basename(file))  # <--------------- dst_dir not defined, self.raster_output_folder, pull name from input sheet key
        sp.call(["rio", "warp", file, dst,
                 "--like", TEMPLATE,
                 "--co", "blockysize=128",
                 "--co", "blockxsize=128",
                 "--co", "compress=lzw",
                 "--co", "tiled=yes"])
    
    def reformat_batch_vectorfiles(self):
        """Batch process the shapefiles indicated by the shapefile_fields; output are rasters"""
        
        # create output raster path
        vectorfile_fields = self.vectorfile_fields
        # print(vectorfile_fields)

        # sequential processing
        for key,value in tqdm(vectorfile_fields.items()):
          
            # get the vector file_path, field and buffer value 
            field_name = value["field"]
            buffer = value["buffer"]
            vector_file_path = value["path"]
            
            # create dst path
            dst_name = key + ".tif"
            dst = os.path.join(self.raster_dir, dst_name)

            # single cpu sequential process
            # if buffer is NaN
            if np.isnan(buffer):
                self.reformat_vectorfile(layer_name = key,file=vector_file_path, dst=dst, field=field_name,buffer=None,overwrite=self.overwrite)
            else:
                self.reformat_vectorfile(layer_name = key,file=vector_file_path, dst=dst, field=field_name,buffer=buffer,overwrite=self.overwrite)
        
        # get the number of cpu #<-------------------------parallel processing hasn't been implemented ---------<<<<<<<<<<
        # n_cpu = self.os.cpu_count()
        # parallel process the vector files 
        # dsts=[]
        # with self.mp.Pool(n_cpu) as pool:
        #     for t in self.tqdm(pool.map(self.reformat_vectorfile,vector_files,dst,field_names),
        #                     total=n_cpu):
        #         dsts.append(t)
        # return dsts
    
    def reformat_vectorfile(self, layer_name, file, dst, field=None,
                            buffer=None, overwrite=False):
        """Preprocess, re-project, and rasterize a vector."""
        # Read and process file
        gdf = self._process_vectorfile(layer_name,file, field, buffer)
        meta = self.meta

        # Skip if overwrite
        if not overwrite and os.path.exists(dst):
            return
        else:
            # Rasterize
            elements = gdf[["raster_value", "geometry"]].values
            shapes = [(g, r) for r, g in elements]

            out_shape = [meta["height"], meta["width"]]
            transform = meta["transform"]
            with rio.Env():
                array = features.rasterize(shapes, out_shape,
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

    def reformat_batch_to_h5(self):
        raster_files = glob.glob(self.os.path.join(self.raster_dir,"*.tif"))
        for file in raster_files:
            dataset_name = os.path.basename(file).split(".")[0]
            description = self.vectorfile_fields[dataset_name]["description"]
            self.add_layer(dataset_name,file,description=description)
    
    @property
    def meta(self):
        """Return the meta information from the template file."""
        with rio.open(self.template) as raster:
            meta = raster.meta
        return meta
    
    @property
    def rasters(self):
        """Return list of all rasters in project rasters folder."""
        print(self.raster_dir)
        return glob.glob(self.os.path.join(self.raster_dir, "*.tif"))

    @property
    def shapefiles(self):
        """Return list of all shapefiles in project shapefiles folder."""
        shps = self.add_layerglob(os.path.join(self.shapefile_dir, "*shp"))
        gpkgs = glob(os.path.join(self.shapefile_dir, "*gpkg"))
        geojsons = glob(os.path.join(self.shapefile_dir, "*geojson"))
        shapefiles = shps + gpkgs + geojsons
        return shapefiles

    def _format_input(self, path, sheet_name):
        """ to format the input csv or excel """
        # read csv or excel
        try:
            input_table = pd.read_csv(path,header=0)
        except:
            input_table = pd.read_excel(path, sheet_name, header=0)
        
        # check the columns required
        column_names = input_table.columns
        for name in ["name", "path", "field", "buffer", "description"]:
            if name not in column_names:
                raise Exception(
                    "Column: {} is not in the input table.".format(name))

        # iterate the table, create a vector_file - field dictionary
        vectorfile_fields = {}
        for index, row in input_table.iterrows():
            item = vectorfile_fields.get(row["name"], dict())
            item["path"] = row["path"]
            item["field"] = row["field"]
            item["buffer"] = row["buffer"]
            item["description"] = row["description"]
            vectorfile_fields[row["name"]] = item

        return vectorfile_fields
    
    def _process_vectorfile(self, layer_name, file, field=None, buffer=None):
        """Process a single file. It includes following steps:
        1. check if the a field exists in a shapefile;
            to make sure the field name and its associated shapefile's path is correct; 
            otherwise raise an error
        2. check if the shapefile has a the same projetion as the template raster
            if not, re-project
        3. if buffer isn't non, then buffer the shapefile 
         """

        # Read in file and check the path
        if not os.path.exists(file):
            raise Exception(
                "This shapefile path doesn't exist:{}".format(file))
        else:
            gdf = gpd.read_file(file)

        # Check the projection;
        if str(gdf.crs).upper() != str(self.meta["crs"]).upper():
            gdf = gdf.to_crs(self.meta["crs"])

        # Check if the field value in the shapefile and Assign raster value
        if field:
            if field not in gdf.columns:
                raise Exception("Field '{}' not in '{}'".format(field, file))
            gdf["raster_value"] = gdf[field]
        else:
            gdf["raster_value"] = 1

        # Account for string values
        if isinstance(gdf["raster_value"].iloc[0], str):
            gdf = self._map_strings(layer_name, gdf, field)

        # Reduce to two fields
        gdf = gdf[["raster_value", "geometry"]]

        # Buffering
        if buffer:
            gdf["geometry"] = gdf["geometry"].buffer(buffer)
        return gdf
    
    def _map_strings(self, layer_name, gdf, field):
        """Map string values to integers and save a lookup dictionary."""
        # Assing integers to unique string values
        strings = gdf[field].unique()
        string_map = {i + 1: v for i, v in enumerate(strings)}
        value_map = {v: k for k, v in string_map.items()}

        # Replace strings with integers
        gdf["raster_value"] = gdf[field].map(value_map)

        # Update the string lookup dictionary
        self.string_lookup[layer_name] = string_map
        
        return gdf


if __name__ == "__main__":
    # rev input table
    table_path = "/Users/jgu3/GDS/01-Project/02-Luma/01-Rawdata/offshore/rev_input.csv"

    # output raster folder
    raster_out_folder = "/Users/jgu3/GDS/01-Project/02-Luma/03-exclusions/raster_batch_output"

    # output h5 file
    h5_path = "/Users/jgu3/GDS/01-Project/02-Luma/03-exclusions/h5_output/PR.h5"

    # raster template
    TEMPLATE = "/Users/jgu3/GDS/01-Project/02-Luma/01-Rawdata/template/pr_template_32161.tif"
    
    reformat = Reformatter(TEMPLATE, table_path,raster_dir=raster_out_folder,
                            sheet_name="rev_input",exclusion_file= h5_path)
    
    # reformat.reformat_batch_vectorfiles()
    # reformat.reformat_batch_to_h5()
    print(reformat.rasters)