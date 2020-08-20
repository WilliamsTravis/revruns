"""
Almost all of the functionality is currently stored in the CLI scripts to
avoid the load time needed to load shared functions, but anything in here can
be accessed through revruns.functions. Place new functions that might be useful
in the future here.

Created on Wed Dec  4 07:58:42 2019

@author: twillia2
"""

import datetime as dt
import json
import multiprocessing as mp
import os
import ssl
import warnings

from glob import glob

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import rasterio as rio

from rasterio.errors import RasterioIOError
from shapely.geometry import Point
from tqdm import tqdm
from xlrd import XLRDError

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 20)
pd.options.mode.chained_assignment = None


# Functions
def isint(x):
    try:
        int(x)
        return True
    except ValueError:
        return False


def get_sheet(file_name, sheet_name=None, starty=0, startx=0, header=True):
    """Read in/check available sheets from an excel spreadsheet file."""

    # Open file
    file = pd.ExcelFile(file_name)
    sheets = file.sheet_names

    # Run with no sheet_name for a list of available sheets
    if not sheet_name:
        print("No sheet specified, returning a list of available sheets.")
        return sheets

    # Try to open sheet, print options if it fails
    try:
        table = file.parse(sheet_name=sheet_name)
    except XLRDError:
        print(sheet_name + " is not available. Available sheets:\n")
        for s in sheets:
            print("   " + s)

    return table

            
def point_line(arg):
    """
    Find the closest transmission line to a point and return a gid and distance.

    Parameters
    ----------
    row : pd.
        A pandas series with a "geometry" column.
    linedf : geopandas.geodataframe.GeoDataFrame
        A geodataframe with a trans_line_gid and shapely geometry objects.

    Returns
    -------
    tuple
        A tuple with the transmission gid, the distance from the point to it,
        and the category of the transmission connection structure.
    """

    df, linedf = arg

    # Find the closest point on the closest line to the target point
    def single_row(point, linedf):
        distances = [point.distance(l) for l in linedf["geometry"]]
        dmin = np.min(distances)
        idx = np.where(distances == dmin)[0][0]

        # We need this gid and this category
        gid = linedf.index[idx]
        category = linedf["category"].iloc[idx]
    
        return gid, dmin, category

    return df.shape
#     distances = df["geometry"].apply(single_row, linedf=linedf)


def write_config(config_dict, path):
    """ Write a configuration dictionary to a json file."""
    with open(path, "w") as file:
        file.write(json.dumps(config_dict, indent=4))


# Class methods
class Data_Path:
    """Data_Path joins a root directory path to data file paths."""

    def __init__(self, data_path):
        """Initialize Data_Path."""

        self.data_path = data_path
        self._expand_check()

    def __repr__(self):

        items = ["=".join([str(k), str(v)]) for k, v in self.__dict__.items()]
        arguments = " ".join(items)
        msg = "".join(["<Data_Path " + arguments + ">"])
        return msg

    def join(self, *args):
        """Join a file path to the root directory path"""

        return os.path.join(self.data_path, *args)

    def contents(self, *args):
        """List all content in the data_path or in sub directories."""

        if not any(["*" in a for a in args]):
            items = glob(self.join(*args, "*"))
        else:
            items = glob(self.join(*args))

        return items

    def folders(self, *args):
        """List folders in the data_path or in sub directories."""

        items = self.contents(*args)
        folders = [i for i in items if os.path.isdir(i)]

        return folders

    def files(self, pattern=None, *args):
        """List files in the data_path or in sub directories."""

        items = self.contents(*args)
        files = [i for i in items if os.path.isfile(i)]
        if pattern:
            files = [f for f in files if pattern in f]
            if len(files) == 1:
                files = files[0]

        return files

    def _expand_check(self):

        # Expand the user path if a tilda is present in the root folder path.
        if "~" in self.data_path:
            self.data_path = os.path.expanduser(self.data_path)

        # Make sure path exists
        os.makedirs(self.data_path, exist_ok=True)


class Exclusions:
    """Build or add to an HDF5 Exclusions dataset."""

    def __init__(self, excl_fpath):
        """Initialize Exclusions object."""
        
        self.excl_fpath = excl_fpath
        self._initialize_h5()

    def __repr__(self):
        msg = "<Exclusions Object:  excl_fpath={}>".format(self.excl_fpath)
        return msg

    def add_layer(self, dname, file, description=None, overwrite=False):
        """Add a raster file and its description to the HDF5 exclusion file."""

        # Open raster object
        try:
            raster = rio.open(file)
        except:
            raise RasterioIOError("file " + file + " does not exist")

        profile = raster.profile
        profile["crs"] = profile["crs"].to_proj4()
        dtype = profile["dtype"]
        profile = json.dumps(dict(profile))

        # Add coordinates and else check that the new file matches everything
        self._set_coords(raster)
        self._check_dims(raster, profile, dname)

        # Add everything to target exclusion HDF
        array = raster.read()
        with h5py.File(self.excl_fpath, "r+") as hdf:
            keys = list(hdf.keys())
            if dname in keys:
                if overwrite:
                    del hdf[dname]

            if not dname in keys:
                hdf.create_dataset(name=dname, data=array, dtype=dtype,
                                   chunks=(1, 128, 128))
                hdf[dname].attrs["file"] = os.path.abspath(file)
                hdf[dname].attrs["profile"] = profile
                if description:
                    hdf[dname].attrs["description"] = description

    def add_layers(self, file_dict, desc_dict=None, overwrite=False):
        """Add multiple raster files and their descriptions."""

        # If description are provided make sure they match the files
        if desc_dict:
            try:
                dninf = [k for k in desc_dict if k not in file_dict]
                fnind = [k for k in file_dict if k not in desc_dict]
                assert not dninf
                assert not fnind
            except:
                mismatches = np.unique(dninf + fnind)
                msg = ("File and description keys do not match. "
                       "Problematic keys: " + ", ".join(mismatches))
                raise AssertionError(msg)
        else:
            desc_dict = {key: None for key in file_dict.keys()}

        # Should we parallelize this?
        for key, file in tqdm(file_dict.items(), total=len(file_dict)):
            description = desc_dict[key]
            self.add_layer(key, file, description, overwrite=overwrite)

    def techmap(self, res_fpath, dname, max_workers=None, map_chunk=2560,
                distance_upper_bound=None, save_flag=True):
        """
        Build a technical resource mapping grid between exclusion rasters cells
        and resource points.

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
                                 max_workers=None, distance_upper_bound=None,
                                 map_chunk=2560, save_flag=save_flag)
        return arrays

    def _check_dims(self, raster, profile, dname):
        # Check new layers against the first added raster
        with h5py.File(self.excl_fpath, "r") as hdf:
            
            # Find any existing layers (these will have profiles)
            lyrs = [k for k in hdf.keys() if hdf[k] and "profile" in
                    hdf[k].attrs.keys()]

            if lyrs:
                key = lyrs[0]
                old = json.loads(hdf[key].attrs["profile"])
                new = json.loads(profile)

                # Check the CRS
                try:
                    assert old["crs"] == new["crs"]
                except:
                    raise AssertionError("CRS for " + dname + " does not match"
                                         " exisitng CRS.")

                # Check the transform
                try:
                    
                    # Standardize these
                    old_trans = old["transform"][:6]
                    new_trans = new["transform"][:6]
                    
                    assert old_trans == new_trans
                except:
                    raise AssertionError("Geotransform for " + dname + " does "
                                         "not match geotransform.")

                # Check the dimesions
                try:
                    assert old["width"] == new["width"]
                    assert old["height"] == new["height"]
                except:
                    raise AssertionError("Width and/or height for " + dname +
                                         " does not match exisitng " +
                                         "dimensions.")

    def _get_coords(self, raster):
        # Get x and y coordinates (One day we'll have one transform order!)
        profile = raster.profile
        geom = raster.profile["transform"]
        xres = geom[0]
        xrot = geom[1]
        ulx = geom[2]
        yrot = geom[3]
        yres = geom[4]
        uly = geom[5]
        
        # Not doing rotations here
        xs = [ulx + col * xres for col in range(profile["width"])]
        ys = [uly + row * yres for row in range(profile["height"])]

        return xs, ys

    def _set_coords(self, raster):
        # Add the lat and lon meshgrids if they aren't already present
        with h5py.File(self.excl_fpath, "r+") as hdf:
            keys = list(hdf.keys())
            if not "latitude" in keys or not "longitude" in keys:
                xs, ys = self._get_coords(raster)
                xgrid, ygrid = np.meshgrid(xs, ys)
                hdf.create_dataset(name="longitude", data=xgrid)
                hdf.create_dataset(name="latitude", data=ygrid)

    def _initialize_h5(self):
        # Create an empty hdf file if one doesn't exist
        date = format(dt.datetime.today(), "%Y-%m-%d %H:%M")
        os.makedirs(os.path.dirname(self.excl_fpath), exist_ok=True)
        if not os.path.exists(self.excl_fpath):
            with h5py.File(self.excl_fpath, "w") as ds:
                ds.attrs["creation_date"] = date


@pd.api.extensions.register_dataframe_accessor("rr")
class PandasExtension:
    """Making dealing with meta objects easier."""

    def __init__(self, pandas_obj):
        if type(pandas_obj) !=  pd.core.frame.DataFrame:
            if type(pandas_obj) !=  gpd.geodataframe.GeoDataFrame:
                raise TypeError("Can only use .rr accessor with a pandas or "
                                     "geopandas data frame.")
        self._obj = pandas_obj

    def decode(self):
        """ Decode the columns of a meta data object from a reV output. 
    
        Fix:
            When an HDF has been transfered or synced across networks, columns
            with byte format might be stored as strings...meaning that they
            will be strings of bytes of strings (i.e. "b'string'").
        """

        for c in self._obj.columns:
            if isinstance(self._obj[c].iloc[0], bytes):
                try:
                    self._obj[c] = self._obj[c].apply(lambda x: x.decode())
                except:
                    self._obj[c] = None
                    print("Column " + c + " could not be decoded.")

    def dist_apply(self, linedf):
        """To apply the distance function in parallel."""  # <------------------ Not quite done

        ncpu = mp.cpu_count()
        chunks = np.array_split(self._obj.index, ncpu)
        args = [(self._obj.loc[idx], linedf) for idx in chunks]
        distances = []
        with mp.Pool() as pool:
            for dists in tqdm(pool.imap(point_line, args),
                              total=len(args)):
                distances.append(dist)
        return distances

    def to_bbox(self, bbox):
        """Return points filtered by a bounding box ([xmin, ymin, xmax, ymax])
        """

        df = self._obj.copy()
        df = df[(df["longitude"] >= bbox[0]) &
                (df["latitude"] >= bbox[1]) &
                (df["longitude"] <= bbox[2]) &
                (df["latitude"] <= bbox[3])]
        return df

    def to_geo(self, latcol=None, loncol=None):
        """ Convert a Pandas data frame to a geopandas geodata frame """

        # Let's not transform in place
        df = self._obj.copy()
        df.rr.decode()

        # Find coordinate columns
        if not latcol or not loncol:
            latcol, loncol = self.find_coords()

        # For a single row
        to_point = lambda x: Point(tuple(x))
        df["geometry"] = df[[loncol, latcol]].apply(to_point, axis=1)
        gdf = gpd.GeoDataFrame(df, crs='epsg:4326', geometry="geometry")
        return gdf

    def to_sarray(self):
        """Create a structured array for storing in HDF5 files."""

        # Create a copy
        df = self._obj.copy()

        # For a single column
        def make_col_type(col, types):
            
            coltype = types[col]
            column = df.loc[:, col]
            
            try:
                if 'numpy.object_' in str(coltype.type):
                    maxlens = column.dropna().str.len()
                    if maxlens.any():
                        maxlen = maxlens.max().astype(int)
                        coltype = ('S%s' % maxlen)
                    else:
                        coltype = 'f2'
                return column.name, coltype
            except:
                print(column.name, coltype, coltype.type, type(column))
                raise

        # All values and types
        v = df.values
        types = df.dtypes
        struct_types = [make_col_type(col, types) for col in df.columns]
        dtypes = np.dtype(struct_types)

        # The target empty array
        array = np.zeros(v.shape[0], dtypes)

        # For each type fill in the empty array
        for (i, k) in enumerate(array.dtype.names):
            try:
                if dtypes[i].str.startswith('|S'):
                    array[k] = df[k].str.encode('utf-8').astype('S')
                else:
                    array[k] = v[:, i]
            except:
                raise
    
        return array, dtypes

    def find_coords(self):
        """Check if lat/lon names are in a pre-made list of possible names."""

        # List all column names
        df = self._obj.copy()
        cols = df.columns

        # For direct matches
        ynames = ["y", "lat", "latitude", "ylat"]
        xnames = ["x", "lon", "long", "longitude", "xlon", "xlong"]

        # Direct matches
        possible_ys = [c for c in cols if c in ynames]
        possible_xs = [c for c in cols if c in xnames]

        # If no matches return item and rely on manual entry
        if len(possible_ys) == 0 or len(possible_xs) == 0:
            raise ValueError("No field names found for coordinates, use "
                             "latcol and loncol arguments.")

        # If more than one match raise error
        elif len(possible_ys) > 1:
            raise ValueError("Multiple possible entries found for y/latitude "
                             "coordinates, use latcol argument: " +
                             ", ".join(possible_ys))
        elif len(possible_xs) > 1:
            raise ValueError("Multiple possible entries found for y/latitude "
                             "coordinates, use latcol argument: " +
                             ", ".join(possible_xs))

        # If there's just one column use that
        else:
            return possible_ys[0], possible_xs[0]
