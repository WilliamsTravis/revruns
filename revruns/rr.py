"""Revruns Functions.

Almost all of the functionality is currently stored in the CLI scripts to
avoid the load time needed to load shared functions, but anything in here can
be accessed directly from revruns. Place new functions that might be useful
in the future here.

Created on Wed Dec  4 07:58:42 2019

@author: twillia2
"""
import json
import os
import shutil

from glob import glob

import h5py
import numpy as np
import pandas as pd

from osgeo import gdal
from tqdm import tqdm

pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 20)
pd.options.mode.chained_assignment = None


def crs_match(crs1, crs2):
    """Check if two coordinate reference systems match."""
    from pyproj import CRS

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


def get_sheet(file_name, sheet_name=None, starty=0, startx=0, header=0):
    """Read in/check available sheets from an excel spreadsheet file."""
    from xlrd import XLRDError

    # Open file
    file = pd.ExcelFile(file_name)
    sheets = file.sheet_names

    # Run with no sheet_name for a list of available sheets
    if not sheet_name:
        print("No sheet specified, returning a list of available sheets.")
        return sheets
    if sheet_name not in sheets:
        raise ValueError(sheet_name + " not in file.")

    # Try to open sheet, print options if it fails
    try:
        table = file.parse(sheet_name=sheet_name, header=header)
    except XLRDError:
        print(sheet_name + " is not available. Available sheets:\n")
        for s in sheets:
            print("   " + s)

    return table


def h5_to_csv(src, dst, dataset):
    """Reformat a reV outpur HDF5 file/dataset to a csv."""
    # Read in meta, time index, and data
    # with h5py.File(src, "r") as ds:
    ds = h5py.File(src, "r")
    # Get an good time index
    if "multi" in src:
        time_key = [ti for ti in ds.keys() if "time_index" in ti][0]
        data = [ds[d][:] for d in ds.keys() if dataset in d]
        data = np.array(data)
    else:
        time_key = "time_index"
        data = ds[dataset][:]

    # Read in needed elements
    meta = pd.DataFrame(ds["meta"][:])
    time_index = [t.decode()[:-9] for t in ds[time_key][:]]

    # Decode meta, use as base
    meta.rr.decode()

    # If its 1-D just give just append the data to meta
    if len(data.shape) == 1:
        meta[dataset] = data
        df = meta.copy()

    # If its more than 1-D, label the array and use that as the table
    elif len(data.shape) > 1:
        # If its 3-D, its a mult-year, find the mean profile first
        if len(data.shape) == 3:
            time_index = [t[5:] for t in time_index]
            data = np.mean(data, axis=0)

        # Now its def 2-D
        df = pd.DataFrame(data)
        df["time_index"] = time_index
        cols = [str(c) for c in df.columns]
        df.columns = cols
        cols = ["time_index"] + cols[:-1]
        df = df[cols]

    df.to_csv(dst, index=False)

    return 0


def isint(x):
    """Check if character string is an integer."""
    try:
        int(x)
        return True
    except ValueError:
        return False


def mode(x):
    """Return the mode of a list of values."""  # <---------------------------- Works with numpy's max might break here
    return max(set(x), key=x.count)


def par_apply(df, field, fun):
    """Apply a function in parallel to a pandas data frame field."""
    import numpy as np
    import pathos.multiprocessing as mp

    from tqdm import tqdm

    def single_apply(arg):
        """Apply a function to a pandas data frame field."""
        cdf, field, fun = arg
        try:
            values = cdf[field].apply(fun)
        except Exception:
            raise
        return values

    ncpu = mp.cpu_count()
    cdfs = np.array_split(df, ncpu)
    args = [(cdf, field, fun) for cdf in cdfs]

    values = []
    with mp.Pool(ncpu) as pool:
        for value in tqdm(pool.imap(single_apply, args), total=ncpu):
            values.append(value)
    values = [v for sv in values for v in sv]

    return values


def write_config(config_dict, path):
    """Write a configuration dictionary to a json file."""
    with open(path, "w") as file:
        file.write(json.dumps(config_dict, indent=4))


class Data_Path:
    """Data_Path joins a root directory path to data file paths."""

    def __init__(self, data_path=".", mkdir=False, warnings=True):
        """Initialize Data_Path."""
        data_path = os.path.abspath(os.path.expanduser(data_path))
        self.data_path = data_path
        self.last_path = os.getcwd()
        self.warnings = warnings
        self._exist_check(data_path, mkdir)
        self._expand_check()

    def __repr__(self):
        """Print the data path."""
        items = ["=".join([str(k), str(v)]) for k, v in self.__dict__.items()]
        arguments = ", ".join(items)
        msg = "".join(["<Data_Path " + arguments + ">"])
        return msg

    def contents(self, *args, recursive=False):
        """List all content in the data_path or in sub directories."""
        if not any(["*" in a for a in args]):
            items = glob(self.join(*args, "*"), recursive=recursive)
        else:
            items = glob(self.join(*args), recursive=recursive)
        items.sort()
        return items

    def extend(self, path, mkdir=False):
        """Return a new Data_Path object with an extended home directory."""
        new = Data_Path(os.path.join(self.data_path, path), mkdir)
        return new

    def folders(self, *args, recursive=False):
        """List folders in the data_path or in sub directories."""
        items = self.contents(*args, recursive=recursive)
        folders = [i for i in items if os.path.isdir(i)]
        return folders

    def files(self, *args, recursive=False):
        """List files in the data_path or in sub directories."""
        items = self.contents(*args, recursive=recursive)
        files = [i for i in items if os.path.isfile(i)]
        return files

    def join(self, *args, mkdir=False):
        """Join a file path to the root directory path."""
        path = os.path.join(self.data_path, *args)
        self._exist_check(path, mkdir)
        path = os.path.abspath(path)
        return path

    @property
    def base(self):
        """Return the base name of the home directory."""
        return os.path.basename(self.data_path)

    @property
    def back(self):
        """Change directory back to last working directory if home was used."""
        os.chdir(self.last_path)
        print(self.last_path)

    @property
    def home(self):
        """Change directories to the data path."""
        self.last_path = os.getcwd()
        os.chdir(self.data_path)
        print(self.data_path)


    def _exist_check(self, path, mkdir=False):
        """Check if the directory of a path exists, and make it if not."""
        # If this is a file name, get the directory
        if "." in path:  # Will break if you use "."'s in your directories
            directory = os.path.dirname(path)
        else:
            directory = path

        # Don't try this with glob patterns
        if "*" not in directory:
            if not os.path.exists(directory):
                if mkdir:
                    if self.warnings:
                        print(f"Warning: {directory} did not exist, "
                              "creating directory.")
                    os.makedirs(directory, exist_ok=True)
                else:
                    if self.warnings:
                        print(f"Warning: {directory} does not exist.")

    def _expand_check(self):
        # Expand the user path if a tilda is present in the root folder path.
        if "~" in self.data_path:
            self.data_path = os.path.expanduser(self.data_path)


@pd.api.extensions.register_dataframe_accessor("rr")
class PandasExtension:
    """Accessing useful pandas functions directly from a data frame object."""

    import multiprocessing as mp
    import warnings

    from json import JSONDecodeError

    import geopandas as gpd
    import pandas as pd
    import numpy as np

    from scipy.spatial import cKDTree
    from shapely.geometry import Point

    def __init__(self, pandas_obj):
        """Initialize PandasExtension object."""
        self.warnings.simplefilter(action='ignore', category=UserWarning)
        if type(pandas_obj) != self.pd.core.frame.DataFrame:
            if type(pandas_obj) != self.gpd.geodataframe.GeoDataFrame:
                raise TypeError("Can only use .rr accessor with a pandas or "
                                "geopandas data frame.")
        self._obj = pandas_obj

    def average(self, value, weight="n_gids", group=None):
        """Return the weighted average of a column.

        Parameters
        ----------
        value : str
            Column name of the variable to calculate.
        weight : str
            Column name of the variable to use as the weights. The default is
            'n_gids'.
        group : str, optional
            Column name of the variable to use to group results. The default is
            None.

        Returns
        -------
        dict | float
            Single value or a dictionary with group, weighted average value
            pairs.
        """
        df = self._obj.copy()
        if not group:
            values = df[value].values
            weights = df[weight].values
            x = self.np.average(values, weights=weights)
        else:
            x = {}
            for g in df[group].unique():
                gdf = df[df[group] == g]
                values = gdf[value].values
                weights = gdf[weight].values
                x[g] = self.np.average(values, weights=weights)
        return x

    def bmap(self):
        """Show a map of the data frame with a basemap if possible."""
        if not isinstance(self._obj, self.gpd.geodataframe.GeoDataFrame):
            print("Data frame is not a GeoDataFrame")

    def decode(self):
        """Decode the columns of a meta data object from a reV output."""
        import ast

        def decode_single(x):
            """Try to decode a single value, pass if fail."""
            try:
                x = x.decode()
            except UnicodeDecodeError:
                x = "indecipherable"
            return x

        for c in self._obj.columns:
            x = self._obj[c].iloc[0]
            if isinstance(x, bytes):
                try:
                    self._obj[c] = self._obj[c].apply(decode_single)
                except Exception:
                    self._obj[c] = None
                    print("Column " + c + " could not be decoded.")
            elif isinstance(x, str):
                try:
                    if isinstance(ast.literal_eval(x), bytes):
                        try:
                            self._obj[c] = self._obj[c].apply(
                                lambda x: ast.literal_eval(x).decode()
                                )
                        except Exception:
                            self._obj[c] = None
                            print("Column " + c + " could not be decoded.")
                except:
                    pass

    def dist_apply(self, linedf):
        """To apply the distance function in parallel (not ready)."""
        from pathos.multiprocessing import ProcessingPool as Pool
        from tqdm import tqdm

        # Get distances
        ncpu = os.cpu_count()
        chunks = self.np.array_split(self._obj.index, ncpu)
        args = [(self._obj.loc[idx], linedf) for idx in chunks]
        distances = []
        with Pool(ncpu) as pool:
            for dists in tqdm(pool.imap(self.point_line, args),
                              total=len(args)):
                distances.append(dists)
        return distances

    def find_coords(self):
        """Check if lat/lon names are in a pre-made list of possible names."""
        # List all column names
        df = self._obj.copy()
        cols = df.columns

        # For direct matches
        ynames = ["y", "lat", "latitude", "Latitude", "ylat"]
        xnames = ["x", "lon", "long", "longitude", "Longitude", "xlon",
                  "xlong"]

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

    def gid_join(self, df_path, fields, agg="mode", left_on="res_gids",
                 right_on="gid"):
        """Join a resource-scale data frame to a supply curve data frame.

        Parameters
        ----------
        df_path : str
            Path to csv with desired join fields.
        fields : str | list
            The field(s) in the right DataFrame to join to the left.
        agg : str
            The aggregating function to apply to the right DataFrame. Any
            appropriate numpy function.
        left_on : str
            Column name to join on in the left DataFrame.
        right_on : str
            Column name to join on in the right DataFrame.

        Returns
        -------
        pandas.core.frame.DataFrame
            A pandas DataFrame with the specified fields in the right
            DataFrame aggregated and joined.
        """
        from pathos.multiprocessing import ProcessingPool as Pool

        # The function to apply to each item of the left dataframe field
        def single_join(x, vdict, right_on, field, agg):
            """Return the aggregation of a list of values in df."""
            x = self._destring(x)
            rvalues = [vdict[v] for v in x]
            rvalues = [self._destring(v) for v in rvalues]
            rvalues = [self._delist(v) for v in rvalues]
            return agg(rvalues)

        def chunk_join(arg):
            """Apply single to a subset of the main dataframe."""
            chunk, df_path, left_on, right_on, field, agg = arg
            rdf = pd.read_csv(df_path)
            vdict = dict(zip(rdf[right_on], rdf[field]))
            chunk[field] = chunk[left_on].apply(single_join, args=(
                vdict, right_on, field, agg
                )
            )
            return chunk

        # Create a copy of the left data frame
        df1 = self._obj.copy()

        # Set the function
        if agg == "mode":
            def mode(x): max(set(x), key=x.count)
            agg = mode
        else:
            agg = getattr(self.np, agg)

        # If a single string is given for the field, make it a list
        if isinstance(fields, str):
            fields = [fields]

        # Split this up and apply the join functions
        chunks = self.np.array_split(df1, os.cpu_count())
        for field in fields:
            args = [(c, df_path, left_on, right_on, field, agg)
                    for c in chunks]
            df1s = []
            with Pool(os.cpu_count()) as pool:
                for cdf1 in pool.imap(chunk_join, args):
                    df1s.append(cdf1)
            df = pd.concat(df1s)

        return df

    def nearest(self, df, fields=None, lat=None, lon=None, no_repeat=False,
                k=5):
        """Find all of the closest points in a second data frame.

        Parameters
        ----------
        df : pandas.core.frame.DataFrame | geopandas.geodataframe.GeoDataFrame
            The second data frame from which a subset will be extracted to
            match all points in the first data frame.
        fields : str | list
            The field(s) in the second data frame to append to the first.
        lat : str
            The name of the latitude field.
        lon : str
            The name of the longitude field.
        no_repeat : logical
            Return closest points with no duplicates. For two points in the
            left dataframe that would join to the same point in the right, the
            point of the left pair that is closest will be associated with the
            original point in the right, and other will be associated with the
            next closest. (not implemented yet)
        k : int
            The number of next closest points to calculate when no_repeat is
            set to True. If no_repeat is false, this value is 1.

        Returns
        -------
        df : pandas.core.frame.DataFrame | geopandas.geodataframe.GeoDataFrame
            A copy of the first data frame with the specified field and a
            distance column.
        """
        # We need geodataframes
        df1 = self._obj.copy()
        original_type = type(df1)
        if not isinstance(df1, self.gpd.geodataframe.GeoDataFrame):
            if "geometry" in df1:
                del df1["geometry"]
            df1 = df1.rr.to_geo(lat, lon)  # <--------------------------------- not necessary, could speed this up just finding the lat/lon columns, we'd also need to reproject for this to be most accurate.
        if not isinstance(df, self.gpd.geodataframe.GeoDataFrame):
            if "geometry" in df:
                del df["geometry"]
            df = df.rr.to_geo(lat, lon)
            df = df.reset_index(drop=True)

        # What from the second data frame do we want to return?
        if fields:
            if isinstance(fields, str):
                fields = [fields]
        else:
            fields = [c for c in df if c != "geometry"]

        # Get arrays of point coordinates
        crds1 = self.np.array(
            list(df1["geometry"].apply(lambda x: (x.x, x.y)))
        )
        crds2 = self.np.array(
            list(df["geometry"].apply(lambda x: (x.x, x.y)))
        )

        # Build the connections tree and query points from the first df
        tree = self.cKDTree(crds2)
        if no_repeat:
            dist, idx = tree.query(crds1, k=k)
            dist, idx = self._derepeat(dist, idx)
        else:
            dist, idx = tree.query(crds1, k=1)

        # We might be relacing a column
        for field in fields:
            if field in df1:
                del df1[field]

        # Rebuild the dataset
        dfa = df1.reset_index(drop=True)
        dfb = df.iloc[idx, :]
        del dfb["geometry"]
        dfb = dfb.reset_index(drop=True)
        df = pd.concat([dfa, dfb[fields], pd.Series(dist, name='dist')],
                       axis=1)

        # If this wasn't already a geopandas data frame reformat
        if not isinstance(df, original_type):
            del df["geometry"]
            df = pd.DataFrame(df)

        return df

    # def papply(self, func, **kwargs):
    #     """Apply a function to a dataframe in parallel chunks."""
    #     from pathos import multiprocessing as mp

    #     from itertools import product

    #     df = self._obj.copy()
    #     cdfs = np.array_split(df, os.cpu_count() - 1)
    #     pool = mp.Pool(mp.cpu_count() - 1)
    #     args = [(cdf, kwargs) for cdf in cdfs]
    #     out = pool.starmap(cfunc, args)

    def scatter(self, x="capacity", y="mean_lcoe", z=None, color="mean_lcoe",
                size=None):
        """Create a plotly scatterplot."""
        import plotly.express as px

        df = self._obj.copy()

        if z is None:
            fig = px.scatter(df, x, y, color=color, size=size)
        else:
            fig = px.scatter_3d(df, x, y, z, color=color, size=size)

        fig.show()

    def to_bbox(self, bbox):
        """Return points within a bounding box [xmin, ymin, xmax, ymax]."""
        df = self._obj.copy()
        df = df[(df["longitude"] >= bbox[0]) &
                (df["latitude"] >= bbox[1]) &
                (df["longitude"] <= bbox[2]) &
                (df["latitude"] <= bbox[3])]
        return df

    def to_geo(self, lat=None, lon=None):
        """Convert a Pandas data frame to a geopandas geodata frame."""
        # Let's not transform in place
        df = self._obj.copy()
        df.rr.decode()

        # Find coordinate columns
        if "geometry" not in df.columns:
            if "geom" not in df.columns:
                try:
                    lat, lon = self.find_coords()
                except ValueError:
                    pass

                # For a single row
                def to_point(x):
                    return self.Point(tuple(x))
                df["geometry"] = df[[lon, lat]].apply(to_point, axis=1)

        # Create the geodataframe - add in projections
        if "geometry" in df.columns:
            gdf = self.gpd.GeoDataFrame(df, crs='epsg:4326',
                                        geometry="geometry")
        if "geom" in df.columns:
            gdf = self.gpd.GeoDataFrame(df, crs='epsg:4326',
                                        geometry="geom")

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
        dtypes = self.np.dtype(struct_types)

        # The target empty array
        array = self.np.zeros(v.shape[0], dtypes)

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

    def _delist(self, value):
        """Extract the value of an object if it is a list with one value."""
        if isinstance(value, list):
            if len(value) == 1:
                value = value[0]
        return value

    # def _derepeat(dist, idx):
    #     """Find the next closest index for repeating cKDTree outputs."""  # <-- Rethink this, we could autmomatically set k tot he max repeats
    #     # Get repeated idx and counts
    #     k = idx.shape[1]
    #     uiidx, nrepeats = np.unique(idx, return_counts=True)
    #     max_repeats = nrepeats.max()
    #     if max_repeats > k:
    #         raise ValueError("There are a maximum of " + str(max_repeats) +
    #                          " repeating points, to use the next closest "
    #                          "neighbors to avoid repeats, set k to this "
    #                          "number.")

    #     for i in range(k - 1):
    #         # New arrays for this axis
    #         iidx = idx[:, i]
    #         idist = dist[:, i]

    def _destring(self, string):
        """Destring values into their literal python types if needed."""
        try:
            return json.loads(string)
        except (TypeError, self.JSONDecodeError):
            return string


class Reformatter:
    """Reformat reV inputs/outputs."""

    import multiprocessing as mp
    import os
    import subprocess as sp

    import gdalmethods as gm
    import geopandas as gpd
    import h5py
    import numpy as np
    import rasterio as rio

    from rasterio import features
    from revruns.constants import GDAL_TYPEMAP
    from tqdm import tqdm

    def __init__(self, data_path, template, target_dir=None,
                 raster_dir=None, shapefile_dir=None, warp_threads=1):
        """Initialize Reformatter object.

        Here, I'd like to implement GDAL methods rather than relying on
        in-memory python methods.

        Parameters
        ----------
        data_path : str
            Path to directory containing 'shapefile' and/or 'raster' folders
            containing files to be reformatting
        template : str
            Path to either a GeoTiff or HDF5 reV exclusion file to use as a
            template for reformatting target files. The HDF5 file requires
            a top level attribute containing a rasterio profile describing the
            arrays it contains.
        target_dir : str
            Target directory for output rasters. Will default to a folder
            named "exclusions" within the given data_path.
        raster_dir : str
            Path to folder containing rasters to reformat. Defaults to
            `data_path` argument.
        shapefile_dir : str
            Path to folder containing shapefiles to reformat. Defaults to
            `data_path` argument.
        warp_threads : int
            Number of threads to use for rasterio warp functions. Defaults to
            1.
        """
        self.dp = Data_Path(data_path)
        self.template = template
        if not raster_dir:
            raster_dir = data_path
        if not shapefile_dir:
            shapefile_dir = data_path
        if not target_dir:
            target_dir = data_path
        self.target_dir = Data_Path(target_dir)
        self.raster_dir = raster_dir
        self.shapefile_dir = shapefile_dir
        self.warp_threads = warp_threads
        self._preflight(target_dir)

    def __repr__(self):
        """Print Reformatter object attributes."""
        tmplt = "<rr.Reformatter Object: data_path={}, template={}>"
        msg = tmplt.format(self.dp.data_path, self.template)
        return msg

    def key(self, file):
        """Create a key from a file name."""
        fname = os.path.basename(file)
        key = os.path.splitext(fname)[0]
        return key

    def reformat_all(self):
        """Reformat all files."""
        print("Reformatting shapefiles...")
        self.reformat_shapefiles()

        print("Reformatting rasters...")
        self.reformat_rasters()

    def reformat_raster(self, file, overwrite=False):
        """Resample and reproject a raster."""
        # Create a target file path (should ask user though)
        dst = self.target_dir.join(os.path.basename(file), mkdir=True)

        # Read source file
        if os.path.exists(dst) and not overwrite:
            print(dst + " exists, skipping...")
            return

        # Open dataset and get source meta and array
        with self.rio.open(file) as src:
            array = src.read(1)
            profile = src.profile

        # Warp the array
        nx = self.meta["width"]
        ny = self.meta["height"]
        narray, _ = self.rio.warp.reproject(
                            source=array,
                            destination=self.np.empty((ny, nx)),
                            src_transform=profile["transform"],
                            src_crs=profile["crs"],
                            dst_transform=self.meta["transform"],
                            dst_crs=self.meta["crs"],
                            resampling=0,  # nearest needed for binary
                            num_threads=self.warp_threads
                        )

        # Write to filearray
        meta = self.meta.copy()
        dtype = narray.dtype
        meta["dtype"] = dtype
        with self.rio.open(dst, "w", **meta) as trgt:
            trgt.write(narray, 1)

    def reformat_rasters(self, overwrite=False):
        """Resample and reproject rasters."""
        files = self.rasters

        # ncpu = self.os.cpu_count()
        # with self.mp.Pool(ncpu) as pool:
        #     for _ in self.tqdm(pool.imap(self.reformat_raster, files),
        #                        total=len(files)):
        #         pass

        for file in self.tqdm(files):
            self.reformat_raster(file, overwrite=overwrite)

    def reformat_shapefile(self, file, dtype="byte", field_dict=None,
                           overwrite=False):
        """Reproject and rasterize a vector."""
        # Create dataset key and destination path
        key = self.key(file)
        dst = self.target_dir.join(key + ".tif")

        # Handle overwrite option
        if os.path.exists(dst) and not overwrite:
            return
        else:
            print(f"Processing {dst}...")

        # Read in shapefile shell for meta information
        gdf_meta = self.gpd.read_file(file, rows=1)

        # This requires a specific field to indicate which values to use
        if not field_dict:
            fields = [col for col in gdf_meta.columns if "raster" in col]
            
            if not any(fields) or len(fields) > 1:
                raise KeyError(f"{file} requires a single 'raster' field "
                               "or a dictionary with file name, field "
                               "name pairs.")
            else:
                field = fields[0]
        else:
            field = field_dict[os.path.basename(file)]

        # The values might be strings
        if isinstance(gdf_meta[field].iloc[0], str):
            gdf = self.gpd.read_file(file)
            gdf = gdf.sort_values(field).reset_index(drop=True)
            values = gdf[field].unique()
            string_values = {i + 1: v for i, v in enumerate(values)}
            map_values = {v: k for k, v in string_values.items()}
            gdf[field] = gdf[field].map(map_values)
            del gdf 
            self.string_values[key] = string_values
        else:
            self.string_values[key] = {}

        # Match CRS
        crs1 = gdf_meta.crs
        crs2 = self.meta["crs"]
        if not crs_match(crs1, crs2):
            print("Reprojecting...")
            file2 = file.replace(".gpkg", "2.gpkg")
            self.gm.reproject_polygon(src=file, dst=file2, t_srs=crs2)

            gdf = self.gpd.read_file(file2, rows=10)
            
            os.remove(file)
            shutil.move(file2, file)

        # Call GDAL
        try:
            self.gm.rasterize(src=file, dst=dst, template_path=self.template,
                              attribute=field, dtype=dtype, all_touch=True)
        except Exception:
            raise("Rasterization failed.")


    def reformat_shapefiles(self, field_dict=None, overwrite=False):
        """Reproject and rasterize vectors."""
        # Run in parallel
        # ncpu = self.os.cpu_count()
        # with self.mp.Pool(ncpu) as pool:
        #     for _ in self.tqdm(pool.imap(self.reformat_shapefile,
        #                                  self.shapefiles),
        #                        total=len(files)):
        #        pass

        # Run serially
        for file in self.shapefiles:
            self.reformat_shapefile(file,
                                    field_dict=field_dict,
                                    overwrite=overwrite)

        # Write updated string value dictionary
        with open(self.string_path, "w") as file:
            file.write(json.dumps(self.string_values, indent=4))

    @property
    def meta(self):
        """Return the meta information from the template file."""
        # Extract profile from HDF or raster file
        try:
            with self.h5py.File(self.template) as h5:
                meta = h5.attrs["profile"]
                if isinstance(meta, str):
                    meta = json.loads(meta)
        except OSError:
            with self.rio.open(self.template) as raster:
                meta = dict(raster.profile)

        # Make sure the file is tiled and compressed
        meta["blockxsize"] = 128
        meta["blockysize"] = 128
        meta["tiled"] = True
        meta["compress"] = "lzw"

        return meta

    @property
    def rasters(self):
        """Return list of all rasters in project rasters folder."""
        rasters = self.dp.contents(self.raster_dir, "*tif")
        rasters.sort()
        return rasters

    @property
    def shapefiles(self):
        """Return list of all shapefiles in project shapefiles folder."""
        shps = self.dp.contents(self.shapefile_dir, "*shp")
        gpkgs = self.dp.contents(self.shapefile_dir, "*gpkg")
        shapefiles = shps + gpkgs
        shapefiles.sort()
        return shapefiles

    def _preflight(self, target_dir):
        """Run preflight checks and setup."""
        # Check that the template file exists
        try:
            assert os.path.exists(self.template)
        except AssertionError:
            print(f"Warning: {self.template} does not exist.")

        # Create a dictionary for a value - string lookup
        self.string_path = self.dp.join("string_values.json")
        if os.path.exists(self.string_path):
            with open(self.string_path, "r") as file:
                self.string_values = json.load(file)
        else:
            self.string_values = {}


class Exclusions(Reformatter):
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


class Profiles:
    """Methods for manipulating generation profiles."""

    import json

    import h5py
    import numpy as np

    def __init__(self, gen_fpath):
        """Initialize Profiles object.

        Parameters
        ----------
        gen_fpath : str
            Path to a reV generation or representative profile file.
        """
        self.gen_fpath = gen_fpath

    def __repr__(self):
        """Return representation string for Profiles object."""
        return f"<Profiles: gen_fpath={self.gen_fpath}"

    def _best(self, row, gen_fpath, variable, lowest):
        """Find the best generation point in a supply curve table row.

        Parameters
        ----------
        row : pd.core.series.Series
            A row from a reV suuply curve table.
        ds : h5py._hl.files.File
            An open h5py file


        Returns
        -------
        int
            The index position of the best profile.
        """
        idx = self.json.loads(row["gen_gids"])  # Don't forget to add res_gid
        idx.sort()
        with self.h5py.File(gen_fpath) as ds:
            if lowest:
                gid = idx[self.np.argmin(ds[variable][idx])]
            else:
                gid = idx[self.np.argmax(ds[variable][idx])]
            value = ds[variable][gid]
        row["best_gen_gid"] = gid
        row[self._best_name(variable, lowest)] = value
        return row

    def _all(self, idx, gen_fpath, variable):
        """Find the best generation point in a supply curve table row.

        Parameters
        ----------
        row : pd.core.series.Series
            A row from a reV suuply curve table.
        ds : h5py._hl.files.File
            An open h5py file


        Returns
        -------
        int
            The index position of the best profile.
        """
        idx = json.loads(idx)
        idx.sort()
        with self.h5py.File(gen_fpath) as ds:
            values = ds[variable][idx]
        return values

    def _best_name(self, variable, lowest):
        """Return the column name of the best value column."""
        if lowest:
            name = f"{variable}_min"
        else:
            name = f"{variable}_max"
        return name

    def _derepeat(self, df):
        master_gids = []
        df["gen_gids"] = df["gen_gids"].apply(json.loads)
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            gids = row["gen_gids"]
            for g in gids:
                if g in master_gids:
                    break
                    gids.remove(g)
                else:
                    master_gids.append(g)

    def get_table(self, sc_fpath, variable, lowest):
        """Apply the _best function in parallel."""
        from pathos import multiprocessing as mp

        # Function to apply to each chunk defined below
        def cfunc(args):
            cdf, gen_fpath, variable, lowest = args
            out = cdf.apply(self._best, gen_fpath=gen_fpath, variable=variable,
                            lowest=lowest, axis=1)
            return out

        # The supply curve data frame
        df = pd.read_csv(sc_fpath)

        # Split the data frame up into chuncks
        ncpu = mp.cpu_count() - 1
        cdfs = np.array_split(df, ncpu)
        arg_list = [(cdf, self.gen_fpath, variable, lowest) for cdf in cdfs]

        # Apply the chunk function in parallel
        outs = []
        with mp.Pool(ncpu) as pool:
            for out in pool.imap(cfunc, arg_list):
                outs.append(out)

        # Concat the outputs back into a single dataframe and sort
        df = pd.concat(outs)
        df = df.sort_values("best_gen_gid")

        return df

    def main(self, sc_fpath, dst, variable="lcoe_fcr-means", lowest=True):
        """Write a dataset of the 'best' profiles within each sc point.

        Parameters
        ----------
        sc_fpath : str
            Path to a supply-curve output table.
        dst : str
            Path to the output HDF5 file.
        variable : str
            Variable to use as selection criteria.
        lowest : boolean
            Select based on the lowest criteria value.
        """
        # Read in the expanded gen value table
        df = pd.read_csv(sc_fpath)
        df["values"] = df["gen_gids"].apply(self._all,
                                            gen_fpath=self.gen_fpath,
                                            variable=variable)

        # Choose which gids to keep to avoid overlap
        tdf = df[["sc_point_gid", "gen_gids", "values"]]

        # Read in the preset table
        df = self.get_table(sc_fpath, variable=variable, lowest=lowest)

        # Subset for common set of fields
        # ...

        # Convert the supply curve table a structure, will use as meta
        sdf, dtypes = df.rr.to_sarray()

        # Create new dataset
        ods = h5py.File(self.gen_fpath, "r")
        nds = h5py.File(dst, "w")

        # One or multiple years?
        keys = [key for key in ods.keys() if "cf_profile" in key]

        # Build the array
        arrays = []
        gids = df["best_gen_gid"].values
        for key in keys:
            break
            gen_array = ods[key][:, gids]


class RRNrwal():
    """Helper functions for using NRWAL."""

    def __init__(self, readme="~/github/NRWAL/README.rst"):
        """Initialize RRNrwal object."""
        self.readme = readme

    @property
    def variables(self):
        """Build a nice variable dictionary."""
        # Read in text lines
        with open(os.path.expanduser(self.readme)) as file:
            lines = file.readlines()

        # Find the table part
        lines = lines[lines.index("    * - Variable Name\n"):]
        lines = lines[:lines.index("\n")]
        lines = [line.replace("\n", "") for line in lines]

        # Chunk on the *
        idxs = []
        for l in lines:
            if "*" in l:
                idx = [lines.index(l), lines.index(l) + 4]
                idxs.append(idx)

        # Pluck out the needed parts
        variables = {}
        for idx in idxs[1:]:
            chunk = lines[idx[0]: idx[1]]
            elements = []
            for c in chunk:
                c = c.replace("`", "")
                c = c[c.index("- ") + 2:]
                elements.append(c)
            name = elements[0]
            variables[name] = {}
            variables[name]["long_name"] = elements[1]
            variables[name]["source"] = elements[2]
            variables[name]["units"] = elements[3]

        return variables

    @property
    def definitions(self):
        """Print the definitions of NRWAL variables."""
        for var, elements in self.variables.items():
            print(var + ": " + elements["long_name"])
