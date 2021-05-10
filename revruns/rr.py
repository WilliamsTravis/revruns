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

from glob import glob

import pandas as pd

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 20)
pd.options.mode.chained_assignment = None


def crs_match(crs1, crs2):
    """Check if two coordinate reference systems match."""
    from pyproj import CRS

    try:
        assert CRS(crs1) == CRS(crs2)
        return True
    except AssertionError:
        return False


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

    def __init__(self, data_path=".", mkdir=False):
        """Initialize Data_Path."""
        data_path = os.path.abspath(os.path.expanduser(data_path))
        self.data_path = data_path
        self.last_path = os.getcwd()
        self._exist_check(data_path, mkdir)
        self._expand_check()

    def __repr__(self):
        """Print the data path."""
        items = ["=".join([str(k), str(v)]) for k, v in self.__dict__.items()]
        arguments = " ".join(items)
        msg = "".join(["<Data_Path " + arguments + ">"])
        return msg

    def join(self, *args, mkdir=False):
        """Join a file path to the root directory path."""
        path = os.path.join(self.data_path, *args)
        self._exist_check(path, mkdir)
        return path

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

    @property
    def home(self):
        """Change directories to the data path."""
        self.last_path = os.getcwd()
        os.chdir(self.data_path)
        print(self.data_path)

    @property
    def back(self):
        """Change directory back to last working directory if home was used."""
        os.chdir(self.last_path)
        print(self.last_path)

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
                    print("Warning: " + directory + " did not exist, creating "
                          "directory.")
                    os.makedirs(directory, exist_ok=True)
                else:
                    print("Warning: " + directory + " does not exist.")

    def _expand_check(self):
        # Expand the user path if a tilda is present in the root folder path.
        if "~" in self.data_path:
            self.data_path = os.path.expanduser(self.data_path)


@pd.api.extensions.register_dataframe_accessor("rr")
class PandasExtension:
    """Accessing useful pandas functions directly from a data frame object."""

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
            df1 = df1.rr.to_geo(lat, lon)  # <--------------------------------- not necessary, could speed this up just finding the lat/lon columns, we'd also need to reproject for this to be most accurate.
        if not isinstance(df, self.gpd.geodataframe.GeoDataFrame):
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
    """Reformat raster or shapefile files into a reV-shaped raster."""

    import multiprocessing as mp
    import os
    import subprocess as sp

    import geopandas as gpd
    import h5py
    import numpy as np
    import rasterio as rio

    from rasterio import features
    from revruns.constants import GDAL_TYPEMAP
    from tqdm import tqdm

    def __init__(self, data_path, template, target_dir=None,
                 raster_dir="rasters", shapefile_dir="shapefiles",
                 warp_threads=1):
        """Initialize Reformatter object.

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
            Path to folder containing rasters to reformat (relative to
            data_path). Defaults to "rasters".
        shapefile_dir : str
            Path to folder containing shapefiles to reformat (relative to
            data_path). Defaults to "shapefiles".
        warp_threads : int
            Number of threads to use for rasterio warp functions. Defaults to
            1.
        """
        self.dp = Data_Path(data_path)
        self.template = template
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

        # Write to file
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

    def reformat_shapefile(self, file, field_dict=None, overwrite=False):
        """Reproject and rasterize a vector."""
        # Create dataset key and destination path
        key = self.key(file)
        dst = self.target_dir.join(key + ".tif")

        # Handle overwrite option
        if os.path.exists(dst) and not overwrite:
            return
        else:
            print(f"Processing {dst}...")

        # Read in shapefile and meta data
        gdf = self.gpd.read_file(file)

        # This requires a specific field to indicate which values to use
        if not field_dict:
            if "raster_value" not in gdf.columns:
                raise KeyError(f"{file} requires a 'raster_value' field "
                               "or a dictionary with file name, field "
                               "name pairs.")
            else:
                field = "raster_value"
        else:
            field = field_dict[os.path.basename(file)]

        # The values might be strings
        if isinstance(gdf[field].iloc[0], str):
            gdf = gdf.sort_values(field).reset_index(drop=True)
            values = gdf[field].unique()
            string_values = {i + 1: v for i, v in enumerate(values)}
            map_values = {v: k for k, v in string_values.items()}
            gdf[field] = gdf[field].map(map_values)
            self.string_values[key] = string_values
        else:
            self.string_values[key] = {}

        # Match CRS
        crs = self.meta["crs"]
        if not crs_match(crs, self.meta["crs"]):
            gdf = gdf.to_crs(crs)

        # Get shapes and values and rasterize
        gdf = gdf[["geometry", field]]
        shapes = [(geom, value) for geom, value in gdf.values]
        with self.rio.Env():
            array = self.features.rasterize(
                        shapes=shapes,
                        out_shape=(self.meta["height"],
                                   self.meta["width"]),
                        transform=self.meta["transform"],
                        all_touched=True
                    )

        # We can't have float16 (maybe others)
        # array = self._recast(array)

        # Write to file
        dtype = str(array.dtype)
        meta = self.meta.copy()
        meta["dtype"] = dtype
        with self.rio.Env():
            with self.rio.open(dst, "w", **meta) as file:
                file.write(array, 1)

    def reformat_shapefiles(self, field_dict=None, overwrite=False):
        """Reproject and rasterize vectors."""
        files = self.shapefiles

        # Run in parallel
        # ncpu = self.os.cpu_count()
        # with self.mp.Pool(ncpu) as pool:
        #     for _ in self.tqdm(pool.imap(self.reformat_shapefile, files),
        #                        total=len(files)):
        #        pass

        # Run serially
        for file in files:
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

        # Create target directory.
        if not target_dir:
            target_dir = self.dp.join("exclusions", mkdir=True)
        self.target_dir = Data_Path(target_dir, mkdir=True)

        # Create a dictionary for a value - string lookup
        self.string_path = self.dp.join("string_values.json")
        if os.path.exists(self.string_path):
            with open(self.string_path, "r") as file:
                self.string_values = json.load(file)
        else:
            self.string_values = {}

    def _recast(self, array):
        """Recast an array to an acceptable GDAL data type."""
        dtype = array.dtype
        if dtype == "float16": # <--------------------------------------------- Are there other data types it can't handle?
            dtype = "float32"  # <--------------------------------------------- How to choose?
            array = array.astype(dtype)
        return array, dtype


class Exclusions(Reformatter):
    """Build or add to an HDF5 Exclusions dataset."""

    import h5py
    import numpy as np
    import rasterio as rio

    from pyproj import Transformer
    from rasterio.errors import RasterioIOError

    def __init__(self, excl_fpath, lookup_path=None):
        """Initialize Exclusions object.

        Parameters
        ----------
            excl_fpath : str
                Path to target HDF5 reV exclusion file.
            lookup_path : str
                Path to json file containing a lookup dictionary for raster
                values derived from shapefiles containing string values
                (optional).
        """
        self.excl_fpath = excl_fpath
        self.lookup_path = lookup_path
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
        profile = json.dumps(dict(profile))

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
                hdf[dname].attrs["profile"] = profile
                if description:
                    hdf[dname].attrs["description"] = description
                if dname in self.lookup:
                    lookup = json.dumps(self.lookup[dname])
                    hdf[dname].attrs["string_values"] = lookup

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
                                 max_workers=None, distance_upper_bound=None,
                                 map_chunk=2560, save_flag=save_flag)
        return arrays

    @property
    def lookup(self):
        "Return dictionary with raster, string value pairs for reference."""
        if self.lookup_path:
            with open(self.lookup_path, "r") as file:
                lookup = json.load(file)
        else:
            lookup = {}
        return lookup

    def _check_dims(self, raster, profile, dname):
        # Check new layers against the first added raster
        with self.h5py.File(self.excl_fpath, "r") as hdf:

            # Find the exisitng profile
            old = json.loads(hdf.attrs["profile"])
            new = json.loads(profile)

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
        transformer = self.Transformer.from_crs(self.profile["crs"],
                                                "epsg:4326", always_xy=True)
        lons, lats = transformer.transform(xs, ys)
        return lons, lats

    def _get_coords(self, profile):
        # Get x and y coordinates (One day we'll have one transform order!)
        profile = json.loads(profile)
        geom = profile["transform"]  # Ensure its in the right order
        xres = geom[0]
        ulx = geom[2]
        yres = geom[4]
        uly = geom[5]

        # Not doing rotations here
        xs = [ulx + col * xres for col in range(profile["width"])]
        ys = [uly + row * yres for row in range(profile["height"])]

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
                xs, ys = self._get_coords(profile)

                # Convert to geographic coordinates
                lons, lats = self._convert_coords(xs, ys)

                # Create grid and upload
                longrid, latgrid = self.np.meshgrid(lons, lats)
                hdf.create_dataset(name="longitude", data=longrid)
                hdf.create_dataset(name="latitude", data=latgrid)
