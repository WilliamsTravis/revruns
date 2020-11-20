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


def get_sheet(file_name, sheet_name=None, starty=0, startx=0, header=True):
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
        table = file.parse(sheet_name=sheet_name)
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


def point_line(arg):
    """Find the closest transmission line to a point.

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


def write_config(config_dict, path):
    """Write a configuration dictionary to a json file."""
    with open(path, "w") as file:
        file.write(json.dumps(config_dict, indent=4))


class Data_Path:
    """Data_Path joins a root directory path to data file paths."""

    def __init__(self, data_path, mkdir=False):
        """Initialize Data_Path."""
        self.data_path = data_path
        self.last_path = os.getcwd()
        self._expand_check(mkdir)

    def __repr__(self):
        """Print the data path."""
        items = ["=".join([str(k), str(v)]) for k, v in self.__dict__.items()]
        arguments = " ".join(items)
        msg = "".join(["<Data_Path " + arguments + ">"])
        return msg

    def join(self, *args, mkdir=False):
        """Join a file path to the root directory path."""
        path = os.path.join(self.data_path, *args)
        if mkdir:
            os.makedirs(os.path.dirname(path), exist_ok=True)
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

    def _expand_check(self, mkdir):
        # Expand the user path if a tilda is present in the root folder path.
        if "~" in self.data_path:
            self.data_path = os.path.expanduser(self.data_path)

        # Check if data path exists
        if not os.path.exists(self.data_path):
            if mkdir:
                os.makedirs(self.data_path, exist_ok=True)
            else:
                print("Warning: " + self.data_path + " does not exist.")


class Exclusions:
    """Build or add to an HDF5 Exclusions dataset."""

    import h5py
    import numpy as np
    import rasterio as rio

    from rasterio.errors import RasterioIOError

    def __init__(self, excl_fpath):
        """Initialize Exclusions object."""
        self.excl_fpath = excl_fpath
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
        except:
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
                    keys.pop(dname)

            if dname not in keys:
                hdf.create_dataset(name=dname, data=array, dtype=dtype,
                                   chunks=(1, 128, 128))
                hdf[dname].attrs["file"] = os.path.abspath(file)
                hdf[dname].attrs["profile"] = profile
                if description:
                    hdf[dname].attrs["description"] = description

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
            except:
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
        with self.h5py.File(self.excl_fpath, "r") as hdf:

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

    def _get_coords(self, profile):
        # Get x and y coordinates (One day we'll have one transform order!)
        geom = profile["transform"]
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

    def _set_coords(self, profile):
        # Add the lat and lon meshgrids if they aren't already present
        with self.h5py.File(self.excl_fpath, "r+") as hdf:
            keys = list(hdf.keys())
            if "latitude" not in keys or "longitude" not in keys:
                xs, ys = self._get_coords(profile)
                xgrid, ygrid = self.np.meshgrid(xs, ys)
                hdf.create_dataset(name="longitude", data=xgrid)
                hdf.create_dataset(name="latitude", data=ygrid)

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


@pd.api.extensions.register_dataframe_accessor("rr")
class PandasExtension:
    """Making dealing with reV output objects easier."""

    from json import JSONDecodeError

    import geopandas as gpd
    import numpy as np

    from scipy.spatial import cKDTree
    from shapely.geometry import Point

    def __init__(self, pandas_obj):
        """Initialize pandas object."""
        import geopandas as gpd

        if type(pandas_obj) != pd.core.frame.DataFrame:
            if type(pandas_obj) != gpd.geodataframe.GeoDataFrame:
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

    def decode(self):
        """Decode the columns of a meta data object from a reV output."""
        import ast
        for c in self._obj.columns:
            x = self._obj[c].iloc[0]
            if isinstance(x, bytes):
                try:
                    self._obj[c] = self._obj[c].apply(lambda x: x.decode())
                except:
                    self._obj[c] = None
                    print("Column " + c + " could not be decoded.")
            elif isinstance(x, str):
                try:
                    if isinstance(ast.literal_eval(x), bytes):
                        try:
                            self._obj[c] = self._obj[c].apply(
                                lambda x: ast.literal_eval(x).decode()
                                )
                        except:
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
            for dists in tqdm(pool.imap(point_line, args),
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
