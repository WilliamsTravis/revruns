# -*- coding: utf-8 -*-
"""
Functions for revruns/brainstorming intuitive and easy config building

Created on Tue Nov 12 13:15:55 2019

@author: twillia2
"""
import json
import os
import ssl
import datetime as dt
import geopandas as gpd
import h5py
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm
from reV.utilities.exceptions import JSONError
 
# Fix remote file transfer issues with ssl (for gpd, find a better way).
ssl._create_default_https_context = ssl._create_unverified_context

# Times New Roman for plots?
plt.rcParams['font.family'] = 'Times New Roman'

# Package data path.
ROOT = os.path.abspath(os.path.dirname(__file__))
def data_path(path):
    """Path to local package data directory"""
    return os.path.join(ROOT, 'data', path)

# Time check.
NOW = dt.datetime.today().strftime("%Y-%m-%d %I:%M %p")

# For CONUS filtering.
CONUS = ['AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'IA',
         'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN',
         'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY',
         'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA',
         'VT', 'WA', 'WI', 'WV', 'WY']

# For checking if a requested output requires economic treatment.
ECON_MODULES = ["lcoe_fcr"]

# Checks for reasonable model output value ranges. No scaling factors here.
VARIABLE_CHECKS = {  
        "poa": (0, 1000),  # 1,000 MW m-2
        "cf_mean": (0, 240),  # 24 %
        "cf_profile": (0, 990)  # 99 %
        }

########## Construction Zone ###########
class Check_Variables():
    def __init__(self, files):
        self.files = files
        self._read_files()

    def checkvars(self):
        """Check a set of hdf5 model output files for potential anomalies.
        
        files = glob('/Users/twillia2/github/data/revruns/run_1/*')
        """    
        # For each data set in each file, check the values
        flagged = {}
        for hdf in tqdm(self.hdfs, position=0):
            # Get the list of sub data sets in each file
            subds = hdf.GetSubDatasets()
            
            # Will meta and time index mess this up?
            # subds.remove("meta")
            # subds.remove("time_index")

            # For each of these sub data sets, get an info dictionary
            for sub in subds:      
                info_str = gdal.Info(sub[0], options=["-stats", "-json"])      
                info = json.loads(info_str)
                filename = info["files"][0]
                desc = info["description"]
                var = desc[desc.index("//") + 2: ]
                max_data = info["bands"][0]["maximum"]
                min_data = info["bands"][0]["minimum"]
                max_threshold = VARIABLE_CHECKS[var][1]
                min_threshold = VARIABLE_CHECKS[var][0]

                # Check the thresholds. Could add more for mean and stdDev. 
                if max_data > max_threshold:
                    filename = os.path.basename(filename)
                    flag = ":".join([filename, var])
                    message = (" - maximum value is greater than " +
                               max_threshold)
                    flagged[flag] = message
                if min_data < min_threshold:
                    filename = os.path.basename(filename)
                    flag = ":".join([filename, var])
                    message = (" - minimum value is less than " +
                               min_threshold)
                    flagged[flag] = message

        # Return the dictionary of messages
        return flagged

    def _read_files(self):
        """Check and read all hdf files."""
        try:
            hdfs = [gdal.Open(file) for file in self.files]
        except OSError:
            print("Could not read all files, are these hdf5 formats?")
        self.hdfs = hdfs
########################################

# Resource data set dimensions. Just the number of grid points for the moment.
RESOURCE_DIMS = {
        "nsrdb_v3": 2018392,
        "wind_conus_v1": 2488136,
        "wind_canada_v1": 2894781,
        "wind_canada_v1bc": 2894781,
        "wind_mexico_v1": 1736130,
        "wind_conus_v1_1": 2488136,
        "wind_canada_v1_1": 289478,
        "wind_canada_v1_1bc": 289478,
        "wind_mexico_v1_1": 1736130
        }

# The Eagle HPC path to each resource data set. Brackets indicate years.
RESOURCE_DATASETS = {
        "nsrdb_v3": "/datasets/NSRDB/v3.0.1/nsrdb_{}.h5",
        "wind_conus_v1": "/datasets/WIND/conus/v1.0.0/wtk_conus_{}.h5",
        "wind_canada_v1": "/datasets/WIND/canada/v1.0.0/wtk_canada_{}.h5",
        "wind_canada_v1bc": "/datasets/WIND/canada/v1.0.0bc/wtk_canada_{}.h5",
        "wind_mexico_v1": "/datasets/WIND/mexico/v1.0.0/wtk_mexico_{}.h5",
        "wind_conus_v1_1": "/datasets/WIND/conus/v1.1.0/wtk_conus_{}.h5",
        "wind_canada_v1_1": "/datasets/WIND/canada/v1.1.0/wtk_canada_{}.h5",
        "wind_canada_v1_1bc": ("/datasets/WIND/canada/v1.1.0bc/" +
                               "wtk_canada_{}.h5"),
        "wind_mexico_v1_1": "/datasets/WIND/mexico/v1.0.0/wtk_mexico_{}.h5"
        }

# The title of each resource data set.
RESOURCE_LABELS = {
        "nsrdb_v3": "National Solar Radiation Database - v3.0.1",
        "wind_conus_v1": ("Wind Integration National Dataset (WIND) " +
                          "Toolkit - CONUS, v1.0.0"),
        "wind_canada_v1": ("Wind Integration National Dataset (WIND) " +
                           "Toolkit - Canada, v1.0.0"),
        "wind_canada_v1bc": ("Wind Integration National Dataset (WIND) " +
                             "Toolkit - Canada, v1.1.0"),
        "wind_mexico_v1": ("Wind Integration National Dataset (WIND) " +
                           "Toolkit - Mexico, v1.0.0"),
        "wind_conus_v1_1":("Wind Integration National Dataset (WIND) " +
                           "Toolkit - CONUS, v1.1.0"),
        "wind_canada_v1_1": ("Wind Integration National Dataset (WIND) " +
                             "Toolkit - Canada, v1.1.0"),
        "wind_canada_v1_1bc": ("Wind Integration National Dataset (WIND) " +
                               "Toolkit - Canada, v1.1.0bc"),
        "wind_mexico_v1_1": ("Wind Integration National Dataset (WIND) " +
                             "Toolkit - Mexico, v1.0.0"),
        }

# Target geographic coordinate system identifiers.  # <------------------------- Check this
TARGET_CRS = ["+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs ",
              {'init': 'epsg:4326'},
              {'type': 'EPSG', 'properties': {'code': 4326}}]

# Default SAM model parameters for pvwattsv5.
SOLAR_SAM_PARAMS = {"azimuth": 180,
                    "array_type": 0,
                    "compute_module": "pvwattsv5",
                    "dc_ac_ratio": 1.1,
                    "gcr": 0.4,
                    "inv_eff": 96,
                    "losses": 14.0757,
                    "module_type": 0,
                    "system_capacity": 5,
                    "tilt": 20}

# Default Wind Turbine Powercurve Powerout (until a better way shows up).
DEFAULT_WTPO = np.zeros(161)
DEFAULT_WTPO[38: 100] = 4500.0
DEFAULT_WTPO[13: 38] = [122.675, 169.234, 222.943, 284.313, 353.853, 432.076,
                        519.492, 616.610, 723.943, 842.001, 971.294, 1112.330,
                        1265.630, 1431.690, 1611.040, 1804.170, 2011.600,
                        2233.840, 2471.400, 2724.790, 2994.530, 3281.120,
                        3585.070, 3906.900, 4247.120]
DEFAULT_WTPO = list(DEFAULT_WTPO)

# Default SAM model parameters for wind.  # <---------------------------------- Check that these are indeed the defaults.
WIND_SAM_PARAMS = {
        "adjust:constant": 0.0,
	    "en_low_temp_cutoff": "placeholder",
        "low_temp_cutoff": -10,
        "en_icing_cutoff": "placeholder",
        "icing_cutoff_temp": 0.0,
        "icing_cutoff_rh": 95.0,
        "system_capacity": 200000,
        "wind_farm_losses_percent": 12.8,
        "wind_farm_wake_model": 0,
        "wind_farm_xCoordinates": None, # <------------------------------------ Don't we set these in the points file?
        "wind_farm_yCoordinates": None,
        "wind_resource_model_choice": 0,
        "wind_resource_shear": 0.140,
        "wind_resource_turbulence_coeff": 0.10,
        "wind_turbine_cutin": 0.0,
        "wind_turbine_hub_ht": 100.0,
        "wind_turbine_powercurve_powerout": DEFAULT_WTPO,
        "wind_turbine_powercurve_windspeeds": list(np.arange(0, 40.25, 0.25)),
        "wind_turbine_rotor_diameter": 167.0,
        "capital_cost" : 245000000,
        "fixed_operating_cost" : 7790000,
        "fixed_charge_rate": 0.052,
        "variable_operating_cost": 0
}

# All default SAM model parameters.  # <-------------------------------------------------- Add as more models are discovered
SAM_PARAMS = {"pv": SOLAR_SAM_PARAMS,
              "wind": WIND_SAM_PARAMS}

# Default 'Top Level' parameters, i.e. those that are shared between runs.
TOP_PARAMS = {"allocation": "rev",
              "feature": "--qos=normal",
              "logdir": "./logs",
              "loglevel": "INFO",
              "memory": 90,
              "memory_utilization_limit": 0.4,
              "nodes": 1,
              "option": "eagle",
              "outdir": "./",
              "outputs": "cf_mean",
              "parallel": False,
              "pointdir": "./project_points",
              "resource": "nsrdb_v3",
              "sites_per_worker": 100,
              "tech": "pv",
              "walltime": 0.5,
              "years": "all"}


# Functions.
def box_points(bbox, crd_path=data_path("nsrdb_v3_coords.csv"), gridids=True):
    """Filter grid ids by geographical bounding box

    Parameters:
        bbox (list): A list containing the geographic coordinates of the
                     desired bounding box in this order:
                         [min lon, min lat, max lon, max lat]
        crd_path (str): The local path to the desired resource coordinate
                        list.

    Returns:
        gids (list): A list of grid IDs.
    """
    # Resource coordinate data frame from get_coordinates
    grid = pd.read_csv(crd_path)

    # Filter the data frame for points within the bounding box
    crds = grid[(grid["lon"] > bbox[0]) &
                (grid["lat"] > bbox[1]) &
                (grid["lon"] < bbox[2]) &
                (grid["lat"] < bbox[3])]

    # Just gridids or coordinates?
    if gridids:
        points = crds.index.to_list()
    else:
        points = crds

    return points


def check_config(config_file):
    """Check that a json file loads without error.

    Try loading with reV.utilities.safe_json_load!
    """
    try:
        with open(config_file, "r") as file:
            json.load(file)
    except json.decoder.JSONDecodeError as error:
        msg = ('JSON Error:\n{}\nCannot read json file: '
               '"{}"'.format(error, config_file))
        raise JSONError(msg)


def compare_profiles(datasets,
                     dataset="cf_profile",
                     units="$\mathregular{m^{-2}}$",
                     title="Output Profiles",
                     cmap="viridis",
                     savefolder=None,
                     dpi=300):
    """Compare profiles from different reV generation models

    Parameters:
        outputs (list): A list of reV output profile numpy arrays.

    Returns:
        (png): An image comparing output profiles over time.
    """
    # Get the profiles from the datasets
    profiles = {key: datasets[key][dataset] for key in datasets}

    # Get the time values from one of the data sets
    keys = list(datasets.keys())
    time = datasets[keys[0]]["time_index"]
    nstep = int(len(time) / 15)
    time_ticks = np.arange(0, len(time), nstep)
    time_labels = pd.to_datetime(time[1::nstep]).strftime("%b %d %H:%M")

    # Get grouping features
    groups = list(profiles.keys())
    for i, grp in enumerate(groups):
        elements = grp.split("_")
        module = elements[0].upper()
        group_feature = elements[1].capitalize()
        year = elements[2]
        elements = [module, group_feature, year]
        group = " ".join(elements)
        groups[i] = group

    # Get the datasets
    outputs = [profiles[key] for key in profiles.keys()]

    # Get some information about the outputs
    noutputs = len(outputs)

    # Transpose outputs so they're horizontal
    outputs = [out.T for out in outputs]

    # Figure level graph elements
    fig, axes = plt.subplots(noutputs, 1, figsize=(20, 4))
    fig.suptitle(title, y=1.15, x=.425, fontsize=20)
    fig.tight_layout()
    fig.text(0.425, 0.001, 'Date', ha='center', va='center', fontsize=15)
    fig.text(0.00, .6, 'Site #', ha='center', va='center', fontsize=15,
             rotation='vertical')

    # We need a common color map
    maxes = [np.max(out) for out in outputs]
    if np.diff(maxes) == 0:
        maxes[0] = maxes[0] - 1
    color_template = outputs[int(np.where(maxes == max(maxes))[0])]
    ctim = axes[0].imshow(color_template, cmap=cmap)
    clim = ctim.properties()['clim']

    # For each axis plot and format
    for i, axis in enumerate(axes):
        axis.imshow(outputs[i], cmap=cmap, clim=clim)
        axis.set_aspect('auto')
        axis.set_title(groups[i], fontsize=15)
        axis.set_xticks([])

    # Set date axis on the last one?
    axes[i].set_xticks(time_ticks)
    axes[i].set_xticklabels(time_labels)
    fig.autofmt_xdate(rotation=-35, ha="left")

    # Set the colorbar
    cbr = fig.colorbar(ctim, ax=axes.ravel().tolist(), shrink=.9,
                       pad=0.02)
    cbr.ax.set_ylabel(units, fontsize=15, rotation=270, labelpad=15)

    # Also save to file
    if savefolder:
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
        file = "_".join([module.lower(), dataset, year]) + ".png"
        path = os.path.join(savefolder, file)
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        plt.close(fig)


def extract_arrays(file):
    """Get all output data sets from an HDF5 file.

    Parameters:
        files (list): An HDF file path

    Output :
        (list): A dictionary of data sets as numpy arrays.
    """
    # Open file
    pointer = h5py.File(file, mode="r")

    # Get keys?
    keys = pointer.keys()

    # Get the three data sets
    data_sets = {key: pointer[key][:] for key in keys}

    # Decode the time index
    time = [t.decode("UTF-8") for t in data_sets["time_index"]]
    data_sets["time_index"] = time

    return data_sets


def get_coordinates(file, savepath):
    """Get all of the coordintes and their grid ids from an hdf5 file"""
    # Get numpy array of coordinates
    with h5py.File(file, mode="r") as pointer:
        crds = pointer["coordinates"][:]

    # Create a data frame and save it
    lats = crds[:, 0]
    lons = crds[:, 1]
    data = pd.DataFrame({"lat": lats, "lon": lons})
    data.to_csv(savepath, index=False)


def project_points(tag, resource="nsrdb_v3", points=1000):
    """Generates a required point file for spatial querying in reV.

    Parameters:
        jobname (str): Job name assigned to SAM configuration.
        resource (str): Energy resource data set key. Set to None for options.
        points (int | str | list): Sample points to generate. Set to an
                                   integer, n, to use the first n grid IDs,
                                   set to a list of points to use those points,
                                   or set to the string "all" to use all
                                   available points in the chosen resource
                                   data set.
        coords (list): A list of geographic coordinates to be converted to grid
                       IDs. (not yet implemented, but leaving this reminder)

    Returns:
        pandas.core.frame.DataFrame: A data frame of grid IDs and SAM config
                                     keys (job name).
    """
    # Create a project_points folder if it doesn't exist
    if not os.path.exists("project_points"):
        os.mkdir("project_points")

    # Print out options
    if not resource:
        print("Available resource datasets: ")
        for key, var in RESOURCE_LABELS.items():
            print("   '" + key + "': " + str(var))
        raise ValueError("'resource' argument required. Please provide the " +
                         "key to one of the above options as the value for " + 
                         "config.top_params['resource'].")

    # Get the coordinates for the resource data set.
    point_path = resource + "_coords.csv"
    try:
        coords = pd.read_csv(data_path(point_path))
    except:
        raise ValueError("Sorry, working on this. Please use the CLI " + 
                         "'rrpoints' on " + 
                         RESOURCE_DATASETS[resource].format(2018) + 
                         " (or any other year) and save the output file " +
                         "to " + data_path(point_path) + ".")

    # Sample or full grid?
    if isinstance(points, int):
        gridids = np.arange(0, points)
    elif points == "all":      
        gridids = np.arange(0, RESOURCE_DIMS[resource])
    else:
        gridids = points

    # Create data frame and join coordinates
    point_df = pd.DataFrame({"gid": gridids, "config": tag})
    point_df = point_df.join(coords)

    # Return data frame
    return point_df


def shape_points(shp, crd_path=data_path("nsrdb_v3_coords.csv")):
    """Find the grid ids for a specified resource grid within a shapefile

    Parameters:
        shp_path (str): A local path to a shape file or remote url to a zipped
                        folder containing a shapefile.
        crd_path (str): The local path to the desired resource coordinate
                        list.

    Returns:
        gids (list): A list of grid IDs.


    Notes:
        This could be done much faster.
    """
    # Read in shapefile with geopandas - remote urls allowed
    if isinstance(shp, str):
        shp = gpd.read_file(shp)

    # Check that the shapefile isn't projected, or else reproject it
    shp_crs = shp.crs
    if shp_crs not in TARGET_CRS:
        shp = shp.to_crs({'init': 'epsg:4326'})

    # The resource data sets are large, subset by bounding box first
    bbox = shp.geometry.total_bounds
    grid = box_points(bbox, crd_path, gridids=False)

    # Are there too many points to make a spatial object?
    gdf = to_geo(grid)

    # Use sjoin and filter out empty results
    points = gpd.sjoin(gdf, shp, how="left")
    points = points[~pd.isna(points["index_right"])]
    gids = list(points.index)

    return gids


def show_colorbars():
    """Shows all available color bar keys"""
    cmaps = [('Perceptually Uniform Sequential', [
                 'viridis', 'plasma', 'inferno', 'magma', 'cividis']),
             ('Sequential', [
                 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
             ('Sequential (2)', [
                 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
                 'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
                 'hot', 'afmhot', 'gist_heat', 'copper']),
             ('Diverging', [
                 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
             ('Cyclic', ['twilight', 'twilight_shifted', 'hsv']),
             ('Qualitative', [
                 'Pastel1', 'Pastel2', 'Paired', 'Accent',
                 'Dark2', 'Set1', 'Set2', 'Set3',
                 'tab10', 'tab20', 'tab20b', 'tab20c']),
             ('Miscellaneous', [
                 'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
                 'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix',
                 'brg', 'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral',
                 'gist_ncar'])]

    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    def plot_color_gradients(cmap_category, cmap_list):
        # Create figure and adjust figure height to number of colormaps
        nrows = len(cmap_list)
        figh = 0.35 + 0.15 + (nrows + (nrows-1)*0.1)*0.22
        fig, axes = plt.subplots(nrows=nrows, figsize=(6.4, figh))
        fig.subplots_adjust(top=1-.35/figh, bottom=.15/figh, left=0.2,
                            right=0.99)

        axes[0].set_title(cmap_category + ' colormaps', fontsize=14)

        for axis, name in zip(axes, cmap_list):
            axis.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
            axis.text(-.01, .5, name, va='center', ha='right', fontsize=10,
                      transform=axis.transAxes)

        # Turn off *all* ticks & spines, not just the ones with colormaps.
        for axis in axes:
            axis.set_axis_off()


    for cmap_category, cmap_list in cmaps:
        plot_color_gradients(cmap_category, cmap_list)

    plt.show()


def single_points(lats, lons, crd_path=data_path("nsrdb_v3_coords.csv")):
    """Take a data frame of single lat/lon coordinates and convert to a list of
    grid ids.

    Parameters:
        lats (list): A list containing the latitude coordinates of the desired
                    points.
        lons (list): A list containing the longitude coordinates of the desired
                    points.
        crd_path (str): The local path to the desired resource coordinate
                        list.

    Returns:
        gids (list): A list of grid IDs.
    """
    # Create a second data frame with the new coordinates
    points = pd.DataFrame({"lat": lats, "lon": lons})
    points = to_geo(points)

    # Filter the grid by the bounding box of the points
    bbox = [min(lons), min(lats), max(lons), max(lats)]
    grid = box_points(bbox, crd_path, gridids=False)
    grid = to_geo(grid)

    # Find which grid points are closest to each target points
    # ...


def to_geo(df, lat="lat", lon="lon"):
    """ Convert a Pandas data frame to a geopandas geodata frame """
    df["geometry"] = df[[lon, lat]].apply(lambda x: Point(tuple(x)), axis=1)
    gdf = gpd.GeoDataFrame(df, crs={'init': 'epsg:4326'},
                           geometry=df["geometry"])
    return gdf


# Classes.
class Config:
    """Sets reV model key values and generates configuration json files."""
    def __init__(self,
                 technology="pv",
                 top_params=TOP_PARAMS.copy(),
                 verbose=True):
        self.points = "all"
        self.top_params = top_params
        self.sam_params = SAM_PARAMS.copy()[technology]
        self.sam_files = {}
        self.verbose = verbose
        self._set_years()
        self._set_points_path()

    def config_all(self, excl_pos_lon=False):
        """ Call all needed sub configurations except for sam

        Parameter:
            excl_pos_lon (boolean): Exclude (potentially problematic) positive
                                    longitudes. This is temporarily here until
                                    a better way is found, since it is the
                                    easiest place to access.
        Returns:
            JSON cofiguration files for reV.
        """

        # Check that there are specified sam files
        try:
            assert len(self.sam_files) > 0
        except AssertionError:
            print("Could not configure GENRATION file, no SAM configuration " +
                  "files detected\n")
            raise

        # Separate parameters for space
        params = self.top_params

        # Create project points
        point_df = project_points(tag=params["set_tag"],
                                  resource=params["resource"],
                                  points=self.points)

        # If we are excluding positive longitudes
        if excl_pos_lon:
            point_df = point_df[point_df["lon"] < 0]

        point_df.to_csv(self.points_path, index=False)
        if self.verbose:
             print("POINTS" + " saved to '" + self.points_path + "'.")

        # If we are using more than one node, collect the outputs
        if params["nodes"] > 1:
            self._config_collect()

        # If we are modeling economic modules, use pipeline and econ
        outputs = self.top_params["outputs"]
        econ_outputs = [o in ECON_MODULES for o in outputs]
        if any(econ_outputs):
            self._config_econ()
            self._config_pipeline()

        # If more than one jobs are needed, use batch and pipeline
        if len(self.sam_files) > 1:
            self._config_batch()
            self._config_pipeline()

        # Configure the generation file
        self._config_gen()

    def config_sam(self, jobname="job"):
        """Configure the System Advisor Model (SAM) portion of a reV model.

        Parameters:
            jobname (str): Job name assigned to SAM configuration.
            resource (str): Energy resource data set key. Set to None for
                            options.
            points (int | str | list): Sample points to generate. Set to an
                                       integer, n, to use the first n grid IDs,
                                       set to a list of points to use those
                                       points, or set to the string "all" to
                                       use all available points in the chosen
                                       resource data set.

        Returns:
            dict: A dictionary of default and user specified SAM parameters.
            file: A local json file
        """
        # Make sure there is a sam config folder
        if not os.path.exists("./sam_configs"):
            os.mkdir("./sam_configs")

        # Separate parameters for space
        config_dict = self.sam_params

        # Create file name using jobname and store this for gen_config
        config_path = os.path.join(".", "sam_configs", jobname + ".json")
        self.sam_files[jobname] = config_path

        # Save file
        with open(config_path, "w") as file:
            file.write(json.dumps(config_dict, indent=4))
        if self.verbose:
            print("SAM job " + jobname + " config file saved to '" +
                  config_path + "'.")

        # Check that the json as written correctly
        check_config(config_path)
        if self.verbose:
            print("SAM job " + jobname + " config file opens.")

        # Return configuration dictionary
        return config_dict


    def _config_batch(self):
        """If running mutliple technologies, this configures a batch run.

        Note:
            This can apparently be done using either multipe sam config files,
            or multiple arguments. I think, to start, I'll just use multiple
            sam config files. That will let us configure more unique setups
            to compare and, once this is done, won't require extra steps.

            Batching the arguments themselves will result in all combinations,
            which might not always be desired.
        """
        # Separate parameters for space
        params = self.top_params

        # Create separate files for each job name
        tag = params["set_tag"]
        sam_dicts = [{tag: file} for _, file in self.sam_files.items()]

        # Create the configuration dictionary
        config_dict = {
            "pipeline_config": "./config_pipeline.json",
            "sets": [
                {
                    "args": {
                        "sam_files": sam_dicts
                    },
                    "files": ["./config_gen.json"],
                    "set_tag": params["set_tag"]
                }
            ]
        }

        # Write json to file
        with open("./config_batch.json", "w") as file:
            file.write(json.dumps(config_dict, indent=4))
        if self.verbose:
            print("BATCH config file saved to './config_batch.json'.")

        # Check that the json as written correctly
        check_config("./config_batch.json")
        if self.verbose:
            print("BATCH config file opens.")

    def _config_collect(self):
        """If there are more than one node we need to combine outputs"""
        # Separate parameters for space
        params = self.top_params

        # Create the dictionary from the current set of parameters
        config_dict = {
            "directories": {
                "logging_directory": params["logdir"],
                "output_directory": params["outdir"],
                "collect_directory": "PIPELINE"  # <--------------------------- Is this auto generated?
            },
            "execution_control": {
                "allocation": params["allocation"],
                "feature": params["feature"],
                "memory": params["memory"],
                "option": params["option"],
                "walltime": params["walltime"]
            },
            "project_control": {
                "file_prefixes": "PIPELINE", # <------------------------------- I guess they're all called PIPELINE_something?
                "dsets": params["outputs"],
                "parallel": params["parallel"],
                "logging_level": params["loglevel"]
            },
            "project_points": self.points_path
        }

        # Save to json using jobname for file name
        with open("./config_collect.json", "w") as file:
            file.write(json.dumps(config_dict, indent=4))
        if self.verbose:
            print("COLLECT config file saved to './config_collect.json'.")

        # Check that the json as written correctly
        check_config("./config_collect.json")
        if self.verbose:
            print("COLLECT config file opens.")

        # Return configuration dictionary
        return config_dict

    def _config_econ(self):
        """Create a econ config file."""
        # Separate parameters for space
        params = self.top_params

        # Get only the econ outputs
        outputs = self.top_params["outputs"]
        econ_outputs = [o in ECON_MODULES for o in outputs]

        # Create the dictionary from the current set of parameters
        config_dict = {
              "cf_file": "PIPELINE",
              "directories": {
                "logging_directory": params["logdir"],
                "output_directory": params["outdir"]
              },
              "execution_control": {
                "allocation": params["allocation"],
                "feature": params["feature"],
                "nodes": params["nodes"],
                "option": params["option"],
                "sites_per_core": params["sites_per_core"],
                "walltime": params["walltime"]
              },
              "project_control": {
                "analysis_years": params["years"],
                "logging_level": params["loglevel"],
                "name": "econ",  # <------------------------------------------- How important is it to set this one?
                "output_request": econ_outputs
              },
              "project_points": "./project_points/project_points.csv",
              "sam_files": self.sam_files  # <--------------------------------- The example keeps the econ sam config separate from the gen config...is that necessary?
            }

        # Save to json using jobname for file name
        with open("./config_econ.json", "w") as file:
            file.write(json.dumps(config_dict, indent=4))
        if self.verbose:
            print("ECON config file saved to './config_econ.json'.")

        # Check that the json as written correctly
        check_config("./config_econ.json")
        if self.verbose:
            print("ECON config file opens.")

        # Return configuration dictionary
        return config_dict      



    def _config_gen(self):
        """create a generation config file."""
        # Separate parameters for space
        params = self.top_params

        # If there are more than one jobs, use batch and pipeline
        if len(self.sam_files) > 1:
            sam_files = "PLACEHOLDER"
        else:
            sam_files = self.sam_files

        # Create the dictionary from the current set of parameters
        config_dict = {
            "directories": {
                "logging_directory": params["logdir"],
                "output_directory": params["outdir"]
            },
            "execution_control": {
                "allocation": params["allocation"],
                "feature": params["feature"],
                "nodes": params["nodes"],
                "option": params["option"],
                "walltime": params["walltime"],
                "sites_per_core": params["sites_per_worker"],
                "memory_utilization_limit": params["memory_utilization_limit"]
            },
            "project_control": {
                "logging_level": params["loglevel"],
                "analysis_years": params["years"],
                "technology": params["tech"],
                "output_request": params["outputs"]
            },
            "project_points": self.points_path,
            "sam_files": sam_files,
            "resource_file": RESOURCE_DATASETS[self.top_params['resource']]
        }

        # Save to json using jobname for file name
        with open("./config_gen.json", "w") as file:
            file.write(json.dumps(config_dict, indent=4))
        if self.verbose:
            print("GEN config file saved to './config_gen.json'.")

        # Check that the json as written correctly
        check_config("./config_gen.json")
        if self.verbose:
            print("GEN config file opens.")

        # Return configuration dictionary
        return config_dict

    def _config_pipeline(self):
        """ What is the pipeline anyways? What conditions require it?"""
        # Separate parameters for space
        params = self.top_params

        # Create the configuration dictionary
        config_dict = {
            "logging": {
                "log_file": None,
                "log_level": params["loglevel"]
            },
            "pipeline": [
                {
                    "generation": "./config_gen.json"
                },
                {
                    "collect": "./config_collect.json"
                }
            ]
        }

        # If there are econ modules
        outputs = self.top_params["outputs"]
        econ_outputs = [o in ECON_MODULES for o in outputs]
        if any(econ_outputs):
            config_dict["pipeline"].append({"econ": "./config_econ.json"})

        # Write json to file
        with open("./config_pipeline.json", "w") as file:
            file.write(json.dumps(config_dict, indent=4))
        if self.verbose:
            print("PIPELINE config file saved to './config_pipeline.json'.")

        # Check that the json as written correctly
        check_config("./config_pipeline.json")
        if self.verbose:
            print("PIPELINE config file opens.")

    def _set_points_path(self):
        """Set the path name for the points file."""
        self.points_path = os.path.join(self.top_params["pointdir"],
                                        "project_points.csv")

    def _set_years(self):
        """Set years attribute to all available if not specified"""
        if self.top_params["years"] == "all":
            if "nsrdb" in self.top_params["resource"]:
                self.years = range(1998, 2019)
            else:
                self.years = range(2007, 2014)
