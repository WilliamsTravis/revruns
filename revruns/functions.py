"""
Notes:

    - Collect cannot apparently be run without a pipeline configuration. This means that a run with more than one
      module will need a pipeline configuration.


Steps to Configure (Order is important for steps 3 through 9):
    1) SAM - This is a json written from a Python dictionary that includes all of the specifications needed for a
             modeled power generator (solar pv, wind turbine, concentrating solar power, etc.). SAM stands for the
             Systems Advisor Model and is the starting point for the reV model. It will generate power generation
             estimates for a point using historical resource data sets. reV will then run at multiple points, or all
             points in a data set.
    2) Project Points - These are the grid IDs (GIDs) of the resource data set where the reV model will simulate. This
                        needs to be saved as a csv with the GIDs in a "gid" column and a key pointing to a SAM
                        configuration file in a "config" column. You may include whatever else you want in this file,
                        only those two columns will be read.
    3) Generation - This module is required for every subsequent module since it uses SAM to generate the initial power
                    figures. Also, though this isn't totally intuitive, it generates our Levelized Cost of Energy
                    figures. You may  If you are running just one SAM configuration using a single node and a single
                    year, this will be all you need to run.
    4) Collect - If you are running the job on multiple nodes, outputs for each year run will be split into chunks.
                 To combine these chunks into single yearly files, a collection configuration will be needed.
    5) Multi-year - If you are running the job for multiple years and want to combine output into a single file, this
                    module will do that for you, and will need a configuration file. It does not appear as though it
                    will combine the larger profile data sets into one.
    6) Aggregation - To reduce the file size and grid resolution to fit outputs into subsequent modules, the aggregation
                     module will resample to larger grid sizes and requires a configuration file.
    7) Supply-curve - To generate cost vs generation supply curves across the area of interest, the "supply-curve"
                      module is used.
    8) Rep-profiles - To select representative profiles for an area of interest, run the "rep-profiles" module. This
                      will choose the most appropriate set (single) profile of whatever output is requested.
    9) Pipeline - This module will run each of the previous modules with a single reV call.
    10) Batch - If you are running multiple SAM configurations, you may run a "batch" job once. This will run the
                full pipeline for each SAM configuration or SAM model parameter that you give it.

Created on Wed Dec  4 07:58:42 2019

@author: twillia2
"""

import json
import os
import ssl
import h5py
import geopandas as gpd
import numpy as np
import pandas as pd

from reV.utilities.exceptions import JSONError
from shapely.geometry import Point

# Fix remote file transfer issues with ssl (for gpd, find a better way).
ssl._create_default_https_context = ssl._create_unverified_context

# For package data
ROOT = os.path.abspath(os.path.dirname(__file__))

# For filtering COUNS
CONUS_FIPS = ['54', '12', '17', '27', '24', '44', '16', '33', '37', '50', '09',
              '10', '35', '06', '34', '55', '41', '31', '42', '53', '22', '13',
              '01', '49', '39', '48', '08', '45', '40', '47', '56', '38', '21',
              '23', '36', '32', '26', '05', '28', '29', '30', '20', '18', '46',
              '25', '51', '11', '19', '04']

# For checking if a requested output requires economic treatment.
ECON_MODULES = ["flip_actual_irr",
                "lcoe_nom",
                "lcoe_real",
                "ppa_price",
                "project_return_aftertax_npv"]

# Checks for reasonable model output value ranges. No scaling factors here.
VARIABLE_CHECKS = {
        "poa": (0, 1000),  # 1,000 MW m-2
        "cf_mean": (0, 240),  # 24 %
        "cf_profile": (0, 990),  # 99 %
        "ghi_mean": (0, 1000),
        "lcoe_fcr": (0, 1000)
        }

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

# Target geographic coordinate system identifiers.
TARGET_CRS = ["+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs ",
              {'init': 'epsg:4326'},
              {'type': 'EPSG', 'properties': {'code': 4326}}]

GEN_TEMPLATE = {
    "directories": {
        "logging_directory": "./logs",
        "output_directory": "./"
    },
    "execution_control": {
        "allocation": "PLACEHOLDER",
        "feature": "--qos=normal",
        "memory_utilization_limit": 0.4,
        "nodes": "PLACEHOLDER",
        "option": "eagle",
        "sites_per_worker": "PLACEHOLDER",
        "walltime": "PLACEHOLDER"
    },

    "logging_level": "INFO",
    "analysis_years": "PLACEHOLDER",
    "technology": "PLACEHOLDER",
    "output_request": "PLACEHOLDER",
    "project_points": "PLACEHOLDER",
    "sam_files": {
        "key": "PLACEHOLDER"
    },
    "resource_file": "PLACEHOLDER"
}

BATCH_TEMPLATE = {
  "pipeline_config": "./config_pipeline.json",
  "sets": [
    {
      "args": {
        "PLACEHOLDER": "PLACEHOLDER",
        "PLACEHOLDER": "PLACEHOLDER"
      },
      "files": [
        "./sam_configs/default.json"
      ],
      "set_tag": "set1"
    }
  ]
}

COLLECT_TEMPLATE = {
    "directories": {
        "collect_directory": "PIPELINE",
        "logging_directory": "./logs",
        "output_directory": "./"
    },
    "execution_control": {
        "allocation": "PLACEHOLDER",
        "feature": "--qos=normal",
        "memory": 90,
        "option": "eagle",
        "walltime": 2.0
    },
    "dsets": "PLACEHOLDER",
    "file_prefixes": "PIPELINE",
    "logging_level": "INFO",
    "parallel": False,
    "project_points": "PLACEHOLDER"
}


MULTIYEAR_TEMPLATE = {
  "directories": {
    "logging_directory": "./logs",
    "output_directory": "./"
  },
  "execution_control": {
    "allocation": "PLACEHOLDER",
    "feature": "--qos=normal",
    "memory": 90,
    "option": "eagle",
    "walltime": 2.0
  },
  "groups": {
    "none": {
      "dsets": "PLACEHOLDER",
      "source_dir": "./outputs",
      "source_files": "PIPELINE",
      "source_prefix": ""
    }
  },
  "logging_control": "INFO"
}

AGGREGATION_TEMPLATE = {
  "cf_dset": "cf_mean-means",
  "data_layers": {
    "slope": {
      "dset": "srtm_slope",
      "method": "mean"
    },
    "model_region": {
      "dset": "reeds_regions",
      "method": "mode"
    }
  },
  "directories": {
    "logging_directories": "./logs",
    "output_directory": "./"
  },
  "excl_dict": {
    "PLACEHOLDER": {
      "exclude_values": "PLACEHOLDER"
    },
  },
  "excl_fpath": "PLACEHOLDER",
  "execution_control": {
    "allocation": "PLACEHOLDER",
    "feature": "--qos=normal",
    "memory": 90,
    "nodes": 4,
    "option": "eagle",
    "walltime": 2.0
  },
  "gen_fpath": "PIPELINE",
  "lcoe_dset": "lcoe_fcr-means",
  "power_density": "PLACEHOLDER",
  "res_class_bins": "PLACEHOLDER",
  "res_class_dset": "PLACEHOLDER",
  "res_fpath": "PLACHOLDER",
  "resolution": 64,
  "tm_dset": "PLACHOLDER"
}

SUPPLYCURVE_TEMPLATE = {
  "directories": {
    "logging_directory": "./logs",
    "output_directory": "./outputs"
  },
  "execution_control": {
    "allocation": "PLACEHOLDER",
    "feature": "--qos=normal",
    "memory": 90,
    "nodes": 4,
    "option": "eagle",
    "walltime": 2.0
  },
  "fixed_charge_rate": "PLACEHOLDER",
  "sc_features": ("/projects/rev/data/transmission/" +
                  "conus_pv_tline_multipliers.csv"),
  "sc_points": "PIPELINE",
  "simple": False,
  "trans_table": ("/projects/rev/data/transmission/" +
                  "conus_trans_lines_cache_offsh_064_sj_infsink.csv"),
  "transmission_costs": {
    "available_capacity": "PLACEHOLDER",
    "center_tie_in_cost": "PLACEHOLDER",
    "line_cost": "PLACEHOLDER",
    "line_tie_in_cost": "PLACEHOLDER",
    "sink_tie_in_cost": "PLACEHOLDER",
    "station_tie_in_cost": "PLACEHOLDER"
  }
}

REPPROFILES_TEMPLATE = {
  "cf_dset": "cf_profile-{}",
  "directories": {
    "logging_directory": "./logs",
    "output_directory": "./outputs"
  },
  "err_method": "rmse",
  "execution_control": {
    "allocation": "PLACEHOLDER",
    "feature": "--qos=normal",
    "memory": 90,
    "nodes": 4,
    "option": "eagle",
    "site_per_worker": 100,
    "walltime": 2.0
  },
  "gen_fpath": "PIPELINE",
  "n_profiles": 1,
  "analysis_years": "PLACEHOLDER",
  "logging_level": "INFO",
  "reg_cols": [
    "model_region",
    "res_class"
  ],
  "rep_method": "meanoid",
  "rev_summary": "PIPELINE"
}

TEMPLATES = {
    "gen": GEN_TEMPLATE,
    "co": COLLECT_TEMPLATE,
    "my": MULTIYEAR_TEMPLATE,
    "ag": AGGREGATION_TEMPLATE,
    "sc": SUPPLYCURVE_TEMPLATE,
    "rp": REPPROFILES_TEMPLATE,
    "ba": BATCH_TEMPLATE
}

# Default SAM model parameters
SOLAR_SAM_PARAMS = {
    "azimuth": "PLACEHOLDER",
    "array_type": "PLACEHOLDER",
    "capital_cost": "PLACEHOLDER",
    "clearsky": "PLACEHOLDER",
    "compute_module": "PLACEHOLDER",
    "dc_ac_ratio": "PLACEHOLDER",
    "fixed_charge_rate": "PLACEHOLDER",
    "fixed_operating_cost": "PLACEHOLDER",
    "gcr": "PLACEHOLDER",
    "inv_eff": "PLACEHOLDER",
    "losses": "PLACEHOLDER",
    "module_type": "PLACEHOLDER",
    "system_capacity": "PLACEHOLDER",
    "tilt": "PLACEHOLDER",
    "variable_operating_cost": "PLACEHOLDER"
}

WIND_SAM_PARAMS = {
        "adjust:constant": "PLACEHOLDER",
        "capital_cost" : "PLACEHOLDER",
        "fixed_operating_cost" : "PLACEHOLDER",
        "fixed_charge_rate": "PLACEHOLDER",
        "icing_cutoff_temp": "PLACEHOLDER",
        "icing_cutoff_rh": "PLACEHOLDER",
        "low_temp_cutoff": "PLACEHOLDER",
        "system_capacity": "PLACEHOLDER",
        "variable_operating_cost": "PLACEHOLDER",
        "wind_farm_losses_percent": "PLACEHOLDER",
        "wind_farm_wake_model": "PLACEHOLDER",
        "wind_farm_xCoordinates": "PLACEHOLDER",
        "wind_farm_yCoordinates": "PLACEHOLDER",
        "wind_resource_model_choice": "PLACEHOLDER",
        "wind_resource_shear":"PLACEHOLDER",
        "wind_resource_turbulence_coeff": "PLACEHOLDER",
        "wind_turbine_cutin": "PLACEHOLDER",
        "wind_turbine_hub_ht": "PLACEHOLDER",
        "wind_turbine_powercurve_powerout": "PLACEHOLDER",
        "wind_turbine_powercurve_windspeeds": "PLACEHOLDER",
        "wind_turbine_rotor_diameter": "PLACEHOLDER"
}

SAM_TEMPLATES = {
    "pvwattsv5": SOLAR_SAM_PARAMS,
    "pvwattsv7": SOLAR_SAM_PARAMS,
    "wind": WIND_SAM_PARAMS
}

# Functions.
def data_path(path):
    """Path to local package data directory"""
    return os.path.join(ROOT, 'data', path)


def box_points(bbox, crd_path=data_path("nsrdb_v3_coords.csv")):
    """Filter grid ids by geographical bounding box

    Parameters:
        bbox (list): A list containing the geographic coordinates of the
                     desired bounding box in this order:
                         [min lon, min lat, max lon, max lat]
        crd_path (str): The local path to the desired resource coordinate
                        list.

    Returns:
        pandas.core.frame.DataFrame: A data frame of grid IDs and coordinates.
    """
    # Resource coordinate data frame from get_coordinates
    grid = pd.read_csv(crd_path)

    # Filter the data frame for points within the bounding box
    crds = grid[(grid["lon"] > bbox[0]) &
                (grid["lat"] > bbox[1]) &
                (grid["lon"] < bbox[2]) &
                (grid["lat"] < bbox[3])]

    return crds


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


def project_points(tag="default", resource="nsrdb_v3", points=1000, gids=None):
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
    if gids is None:
        if isinstance(points, int):
            gridids = np.arange(0, points)
            point_df = pd.DataFrame({"gid": gridids, "config": tag})
            point_df = point_df.join(coords)
        elif isinstance(points, str) and points == "all":
            gridids = np.arange(0, RESOURCE_DIMS[resource])
            point_df = pd.DataFrame({"gid": gridids, "config": tag})
            point_df = point_df.join(coords)
        else:
            point_df = points.copy()
            point_df["gid"] = point_df.index
            point_df["config"] = tag

    # Let's just have a list of GIDs over ride everything for now
    else:
        point_df = coords.iloc[gids]
        point_df["gid"] = point_df.index
        point_df["config"] = tag

    # Return data frame
    return point_df


def conus_points(resource):
    """It takes so long to get all the CONUS points, so until I've fixed
    shape_points, I'm going to store these in the data folder.
    """

    # If it isn't saved yet, retrieve and save
    if not os.path.exists(data_path(resource + "_conus_coords.csv")):
        shp = gpd.read_file("https://www2.census.gov/geo/tiger/TIGER2017/" +
                            "STATE/tl_2017_us_state.zip")
        shp = shp[shp["STATEFP"].isin(CONUS_FIPS)]
        points = shape_points(shp, resource)
        points.to_csv(data_path(resource + "_conus_coords.csv"), index = False)
    else:
        points = pd.read_csv(data_path(resource + "_conus_coords.csv"))

    return points


def shape_points(shp, resource="nsrdb_v3"):
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

    # Get the coordinates associated with the resource
    crd_file = resource + "_coords.csv"
    crd_path = data_path(crd_file)

    # The resource data sets are large, subset by bounding box first
    bbox = shp.geometry.total_bounds
    grid = box_points(bbox, crd_path)

    # Are there too many points to make a spatial object?
    gdf = to_geo(grid)

    # Use sjoin and filter out empty results
    points = gpd.sjoin(gdf, shp, how="left")
    points = points[~pd.isna(points["index_right"])]

    # We only need the coordinates here.
    points = points[["lat", "lon"]]

    return points


def to_geo(df, lat="lat", lon="lon"):
    """ Convert a Pandas data frame to a geopandas geodata frame """
    df["geometry"] = df[[lon, lat]].apply(lambda x: Point(tuple(x)), axis=1)
    gdf = gpd.GeoDataFrame(df, crs={'init': 'epsg:4326'},
                           geometry=df["geometry"])
    return gdf


def to_sarray(df):
    """Encode data frame values, return a structured array and an array of
    dtypes. This is needed for storing pandas data frames in the h5 format.
    """

    for c in df.columns:
        if isinstance(df[c].iloc[0], bytes):
                df[c] = df[c].apply(lambda x: x.decode())

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

    v = df.values
    types = df.dtypes
    struct_types = [make_col_type(col, types) for col in df.columns]
    dtypes = np.dtype(struct_types)
    array = np.zeros(v.shape[0], dtypes)
    for (i, k) in enumerate(array.dtype.names):
        try:
            if dtypes[i].str.startswith('|S'):
                array[k] = df[k].str.encode('utf-8').astype('S')
            else:
                array[k] = v[:, i]
        except:
            print(k, v[:, i])
            raise

    return array, dtypes


def write_config(config_dict, path, verbose):
    """ Write a configuration dictionary to a json file."""
    # what type of configuration is this?
    module = path.split("_")[1].replace(".json", "").upper()

    # Write json to file
    with open(path, "w") as file:
        file.write(json.dumps(config_dict, indent=4))
    if verbose:
        print(module + " config file saved to " + path + ".")

    # Check that the json as written correctly
    check_config(path)
    if verbose:
        print(module + " config file opens.")
