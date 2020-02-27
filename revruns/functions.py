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

# Target geographic coordinate system identifiers.  # <------------------------- Check this
TARGET_CRS = ["+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs ",
              {'init': 'epsg:4326'},
              {'type': 'EPSG', 'properties': {'code': 4326}}]

# Default SAM model parameters for pvwattsv5.
SOLAR_SAM_PARAMS = {"azimuth": 180,
                    "array_type": 0,
                    "capital_cost": 39767200,
                    "clearsky": False,
                    "compute_module": "pvwattsv5",
                    "dc_ac_ratio": 1.1,
                    "fixed_charge_rate": 0.096,
                    "fixed_operating_cost": 260000,
                    "gcr": 0.4,
                    "inv_eff": 96,
                    "losses": 14.07566,
                    "module_type": 0,
                    "system_capacity": 5000,
                    "tilt": "latitude",
                    "variable_operating_cost": 0
                    }


GEN_TEMPLATE = {
    "directories": {
        "logging_directory": "./logs",
        "output_directory": "./outputs"
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
    "project_control": {
        "logging_level": "INFO",
        "analysis_years": "PLACEHOLDER",
        "technology": "PLACEHOLDER",
        "output_request": "PLACEHOLDER"
    },
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
        "output_directory": "./outputs"
    },
    "execution_control": {
        "allocation": "PLACEHOLDER",
        "feature": "--qos=normal",
        "memory": 90,
        "option": "eagle",
        "walltime": 2.0
    },
    "project_control": {
        "dsets": "PLACEHOLDER",
        "file_prefixes": "PIPELINE",
        "logging_level": "INFO",
        "parallel": False
    },
    "project_points": "PLACEHOLDER"
}


MULTIYEAR_TEMPLATE = {
  "directories": {
    "logging_directory": "./logs",
    "output_directory": "./outputs"
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
  "project_control": {
    "logging_control": "INFO"
  }
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
    "output_directory": "./outputs"
  },
  "excl_dict": {
    "PLACEHOLDER": {
      "exclude_values": "PLACEHOLDER"
    },
  },
  "excl_fpath": "/projects/rev/data/exclusions/CONUS_Exclusions.h5",
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
  "project_control": {
    "analysis_years": "PLACEHOLDER",
    "logging_level": "INFO"
  },
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
        "capital_cost" : 245000000,
        "fixed_operating_cost" : 7790000,
        "fixed_charge_rate": 0.052,
        "icing_cutoff_temp": 0.0,
        "icing_cutoff_rh": 95.0,
        "low_temp_cutoff": -10,
        "system_capacity": 200000,
        "variable_operating_cost": 0,
        "wind_farm_losses_percent": 12.8,
        "wind_farm_wake_model": 0,
        "wind_farm_xCoordinates": [0],
        "wind_farm_yCoordinates": [0],
        "wind_resource_model_choice": 0,
        "wind_resource_shear": 0.140,
        "wind_resource_turbulence_coeff": 0.10,
        "wind_turbine_cutin": 0.0,
        "wind_turbine_hub_ht": 100.0,
        "wind_turbine_powercurve_powerout": DEFAULT_WTPO,
        "wind_turbine_powercurve_windspeeds": list(np.arange(0, 40.25, 0.25)),
        "wind_turbine_rotor_diameter": 167.0
}

# All default SAM model parameters.
SAM_PARAMS = {"pv": SOLAR_SAM_PARAMS,
              "wind": WIND_SAM_PARAMS}

# Default 'Top Level' parameters, i.e. those that are shared between runs.
TOP_PARAMS = {"allocation": "rev",
              "feature": "--qos=normal",
              "keep_chunks": False,
              "logdir": "./logs",
              "loglevel": "INFO",
              "memory": 90,
              "memory_utilization_limit": 0.4,
              "multi_year": False,
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


#def compare_profiles(datasets,
#                     dataset="cf_profile",
#                     units="$\mathregular{m^{-2}}$",
#                     title="Output Profiles",
#                     cmap="viridis",
#                     savefolder=None,
#                     dpi=300):
#    """Compare profiles from different reV generation models
#
#    Parameters:
#        outputs (list): A list of reV output profile numpy arrays.
#
#    Returns:
#        (png): An image comparing output profiles over time.
#    """
#    # Get the profiles from the datasets
#    profiles = {key: datasets[key][dataset] for key in datasets}
#
#    # Get the time values from one of the data sets
#    keys = list(datasets.keys())
#    time = datasets[keys[0]]["time_index"]
#    nstep = int(len(time) / 15)
#    time_ticks = np.arange(0, len(time), nstep)
#    time_labels = pd.to_datetime(time[1::nstep]).strftime("%b %d %H:%M")
#
#    # Get grouping features
#    groups = list(profiles.keys())
#    for i, grp in enumerate(groups):
#        elements = grp.split("_")
#        module = elements[0].upper()
#        group_feature = elements[1].capitalize()
#        year = elements[2]
#        elements = [module, group_feature, year]
#        group = " ".join(elements)
#        groups[i] = group
#
#    # Get the datasets
#    outputs = [profiles[key] for key in profiles.keys()]
#
#    # Get some information about the outputs
#    noutputs = len(outputs)
#
#    # Transpose outputs so they're horizontal
#    outputs = [out.T for out in outputs]
#
#    # Figure level graph elements
#    fig, axes = plt.subplots(noutputs, 1, figsize=(20, 4))
#    fig.suptitle(title, y=1.15, x=.425, fontsize=20)
#    fig.tight_layout()
#    fig.text(0.425, 0.001, 'Date', ha='center', va='center', fontsize=15)
#    fig.text(0.00, .6, 'Site #', ha='center', va='center', fontsize=15,
#             rotation='vertical')
#
#    # We need a common color map
#    maxes = [np.max(out) for out in outputs]
#    if np.diff(maxes) == 0:
#        maxes[0] = maxes[0] - 1
#    color_template = outputs[int(np.where(maxes == max(maxes))[0])]
#    ctim = axes[0].imshow(color_template, cmap=cmap)
#    clim = ctim.properties()['clim']
#
#    # For each axis plot and format
#    for i, axis in enumerate(axes):
#        axis.imshow(outputs[i], cmap=cmap, clim=clim)
#        axis.set_aspect('auto')
#        axis.set_title(groups[i], fontsize=15)
#        axis.set_xticks([])
#
#    # Set date axis on the last one?
#    axes[i].set_xticks(time_ticks)
#    axes[i].set_xticklabels(time_labels)
#    fig.autofmt_xdate(rotation=-35, ha="left")
#
#    # Set the colorbar
#    cbr = fig.colorbar(ctim, ax=axes.ravel().tolist(), shrink=.9,
#                       pad=0.02)
#    cbr.ax.set_ylabel(units, fontsize=15, rotation=270, labelpad=15)
#
#    # Also save to file
#    if savefolder:
#        if not os.path.exists(savefolder):
#            os.makedirs(savefolder)
#        file = "_".join([module.lower(), dataset, year]) + ".png"
#        path = os.path.join(savefolder, file)
#        fig.savefig(path, bbox_inches="tight", dpi=dpi)
#        plt.close(fig)



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


class Config:
    """Sets reV model key values and generates configuration json files."""
    def __init__(self,
                 technology="pv",
                 top_params=TOP_PARAMS.copy(),
                 verbose=True):
        self.points = "all"
        self.gids = None
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
            assert self.sam_files is not None
            assert len(self.sam_files) > 0
        except AssertionError:
            print("Could not configure GENRATION file, no SAM configuration " +
                  "files detected.\n")
            raise

        # Separate parameters for space
        params = self.top_params
        outputs = self.top_params["outputs"]
        econ_outputs = [o for o in outputs if o in ECON_MODULES]

        # If we are using more than one node, collect the outputs
        if params["nodes"] > 1:
            self._config_collect()
            self._config_pipeline()

        # If we are modeling certain economic modules, use pipeline and econ
        if any(econ_outputs):
            self._config_econ()
            self._config_pipeline()

        # If more than one jobs are needed, use batch and pipeline
        if len(self.sam_files) > 1:
            self._config_batch()
            self._config_pipeline()
            tag = params["set_tag"]
        else:
            tag = list(self.sam_files.keys())[0]

        # If we are combining yearly output
        if len(params["years"]) > 1 and params["multi_year"]:
            self._config_multiyear()

        # Configure the generation file
        self._config_gen()

        # Create project points - changes
        point_df = project_points(tag=tag,
                                  resource=params["resource"],
                                  points=self.points,
                                  gids=self.gids)

        # If we are excluding positive longitudes
        if excl_pos_lon:
            point_df = point_df[point_df["lon"] < 0]

        # Save project points
        point_df.to_csv(self.points_path, index=False)
        if self.verbose:
            print("POINTS" + " saved to '" + self.points_path + "'.")

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

#    def _config_aggregation(self):
#        """If aggregating... """
#
#        # Separate parameters for space
#        params = self.top_params
#
#        # Create configuration dictionary
#        config_dict = {}
#
#        # Write json to file
#        self._write_config(config_dict, "./config_aggregation.json")
#
#        # Return configuration dictionary
#        return config_dict


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
        self._write_config(config_dict, "./config_batch.json")

        # Return configuration dictionary
        return config_dict

    def _config_collect(self):
        """If there are more than one node we need to combine outputs"""

        # Separate parameters for space
        params = self.top_params

        # Create the dictionary from the current set of parameters
        config_dict = {
            "directories": {
                "collect_directory": "PIPELINE", # <--------------------------- Is this auto generated?
                "logging_directory": params["logdir"],
                "output_directory": params["outdir"]
            },
            "execution_control": {
                "allocation": params["allocation"],
                "feature": params["feature"],
                "memory": params["memory"],
                "option": params["option"],
                "walltime": params["walltime"]
            },
            "project_control": {
                "dsets": params["outputs"],
                "file_prefixes": "PIPELINE", # <------------------------------- I guess they're all called PIPELINE_something?
                "keep_chunks": params["keep_chunks"],
                "logging_level": params["loglevel"],
                "parallel": params["parallel"]
            },
            "project_points": self.points_path
        }

        # Save to json using jobname for file name
        self._write_config(config_dict, "./config_collect.json")

        # Return configuration dictionary
        return config_dict

    def _config_econ(self):
        """Create a econ config file."""

        # Separate parameters for space
        params = self.top_params

        # Get only the econ outputs
        outputs = self.top_params["outputs"]
        econ_outputs = [o for o in outputs if o in ECON_MODULES]

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
                "sites_per_worker": params["sites_per_worker"],
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
        self._write_config(config_dict, "./config_econ.json")

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
                "memory_utilization_limit": params["memory_utilization_limit"],
                "nodes": params["nodes"],
                "option": params["option"],
                "sites_per_worker": params["sites_per_worker"],
                "walltime": params["walltime"]
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
        self._write_config(config_dict, "./config_gen.json")

        # Return configuration dictionary
        return config_dict

    def _config_multiyear(self):
        """ If we want to get multiyear figures we need this."""

        # Separate parameters for space
        params = self.top_params

        # Create the configuration dictionary
        config_dict = {
            "directories": {
                "logging_directory": params["logdir"],
                "output_directory":  params["outdir"]
            },
            "execution_control": {
                "allocation":  params["allocation"],
                "feature": params["allocation"],
                "memory": params["memory"],
                "option": params["option"],
                "walltime": params["walltime"]
            },
            "groups": {
                "none": {
                    "dsets": params["outputs"],
                    "source_dir": params["outdir"],
                    "source_files": "PIPELINE",
                    "source_prefix": ""
                }
              },
              "project_control": {
                "logging_level": params["loglevel"]
              }
        }

        # Save to json using jobname for file name
        self._write_config(config_dict, "./config_multi-year.json")

        # Return configuration dictionary
        return config_dict

    def _config_pipeline(self):
        """ If we are trying to run multiple modules at once we need this."""

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
                }
            ]
        }

        # If there are econ modules
        outputs = self.top_params["outputs"]
        econ_outputs = [o in ECON_MODULES for o in outputs]
        if any(econ_outputs):
            config_dict["pipeline"].append({"econ": "./config_econ.json"})

        # If there are multiple nodes, collect. Is this run in order?
        if self.top_params["nodes"] > 1:
            config_dict["pipeline"].append(
                    {"collect": "./config_collect.json"})

        # If multiyear is on
        if self.top_params["multi_year"] and len(self.top_params["years"]) > 1:
            config_dict["pipeline"].append(
                    {"multi-year": "./config_multi-year.json"})

        # Write json to file
        self._write_config(config_dict, "./config_pipeline.json")

#    def _config_rep_profiles(self):
#        """If generating representative profiles... """
#
#        # Separate parameters for space
#        params = self.top_params
#
#        # Create configuration dictionary
#        config_dict = {}
#
#        # Write json to file
#        self._write_config(config_dict, "./config_rep-profiles.json")
#
#        # Return configuration dictionary
#        return config_dict

#    def _config_supply_curve(self):
#        """If generating supply curves... """
#
#        # Separate parameters for space
#        params = self.top_params
#
#        # Create configuration dictionary
#        config_dict = {}
#
#        # Write json to file
#        self._write_config(config_dict, "./config_supply-curve.json")
#
#        # Return configuration dictionary
#        return config_dict

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

    def _write_config(self, config_dict, path):
        """ Write a configuration dictionary to a json file."""
        # what type of configuration is this?
        module = path.split("_")[1].replace(".json", "").upper()

        # Write json to file
        with open(path, "w") as file:
            file.write(json.dumps(config_dict, indent=4))
        if self.verbose:
            print(module + " config file saved to " + path + ".")

        # Check that the json as written correctly
        check_config(path)
        if self.verbose:
            print(module + " config file opens.")
