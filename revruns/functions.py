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

import geopandas as gpd
import h5py
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


def to_geo(df, lat="latitude", lon="longitude"):
    """ Convert a Pandas data frame to a geopandas geodata frame """

    df["geometry"] = df[[lon, lat]].apply(lambda x: Point(tuple(x)), axis=1)
    gdf = gpd.GeoDataFrame(df, crs='epsg:4326', geometry="geometry")
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


def write_config(config_dict, path):
    """ Write a configuration dictionary to a json file."""

    # what type of configuration is this?
    module = path.split("_")[1].replace(".json", "").upper()

    # Write json to file
    with open(path, "w") as file:
        file.write(json.dumps(config_dict, indent=4))

    # Check that the json as written correctly
    check_config(path)


# Configuration Functions
def points(paths):
    """Create project points.

    Parameters
    ----------
    paths : revruns
    .functions.Paths
        A 'Paths' class object that generates path locations for all ATB reV
        model elements.

    Returns
    -------
    None.
    """

    # Use one of the WTK files
    wtk_sample = "/datasets/WIND/conus/v1.0.0/wtk_conus_2013.h5"

    # Check if it exists
    if not os.path.exists(paths.points_path):
        with h5py.File(wtk_sample, "r") as file:
            meta = pd.DataFrame(file["meta"][:])

        # Set gid and config
        meta["config"] = ""
        meta["config"][meta["offshore"] == 1] = "offshore"
        meta["config"][meta["offshore"] == 0] = "onshore"
        meta["gid"] = meta.index

        # Use just the gids and config columns
        pp = meta[["gid", "config"]]

        # Save along the chosen path :)
        pp.to_csv(paths.points_path, index=False)


def sam(paths):
    """Use the master atb sheet to build SAM configurations.

    Parameters
    ----------
    paths : atb_setup.functions.Paths
        A 'Paths' class object that generates path locations for all ATB reV
        model elements.

    Returns
    -------
    None.
    """

    # Get all the tech tables
    tables = paths.tech_tables

    # Get the data tables
    onshore_constants = tables["onshore_constants"]
    offshore_constants = tables["offshore_constants"]
    onshore_curves = tables["onshore_curves"]
    offshore_curves = tables["offshore_curves"]

    # Onshore
    for scenario in paths.tech_scenarios:
        # break
        # Get constant values
        constants = dict(zip(onshore_constants["parameter"],
                             onshore_constants[scenario]))

        capital_cost = constants[TRANSLATIONS["capital_cost"]]
        fixed_operating_cost = constants[TRANSLATIONS["fixed_operating_cost"]]
        fixed_charge_rate = constants[TRANSLATIONS["fixed_charge_rate"]]
        system_capacity = constants[TRANSLATIONS["system_capacity"]]
        wind_turbine_hub_ht = constants[TRANSLATIONS["wind_turbine_hub_ht"]]
        rotor_diameter = constants[TRANSLATIONS["wind_turbine_rotor_diameter"]]
        losses_percent = constants[TRANSLATIONS["wind_farm_losses_percent"]]

        # Curves
        onshore_curves = onshore_curves[onshore_curves["wind_speed"] < 25.25]
        windspeeds = onshore_curves["wind_speed"].tolist()
        powerouts = onshore_curves[scenario].tolist()

        # Create Configuration
        config = {}
        config["adjust:constant"] = 0.0
        config["capital_cost"] = capital_cost
        config["fixed_operating_cost"] = fixed_operating_cost
        config["fixed_charge_rate"] = fixed_charge_rate
        config["system_capacity"] = system_capacity * 1000  # <---------------- unit conversion
        config["wind_farm_xCoordinates"] = [0]
        config["wind_farm_yCoordinates"] = [0]
        config["wind_farm_losses_percent"] = losses_percent * 100  # <--------- unit conversion
        config["wind_farm_wake_model"] = 0
        config["wind_resource_model_choice"] = 0
        config["wind_resource_shear"] = 0.14
        config["wind_resource_turbulence_coeff"] = 0.1
        config["wind_turbine_cutin"] = 0.0
        config["wind_turbine_hub_ht"] = wind_turbine_hub_ht
        config["wind_turbine_rotor_diameter"] = rotor_diameter
        config["variable_operating_cost"] = 0
        config["wind_turbine_powercurve_powerout"] = powerouts
        config["wind_turbine_powercurve_windspeeds"] = windspeeds

        # Write to file
        file_path = paths.sam_paths["land"][scenario]
        with open(file_path, "w") as config_file:
            config_file.write(json.dumps(config, indent=4))

    # Offshore - only mid case for now but making all of them for later
    for scenario in paths.tech_scenarios:

        # Get constant values
        constants = dict(zip(offshore_constants["parameter"],
                             offshore_constants[scenario]))

        system_capacity = constants[TRANSLATIONS["system_capacity"]]
        wind_turbine_hub_ht = constants[TRANSLATIONS["wind_turbine_hub_ht"]]
        rotor_diameter = constants[TRANSLATIONS["wind_turbine_rotor_diameter"]]
        losses_percent = constants[TRANSLATIONS["wind_farm_losses_percent"]]

        # Curves
        windspeeds = offshore_curves["wind_speed"].dropna().tolist()
        powerouts = offshore_curves[scenario].dropna().tolist()

        # Create config
        config = {}

        config["adjust:constant"] = 0.0
        config["sub_tech"] = "PLACEHOLDER"
        config["system_capacity"] = system_capacity * 1000  # <---------------- Watch out for units
        config["wind_farm_losses_percent"] = 0
        config["wind_farm_wake_model"] = 0.0
        config["wind_farm_xCoordinates"] = [0]
        config["wind_farm_yCoordinates"] = [0]
        config["wind_resource_model_choice"] = 0.0
        config["wind_resource_shear"] = 0.14
        config["wind_resource_turbulence_coeff"] = 0.1
        config["wind_turbine_cutin"] = 0.0
        config["wind_turbine_rotor_diameter"] = rotor_diameter
        config["wind_turbine_hub_ht"] = wind_turbine_hub_ht
        config["wind_turbine_powercurve_powerout"] = powerouts
        config["wind_turbine_powercurve_windspeeds"] = windspeeds
        config["variable_operating_cost"] = 0.0
        config["capital_cost"] = 99999  # <------------------------------------ Placeholder
        config["fixed_charge_rate"] = 0.099999  # <---------------------------- Placeholder
        config["fixed_operating_cost"] = 99999  # <---------------------------- Placeholder

        # Write file
        file_path = paths.sam_paths["offshore"][scenario]
        with open(file_path, "w") as config_file:
            config_file.write(json.dumps(config, indent=4))

def generation(paths):
    """Use the master atb sheet to build generation configuration jsons.

    Parameters
    ----------
    paths : atb_setup.functions.Paths
        A 'Paths' class object that generates path locations for all ATB reV
        model elements.

    Returns
    -------
    None.
    """

    # What varies? Sam configurations only right?
    for scenario in paths.tech_scenarios:
        sam_files = {
            "onshore": paths.sam_paths["land"][scenario],
            "offshore": paths.sam_paths["offshore"]["mid"]  # <---------------- Fixed at mid case for 2020 runs
        }
        config = {}
        config["directories"] = {
            "log_directory": "./logs",
            "output_directory": "./outputs"
        }
        config["execution_control"] = {
            "allocation": ALLOCATION,
            "option": OPTION,
            "feature": FEATURE,
            "memory_utilization_limit": 0.4,
            "nodes": 20,
            "sites_per_worker": 100,
            "walltime": 2.0
        }
        config["log_level"] = "INFO"
        config["analysis_years"] = [
            2007,
            2008,
            2009,
            2010,
            2011,
            2012,
            2013
        ]
        config["technology"] = "windpower"
        config["output_request"] = [
            "cf_profile",
            "cf_mean",
            "lcoe_fcr",
            "ws_mean"
        ]
        config["project_points"] = paths.points_path
        config["sam_files"] = sam_files
        config["resource_file"] = "/datasets/WIND/conus/v1.0.0/wtk_conus_{}.h5"



        # Write to file
        file = paths.gen_paths[scenario]
        with open(file, "w") as config_file:
            config_file.write(json.dumps(config, indent=4))


def bats(paths):
    """Write bat curtailment configurations, add generation configs, and update
    aggregation configs where necessary. Must be run after every other config
    module is complete (i.e. after pipeline)."""

    # Write curtailment configs
    for key, path in paths.bat_paths["sam"].items():
        with open(path, "w") as config_file:
            config_file.write(json.dumps(BAT_CONFIGS[key], indent=4))

    # Create new generation pipelines based on the mid case
    mid_gen = paths.gen_paths["mid"]
    mid_folder = os.path.dirname(mid_gen)
    for key, gen_path in paths.bat_paths["gen"].items():

        # Create new directory
        bat_dir = os.path.dirname(gen_path)
        os.makedirs(bat_dir, exist_ok=True)

        # Copy over jsons from the mid case gen folder
        original_configs = glob(os.path.join(mid_folder, "*json"))
        new_configs = [os.path.join(bat_dir, os.path.basename(c)) for
                       c in original_configs]
        for i, file in enumerate(original_configs):
            new_file = new_configs[i]
            shutil.copy(file, new_file)

        # open the old gen file
        with open(gen_path, "r") as file:
            gen_config = json.load(file)

        # Change this bit
        gen_config["curtailment"] = paths.bat_paths["sam"][key]

        # Write it back again
        with open(gen_path, "w") as file:
            file.write(json.dumps(gen_config, indent=4))

        # Open the old pipeline file
        pipe_path = gen_path.replace("config_gen.json", "config_pipeline.json")
        with open(pipe_path) as file:
            pipe_config = json.load(file)

        # Change these bits
        for entry in pipe_config["pipeline"]:
            key = list(entry.keys())[0]
            value = entry[key]
            old_dir = os.path.dirname(value)
            new_val = value.replace(old_dir, bat_dir)
            entry[key] = new_val

        # Write it back again
        with open(pipe_path, "w") as file:
            file.write(json.dumps(pipe_config, indent=4))

    # Find the aggregations that need new generations
    agg_paths = paths.aggregation_paths
    agg_descs = paths.agg_scenario_descriptions
    to_update = {}
    for scenario, path in agg_paths.items():
        desc = agg_descs[scenario]
        if "Bat Curtailment" in desc:
            if "Smart" in desc:
                to_update["smart"] = path
            elif "Blanket" in desc:
                to_update["blanket"] = path

    # Update these
    for key, agg_path in to_update.items():

        # New gen path
        bat_dir = os.path.dirname(paths.bat_paths["gen"][key])
        gen_fpath = os.path.join(bat_dir, "outputs", "outputs_multi-year.h5")

        # open the old file
        with open(agg_path, "r") as file:
            agg_config = json.load(file)

        # Change this bit
        agg_config["gen_fpath"] = gen_fpath

        # Write it back again
        with open(agg_path, "w") as file:
            file.write(json.dumps(agg_config, indent=4))

        # Open the rep_profiles path and do the same
        rep_path = agg_path.replace("config_aggregation", "config_rep-profiles")
        with open(rep_path, "r") as file:
            rep_config = json.load(file)
        rep_config["gen_fpath"] = gen_fpath
        with open(rep_path, "w") as file:
            file.write(json.dumps(rep_config, indent=4))


def offshore(paths):
    """Use the master atb sheet to build offshore configuration jsons.

    Parameters
    ----------
    paths : atb_setup.functions.Paths
        A 'Paths' class object that generates path locations for all ATB reV
        model elements.

    Returns
    -------
    None.
    """

    for scenario in paths.tech_scenarios:
        config = {}
        sam_files = {
            "onshore": paths.sam_paths["land"][scenario],
            "offshore": paths.sam_paths["offshore"]["mid"]  # <---------------- Fixed at mid case for 2020 runs
        }
        config["directories"] = {
            "log_directory": "./logs",
            "output_directory": "./outputs"
        }
        config["execution_control"] = {
            "allocation": ALLOCATION,
            "feature": FEATURE,
            "option": OPTION,
            "walltime": 1.0
        }
        config["log_level"] = "INFO"
        config["gen_fpath"] = "PIPELINE"
        config["offshore_fpath"] = ORCA_FPATH
        config["project_points"] = paths.points_path
        config["sam_files"] = sam_files

        # Write to file
        file = paths.offshore_paths[scenario]
        with open(file, "w") as config_file:
            config_file.write(json.dumps(config, indent=4))

def econ(paths):
    """Use the master atb sheet to build econ cofiguration jsons.

    Parameters
    ----------
    paths : atb_setup.functions.Paths
        A 'Paths' class object that generates path locations for all ATB reV
        model elements.

    Returns
    -------
    None.
    """

    for scenario in paths.tech_scenarios:

        # Scenario specific paths
        sam_files = {
            "onshore": paths.sam_paths["land"][scenario],
            "offshore": paths.sam_paths["offshore"]["mid"]  # <---------------- Fixed at mid case for 2020 runs
        }
        site_data_path = paths.site_data_paths[scenario]

        # Build the configuration dictionary
        config = {}
        config["analysis_years"] = [
            2007,
            2008,
            2009,
            2010,
            2011,
            2012,
            2013
        ]
        config["append"] = True
        config["cf_file"] = "PIPELINE"
        config["directories"] = {
            "log_directory": "./logs",
            "output_directory": "./outputs"
            }
        config["execution_control"] = {
            "allocation": ALLOCATION,
            "option": OPTION,
            "feature": FEATURE,
            "memory_utilization_limit": 0.4,
            "nodes": 20,
            "sites_per_worker": 100,
            "walltime": 3.0
        }
        config["log_level"] = "INFO"
        config["output_request"] = ["lcoe_fcr"]
        config["project_points"] = paths.points_path
        config["sam_files"] = sam_files
        config["site_data"] = site_data_path
        config["technology"] = "windpower"

        # Write to file
        file = paths.econ_paths[scenario]
        with open(file, "w") as config_file:
            config_file.write(json.dumps(config, indent=4))

def collect(paths):
    """Use the master atb sheet to build collect configuration jsons.

    Parameters
    ----------
    paths : atb_setup.functions.Paths
        A 'Paths' class object that generates path locations for all ATB reV
        model elements.

    Returns
    -------
    None.
    """

    for scenario in paths.tech_scenarios:
        config = {}
        config["directories"] = {
            "collect_directory": "PIPELINE",
            "log_directory": "./logs",
            "output_directory": "./outputs"
        }
        config["execution_control"] = {
            "allocation": ALLOCATION,
            "feature": FEATURE,
            "option": OPTION,
            "walltime": 1.0,
            "memory": 96,
        }
        config["dsets"] = [
            "cf_profile",
            "cf_mean",
            "lcoe_fcr",
            "ws_mean"
        ]
        config["file_prefixes"] = "PIPELINE"
        config["log_level"] = "INFO"
        config["parallel"] = False
        config["project_points"] = None

        # Write to file
        file = paths.collect_paths[scenario]
        with open(file, "w") as config_file:
            config_file.write(json.dumps(config, indent=4))


def multi_year(paths):
    """Use the master atb sheet to build multi-year configuration jsons.

    Parameters
    ----------
    paths : atb_setup.functions.Paths
        A 'Paths' class object that generates path locations for all ATB reV
        model elements.

    Returns
    -------
    None.
    """

    for scenario in paths.tech_scenarios:
        config = {}

        config["directories"] = {
            "log_directory": "./logs",
            "output_directory": "./outputs"
        }
        config["execution_control"] = {
            "allocation": ALLOCATION,
            "feature": FEATURE,
            "option": OPTION,
            "memory": 192,
            "walltime": 1.0
        }
        config["groups"] = {
            "none": {
                "dsets": [
                    "cf_profile",
                    "cf_mean",
                    "lcoe_fcr",
                    "ws_mean"
                ],
                "source_dir": "./outputs",
                "source_files": "PIPELINE",
                "source_prefix": ""
            }
        }
        config["log_control"] = "INFO"

        # Write to file
        file = paths.multiyear_paths[scenario]
        with open(file, "w") as config_file:
            config_file.write(json.dumps(config, indent=4))



def aggregation(paths):
    """Use the master atb sheet to build aggregation configuration jsons.

    Parameters
    ----------
    paths : atb_setup.functions.Paths
        A 'Paths' class object that generates path locations for all ATB reV
        model elements.

    Returns
    -------
    None.
    """

    # Get the master tables for this module
    agg_table = paths.agg_table
    char_table = paths.character_table
    char_table = char_table[["method", "dataset", "tif_name"]].dropna()

    for scenario in paths.agg_scenarios:

        # Find the appropriate generation outputs folder
        tech_scenario = scenario.split("_")[-1]
        gen_dir = os.path.dirname(paths.gen_paths[tech_scenario])
        gen_file = os.path.join(gen_dir, "outputs", "outputs_multi-year.h5")

        # Create the configuration
        config = {}
        config["directories"] = {
            "log_directories": "./logs",
            "output_directory": "./"
        }
        config["cf_dset"] = "cf_mean-means"
        config["excl_fpath"] = EXCLUSIONS
        config["execution_control"] = {
            "allocation": ALLOCATION,
            "feature":  FEATURE,
            "option": OPTION,
            "memory": 90,
            "nodes": 10,
            "walltime": 1.0
        }
        config["gen_fpath"] = gen_file
        config["lcoe_dset"] = "lcoe_fcr-means"
        config["res_class_dset"] = "ws_mean-means"
        config["res_fpath"] = "/datasets/WIND/conus/v1.0.0/wtk_conus_{}.h5"
        config["resolution"] = AG_RESOLUTION
        config["tm_dset"] = "techmap_wtk"

        # Characterization datasets
        config["data_layers"] = {}
        for _, row in char_table.iterrows():
            if ".tif" in row["tif_name"]:
                dset = row["dataset"]
                method = row["method"]
                config["data_layers"][dset] = {
                    "dset": dset,
                    "method": method
                }

        # Exclusion datasets
        scenario_table = agg_table[["dataset", scenario, "excl_values"]]
        scenario_table = scenario_table.dropna().drop_duplicates()
        config["excl_dict"] = {}
        for _, row in scenario_table.iterrows():

            dset = row["dataset"]
            if row[scenario]:
                value = row["excl_values"]
                if isinstance(value, (int, float)):
                    config["excl_dict"][dset] = {
                        "exclude_values": [value]
                    }
                if value == "%":
                    config["excl_dict"][dset] = {
                        "use_as_weights": 1
                        }

            # Add in albers
            config["excl_dict"]["albers"] = {
                "include_values": [1]
                }

            # For legacy, add in naris wind
            if "_l_" in scenario:
                config["excl_dict"]["naris_wind"] = {
                    "use_as_weights": 1
                    }

        # Power density
        tech_scen = scenario.split("_")[-1]
        table = paths.tech_tables["onshore_constants"]
        label = TRANSLATIONS["power_density"]
        power_density = table[tech_scen][table["parameter"] == label].values[0]
        config["power_density"] = power_density
        config["res_class_bins"] = None

        # Write to file
        file = paths.aggregation_paths[scenario]
        with open(file, "w") as config_file:
            config_file.write(json.dumps(config, indent=4))


def supply_curve(paths):
    """Use the master atb sheet to build supply-curve configuration jsons.

    Parameters
    ----------
    paths : atb_setup.functions.Paths
        A 'Paths' class object that generates path locations for all ATB reV
        model elements.

    Returns
    -------
    None.
    """

    # Get the technology tables
    itable = paths.tech_tables["interconnections"]
    ostable = paths.tech_tables["onshore_constants"]

    # Get transmission costs
    stic_str = TRANSLATIONS["station_tie_in_cost"]
    ltic_str = TRANSLATIONS["line_tie_in_cost"]
    lc_str = TRANSLATIONS["line_cost"]
    stic = itable["cost"][itable["parameter"] == stic_str].iloc[0]
    ltic = itable["cost"][itable["parameter"] == ltic_str].iloc[0]
    lc = itable["cost"][itable["parameter"] == lc_str].iloc[0]

    for ag_scenario in paths.agg_scenarios:

        # We need both scenarios
        tech_scnerio = ag_scenario.split("_")[-1]

        # Get fixed charge rate
        fcr_str = "Fixed Charge Rate"
        fcr = ostable[tech_scnerio][ostable["parameter"] == fcr_str].iloc[0]

        # Build configuration dictionary
        config = {}

        # Conistent elements
        config["directories"] = {
            "log_directories": "./logs",
            "output_directory": "./"
        }
        config["execution_control"] = {
            "allocation": ALLOCATION,
            "feature": FEATURE,
            "option": OPTION,
            "memory": 190,
            "nodes": 10,
            "walltime": 1.0
        }
        config["sc_features"] = SC_MULTIPLIERS
        config["sc_points"] = "PIPELINE"
        config["simple"] = False
        config["trans_table"] = SC_TRANSMISSION
        config["fixed_charge_rate"] = fcr
        config["transmission_costs"] = {
            "line_cost": lc,
            "line_tie_in_cost": ltic,
            "station_tie_in_cost": stic,
            "available_capacity": 1.0,
            "center_tie_in_cost": 0,
            "sink_tie_in_cost": 14000
        }

        # Write to file
        file = paths.sc_paths[ag_scenario]
        with open(file, "w") as config_file:
            config_file.write(json.dumps(config, indent=4))


def rep_profiles(paths):
    """Use the master atb sheet to build rep_profiles configuration jsons.

    Parameters
    ----------
    paths : atb_setup.functions.Paths
        A 'Paths' class object that generates path locations for all ATB reV
        model elements.

    Returns
    -------
    None.
    """

    for ag_scenario in paths.agg_scenarios:

        # Find the appropriate generation outputs folder
        tech_scenario = ag_scenario.split("_")[-1]
        gen_dir = os.path.dirname(paths.gen_paths[tech_scenario])
        gen_file = os.path.join(gen_dir, "outputs", "outputs_multi-year.h5")

        # Build configuration dictionary
        config = {}
        config["cf_dset"] = "cf_profile-{}"
        config["directories"] = {
            "log_directory": "./logs",
            "output_directory": "./"
        }
        config["err_method"] = "rmse"
        config["execution_control"] = {
            "allocation": ALLOCATION,
            "feature": FEATURE,
            "option": OPTION,
            "memory": 196,
            "nodes": 10,
            "site_per_worker": 100,
            "walltime": 4.0
        }
        config["gen_fpath"] = gen_file
        config["n_profiles"] = NPROFILES
        config["analysis_years"] = [
            2007,
            2008,
            2009,
            2010,
            2011,
            2012,
            2013
        ]
        config["log_level"] = "INFO"
        config["reg_cols"] = ["sc_gid"]
        config["rep_method"] = "meanoid"
        config["rev_summary"] = "PIPELINE"

        # Write to file
        file = paths.rp_paths[ag_scenario]
        with open(file, "w") as config_file:
            config_file.write(json.dumps(config, indent=4))


def pipeline(paths):
    """Use the master atb sheet to build pipeline configuration jsons.

    Parameters
    ----------
    paths : atb_setup.functions.Paths
        A 'Paths' class object that generates path locations for all ATB reV
        model elements.

    Returns
    -------
    None.
    """

    # Generation
    for scenario in paths.tech_scenarios:

        config = {}

        config["logging"] = {
            "log_file": None,
            "log_level": "INFO"
        }
        config["pipeline"] = [
            {
                "generation": paths.gen_paths[scenario]
            },
            {
                "econ": paths.econ_paths[scenario]
            },
            {
                "offshore": paths.offshore_paths[scenario]
            },
            {
                "collect": paths.collect_paths[scenario]
            },
            {
                "multi-year": paths.multiyear_paths[scenario]
            }
        ]

        # write to file
        file = paths.pipeline_paths["gen"][scenario]
        with open(file, "w") as config_file:
            config_file.write(json.dumps(config, indent=4))


    # Aggregation
    for scenario in paths.agg_scenarios:

        config = {}

        config["logging"] = {
            "log_file": None,
            "log_level": "INFO"
        }
        config["pipeline"] = [
            {
                "supply-curve-aggregation": paths.aggregation_paths[scenario]
            },
            {
                "supply-curve": paths.sc_paths[scenario]
            },
            {
                "rep-profiles": paths.rp_paths[scenario]
            }
        ]

        # write to file
        file = paths.pipeline_paths["aggregation"][scenario]
        with open(file, "w") as config_file:
            config_file.write(json.dumps(config, indent=4))


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

        items = glob(self.join(*args, "*"))

        return items

    def folders(self, *args):
        """List folders in the data_path or in sub directories."""

        items = self.contents(*args)
        folders = [i for i in items if os.path.isdir(i)]

        return folders

    def files(self, *args):
        """List files in the data_path or in sub directories."""

        items = self.contents(*args)
        folders = [i for i in items if os.path.isfile(i)]
        print(self.join(*args))

        return folders

    def _expand_check(self):

        # Expand the user path if a tilda is present in the root folder path.
        if "~" in self.data_path:
            self.data_path = os.path.expanduser(self.data_path)

        # Make sure path exists
        os.makedirs(self.data_path, exist_ok=True)