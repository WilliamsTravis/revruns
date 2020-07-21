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

# Fix remote file transfer issues with ssl (for gpd, find a better way).
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings(action='ignore', category=UserWarning)


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


def decode_cols(df):
    """ Decode the columns of a meta data object from a reV output. 

    Fix:
        When an HDF has been transfered or synced across networks, columns with
        byte format might be stored as strings...meaning that they will be
        strings of bytes of strings (i.e. "b'string'").
    """

    for c in df.columns:
        if isinstance(df[c].iloc[0], bytes):
                df[c] = df[c].apply(lambda x: x.decode())

    return df




def write_config(config_dict, path, verbose=False):
    """ Write a configuration dictionary to a json file."""

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
class JSONError(Exception):
    """
    Error reading json file.
    """


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
                    assert old["transform"] == new["transform"]
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
        try:
            assert type(pandas_obj) ==  pd.core.frame.DataFrame
        except:
            raise AssertionError("Can only use .rr accessor with a pandas "
                                 "data frame.")
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

    def to_bbox(self, bbox):
        """Return points filtered by a bounding box ([xmin, ymin, xmax, ymax])
        """

        df = self._obj.copy()
        df = df[(df["longitude"] >= bbox[0]) &
                (df["latitude"] >= bbox[1]) &
                (df["longitude"] <= bbox[2]) &
                (df["latitude"] <= bbox[3])]
        return df

    def to_geo(self):
        """ Convert a Pandas data frame to a geopandas geodata frame """

        # Let's not transform in place
        df = self._obj.copy()
        df.rr.decode()

        # For a single row
        to_point = lambda x: Point(tuple(x))
        df["geometry"] = df[["longitude", "latitude"]].apply(to_point, axis=1)
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
