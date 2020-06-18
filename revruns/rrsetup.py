# -*- coding: utf-8 -*-
"""
Run this to setup configuration files from a standard template.
"""

import os
import pkgutil

import click
import numpy as np
import pandas as pd

from revruns import RESOURCE_DATASETS, TEMPLATES, SAM_TEMPLATES, write_config
from xlrd import XLRDError


# Help printouts
DATA = pkgutil.get_data("revruns", "data/rev_inputs.xlsx")
MODULE_CONFIG_PATHS = {
    "generation": "config_generation.json",
    "collect": "config_collect.json",
    "econ": "config_econ.json",
    "offshore": "config_offshore.json",
    "multi-year": "config_multi-year.json",
    "aggregation": "config_aggregation.json",
    "supply-curve": "config_supply-curve.json",
    "rep-profiles": "config_rep-profiles.json",
    "batch": "config_batch.json"
}
DEFAULT_YEARS = {
    "csp": [y for y in range(1998, 2019)],
    "pvwattsv5": [y for y in range(1998, 2019)],
    "pvwattsv7": [y for y in range(1998, 2019)],
    "windpower": [y for y in range(2007, 2014)]
}

# Execution control constants
ALLOCATION = "rev"
FEATURE = "--qos=normal"
OPTION = "eagle"

# For random configurations
NPROFILES = 1
AG_RESOLUTION = 128

# Path constants
EXCLUSIONS = "/projects/rev/data/exclusions/ATB_Exclusions.h5"
SC_MULTIPLIERS = ("/projects/rev/data/transmission/"
                  "conus_128_tline_multipliers.csv")
SC_TRANSMISSION = ("/projects/rev/data/transmission/"
                   "land_offshore_allconns_128.csv")
ORCA_FPATH = ("/projects/rev/data/transmission/preliminary_orca_results_"
              "09042019_JN_gcb_capexmults.csv")

# Look up keys for ORCA
ORCA_KEYS = {
    'turbine_capacity': 15,
    'cost_reductions': 'OW3F2019',
    'cost_reduction_year': 2030,
    'rna_capex_eq': 'OW3F2019',
    "tower_capex_eq": "OW3F2019_constant_tower",
    "pslt_capex_eq": "OW3F2019",
    "substructure_capex_eq": "OW3F2019",
    "sub_install_capex_eq": "OW3F2019",
    "foundation_capex_eq": "OW3F2019",
    "turbine_install_eq": "OW3F2019",
    "export_cable_capex_eq": "OW3F2019"
    }


def get_sheet(file_name, sheet_name=None):
    """Read in a sheet from and excel spreadsheet file."""

    # Open file
    file = pd.ExcelFile(file_name)
    sheets = file.sheet_names

    # Run with no sheet_name for a list of available sheets
    if not sheet_name:
        return sheets

    # Try to open sheet, print options if it fails
    try:
        table = file.parse(sheet_name=sheet_name)
    except XLRDError:
        print(sheet_name + " is not available. Available sheets:\n")
        for s in sheets:
            print("   " + s)

    return table


def write_template(dst=None):
    """Write a template rev input configuration file."""

    if not dst:
        dst = "./rev_inputs.xlsx"

    # Get each sheet data
    sheets = get_sheet(DATA)
    sheets = {s: get_sheet(DATA, s) for s in sheets}
    
    # Create a writer
    writer = pd.ExcelWriter(dst, engine='xlsxwriter')

    # write each
    for s, df in sheets.items():
        df.to_excel(writer, sheet_name=s)

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


class Inputs:

    def __init__(self, master, proj_dir):
        """Initiate Paths object.

        proj_dir = "/shared-projects/rev/projects/southern_co/scratch"
        master_sheet = "/shared-projects/rev/projects/southern_co/scratch/rev_inputs.xlsx"
        """

        self.master = master
        self.module_paths = MODULE_CONFIG_PATHS
        self._expand_proj_dir(proj_dir)
        self._sheets()
        self._set_paths()

    def __repr__(self):
        mpath = self.__dict__["master"]
        ppath = self.__dict__["proj_dir"]
        msg = "master=" + mpath + "  proj_dir=" + ppath
        msg = "<Inputs " + msg + ">"
        return msg

    def build(self):
        """Build configurations with inputs available from the master sheet."""

        # Build module configs
        
        
        
        
        
        write_configs

        # Build SAM configs

        # Build project points


        


    def _expand_proj_dir(self, proj_dir):

        # Expand user or '.'
        proj_dir = os.path.expanduser(proj_dir)
        self.proj_dir = os.path.abspath(proj_dir)

    # Generation
    def _set_path(self, paths, spath, module):

        # Assign a configuration file path to each module iteration
        scen = os.path.basename(spath)
        config_path = self.module_paths[module]
        path = os.path.join(spath, module, config_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        paths[module][scen] = path
        return paths

    def _set_paths(self):
        """Build a paths object for each configuration."""

        # This will be just a dictionary?
        modules = self.module_paths.keys()
        paths = {}
        paths["scenarios"] = {}
        for m in modules:
            paths[m] = {}

        # Scenarios
        scens = self.scenarios["scenario_name"].unique()
        for scen in scens:
            spath = os.path.join(self.proj_dir, scen)
            os.makedirs(spath, exist_ok=True)
            paths["scenarios"][scen] = spath

            for m in modules:
                paths = self._set_path(paths, spath, m)

        # Set paths dictionary as attribute
        self.paths = paths

    def _sheets(self):
        
        # Set each master sheet as an attribute
        sheets = get_sheet(self.master)
        sheets = {s: get_sheet(self.master, s) for s in sheets}
        for s, df in sheets.items():
            setattr(self, s, df)




# PIPELINE_TEMPLATE =  {
#     "logging": {
#         "log_level": "INFO"
#     },
#     "pipeline": []
# }

# @click.command()
# @click.option("--generation", "-gen", is_flag=True, help=AGGREGATION_HELP)
# @click.option("--collect", "-co", is_flag=True, help=COLLECT_HELP)
# @click.option("--multiyear", "-my", is_flag=True, help=MULTIYEAR_HELP)
# @click.option("--aggregation", "-ag", is_flag=True, help=AGGREGATION_HELP)
# @click.option("--supplycurve", "-sc", is_flag=True, help=SUPPLYCURVE_HELP)
# @click.option("--repprofiles", "-rp", is_flag=True, help=REPPROFILES_HELP)
# @click.option("--batch", "-ba", is_flag=True, help=BATCH_HELP)
# @click.option("--tech", "-t", default="pvwattsv5", help=GEN_HELP)
# @click.option("--full", "-f", is_flag=True, help=FULL_HELP)
# @click.option("--allocation", "-alloc", default="PLACEHOLDER", help=ALLOC_HELP)
# @click.option("--output_dir", "-outdir", default="./", help=OUTDIR_HELP)
# @click.option("--log_dir", "-logdir", default="./logs", help=LOGDIR_HELP)
# @click.option("--verbose", "-v", is_flag=True)
# def main(generation, collect, multiyear, aggregation, supplycurve, repprofiles,
#          batch, tech, full, allocation, output_dir, log_dir,
#          verbose):
#     """Write template configuration json files for each reV module specified.
#     Additionaly options will set up template sam_files and default project
#     point files. Files will be written to your current directory.

#     In the output configuration jsons the term 'PLACEHOLDER' is SET for values
#     that require user inputs. Other inputs have defaults that may or may not 
#     be appropriate. To use all available points for the specified generator,
#     provide the '--all-points' or '-ap' flag. To use all available years for a
#     specified generator, provide the '--all-years' or '-ay' flag.

#     Sample Arguments
#     ----------------
#         generation = False
#         collect = False
#         multiyear = False
#         aggregation = True
#         supplycurve = True
#         repprofiles = True
#         batch = False
#         tech = "pvwattsv5"
#         allpoints = True
#         allyears = True
#         full = False
#         verbose = True
#     """

#     # Get requested modules as strings
#     strings = np.array(["gen", "co", "my", "ag", "sc", "rp", "ba"])
#     requested = np.array([generation, collect, multiyear, aggregation,
#                           supplycurve, repprofiles, batch])

#     # Convert module selections from booleans to key strings
#     if full:
#         modules = strings
#     else:
#         modules = strings[requested]

#     # Retrieve the template objects
#     templates = {m: TEMPLATES[m] for m in modules}

#     # Assign all years (easier to subtract than add), and add allocation
#     years = DEFAULT_YEARS[tech]
#     for m in templates.keys():
#         if "analysis_years" in templates[m]:
#             templates[m]["analysis_years"] = years
#         if "execution_control" in templates[m]:
#             templates[m]["execution_control"]["allocation"] = allocation
#         if "directories" in templates[m]:
#             templates[m]["directories"]["output_directory"] = output_dir
#             templates[m]["directories"]["log_directories"] = log_dir

#     # Assign all points if specified  <---------------------------------------- Not done.
#     os.makedirs("./project_points", exist_ok=True)

#     # Write sam template:
#     os.makedirs("./sam_configs", exist_ok=True)
#     sam_template = SAM_TEMPLATES[tech]

#     # Write json file for each template
#     for m in templates.keys():
#         config = templates[m]
#         path = DEFAULT_PATHS[m]
#         write_config(config, path, verbose)

#     # If there are more than one module, write pipeline configuration
#     if len(modules) > 1:
#         pipeline = PIPELINE_TEMPLATE.copy()
#         for m in modules:
#             dict = {MODULE_NAMES[m]: DEFAULT_PATHS[m]}
#             pipeline["pipeline"].append(dict)
#         write_config(pipeline, "./config_pipeline.json", verbose)


# if "__name__" == "__main__":
#     main()

