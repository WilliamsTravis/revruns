# -*- coding: utf-8 -*-
"""
Run this to setup configuration files from a standard template.
"""

import os
import pkgutil
import shutil

import click
import numpy as np
import pandas as pd

from colorama import Fore, Style
from revruns import ROOT, RESOURCE_DATASETS, TEMPLATES, SAM_TEMPLATES
from revruns import  write_config
from xlrd import XLRDError


# Help printouts
FILE_HELP = ("A standard rev input excel file from which to generation "
             "configurations. A new template input file can be generation "
             "using the '--template' or '-t' flag. (string)")
TEMPLATE_HELP = ("Generate a new template rev input excel sheet. Will save as "
                 "'rev_inputs.xslx' in the directory set for 'project_dir' "
                 " (boolean)")
PDIR_HELP = ("Directory in which to write rrsetup outputs. Defaults to "
             "current directory. (string)")
DATA = pkgutil.get_data("revruns", "data/rev_inputs.xlsx")
MODULE_CONFIG_PATHS = {
    "generation": {
        "folder": "generation",
        "file": "config_generation.json"
    },
    "collect": {
        "folder": "generation", 
        "file": "config_collect.json"
    },
    "econ": {
        "folder": "generation", 
        "file": "config_econ.json"
    },
    "offshore": {
        "folder": "generation", 
        "file": "config_offshore.json"
    },
    "multi-year": {
        "folder": "generation", 
        "file": "config_multi-year.json"
    },
    "aggregation": {
        "folder": "aggregation", 
        "file": "config_aggregation.json"
    },
    "supply-curve-aggregation": {
        "folder": "aggregation", 
        "file": "config_supply-curve.json"
    },
    "rep-profiles": {
        "folder": "aggregation", 
        "file": "config_rep-profiles.json"
    }
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

    # Get package data path
    dpath = os.path.join(ROOT, "data", "rev_inputs.xlsx")

    # Just copy it over
    if not os.path.exists(dst):
        shutil.copy(dpath, dst)
    else:
        print(Fore.YELLOW + dst + " exists. Choose a different name for "
              "the template." + Style.RESET_ALL)


class Paths:

    def __init__(self, master, proj_dir):
        """Initiate Paths object.

        proj_dir = "/shared-projects/rev/projects/southern_co/scratch"
        master_sheet = "/shared-projects/rev/projects/southern_co/scratch/rev_inputs.xlsx"
        """

        self.module_paths = MODULE_CONFIG_PATHS
        self._expand_paths(master, proj_dir)
        self._sheets()
        self._set_paths()

    def __repr__(self):
        mpath = self.__dict__["master"]
        ppath = self.__dict__["proj_dir"]
        msg = "master=" + mpath + "  proj_dir=" + ppath
        msg = "<Paths " + msg + ">"
        return msg

    def _expand_paths(self, master, proj_dir):

        # Expand user or '.'
        proj_dir = os.path.expanduser(proj_dir)
        master = os.path.expanduser(master)
        self.proj_dir = os.path.abspath(proj_dir)
        self.master = os.path.abspath(master)

    def _set_path(self, paths, spath, module):

        # Assign a configuration file path to each module iteration
        scen = os.path.basename(spath)
        folder = self.module_paths[module]["folder"]
        file = self.module_paths[module]["file"]
        path = os.path.join(spath, folder, file)
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

    def _set_sam(self):
        """Set the sam paths separately."""




    def _sheets(self):
        
        # Set each master sheet as an attribute
        try:
            sheets = get_sheet(self.master)
        except:
            raise FileNotFoundError(Fore.RED + self.master + " not found." +
                                    Style.RESET_ALL)
        sheets = {s: get_sheet(self.master, s) for s in sheets}
        for s, df in sheets.items():
            setattr(self, s, df)


class Setup(Paths):
    def __init__(self, master, proj_dir):
        super().__init__(master, proj_dir)

    def __repr__(self):
        mpath = self.__dict__["master"]
        ppath = self.__dict__["proj_dir"]
        msg = "master=" + mpath + "  proj_dir=" + ppath
        msg = "<Setup " + msg + ">"
        return msg    

    

    def generation(self):
        """Create all generation configurations."""

        # Get paths dictionary
        paths = self.paths

        # Get scenarios
        for scen, spath in paths["scenarios"].items():
            print(scen, spath)
            path = paths["generation"][scen])
            
            config = "Get theses parameters from self.project_control maybe"
            write_config(config, path)


def setup_project(master, proj_dir):
    """Setup all configuration files for a project from a rev input file.
    
    Sample Arguments
    ----------------
    master = "/shared-projects/rev/projects/southern_co/scratch/rev_inputs.xlsx"
    proj_dir = "/shared-projects/rev/projects/southern_co/scratch"
    """

    # Initiate setup object
    setup = Setup(master, proj_dir)

    # Build everything
    setup.build_all()
    



@click.command()
@click.option("--file", "-f", default=None, help=FILE_HELP)
@click.option("--project_dir", "-d", default=".", help=PDIR_HELP)
@click.option("--template", "-t", is_flag=True, help=TEMPLATE_HELP)
def main(file, project_dir, template):
    """To start this will either wite a new rev input template or build the
    configurations from an existing one. It might also be useful to be able to
    set exectution parameters here."""

    # Expand project dir
    project_dir = os.path.expanduser(project_dir)
    project_dir = os.path.abspath(project_dir)

    # Write new template if asked
    if template:
        dst = os.path.join(project_dir, "rev_inputs.xlsx")
        write_template(dst)

    # If a template file is provided set everything up
    if file:
        print(Fore.GREEN + " setting up project in " + project_dir + " ..." + 
              Style.RESET_ALL)
        # setup_project(file, project_dir)

# if __name__ == "__main__":
#     main()
