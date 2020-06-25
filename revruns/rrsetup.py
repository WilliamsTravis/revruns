# -*- coding: utf-8 -*-
"""Setup configuration files from a standard template.
"""

import os
import pkgutil
import shutil

import click
import h5py
import pandas as pd

from colorama import Fore, Style
from revruns.constants import ROOT, RESOURCE_DATASETS, TEMPLATES, SAM_TEMPLATES
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
AG_RESOLUTION = {
    "pv": 128,
    "wind": 64
}

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


def build_points(dataset, config, dst):
    """Build default project point files based on the resource."""

    # Get the data set path
    dpath = RESOURCE_DATASETS[dataset].format(2010)

    # Build the meta data frame
    with h5py.File(dpath, "r") as ds:
        points = pd.DataFrame(ds["meta"][:])

    # Clena this up a bit
    points["gid"] = points.index
    points["config"] = config
    points = points[["latidue", "longitude", "config"]]

    # Save
    points.to_csv(dst, index=False)


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
        print(Fore.YELLOW + dst + " exists. Choose a different path for "
              "the template." + Style.RESET_ALL)

def write_config(config_dict, path, verbose=False):
    """ Write a configuration dictionary to a json file."""

    # Write json to file
    with open(path, "w") as file:
        file.write(json.dumps(config_dict, indent=4))


class Paths:

    def __init__(self, master, proj_dir):
        """Initiate Paths object.

        proj_dir = "~/github/revruns/tests/project"
        master = "~/github/revruns/revruns/data/rev_inputs.xlsx"
        self = Paths(master, proj_dir)
        """

        self.module_paths = MODULE_CONFIG_PATHS
        self._expand_paths(master, proj_dir)
        self._sheets()
        self._set_paths()
        self._set_sam()

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
        scens = self.scenarios["scenario_name"].dropna().unique()
        for scen in scens:
            spath = os.path.join(self.proj_dir, scen)
            os.makedirs(spath, exist_ok=True)
            paths["scenarios"][scen] = spath

            for m in modules:
                paths = self._set_path(paths, spath, m)

        # Now project points
        pcontrol = self.project_control 
        scenarios = list(pcontrol)[1:]
        point_paths = {}
        for scen in scenarios:
            resources = pcontrol[pcontrol["parameter"] == "resource"]
            r = resources[scen].values[0]
            path = os.path.join(self.proj_dir, "project_points", r + ".csv")
            point_paths[scen] = path
        paths["points"] = point_paths

        # Set paths dictionary as attribute
        self.paths = paths

    def _set_sam(self):
        """Set the sam paths separately."""

        scens = self.scenarios
        names = scens["system_name"].dropna().unique()
        self.paths["sam"] = {}
        for n in names:
            path =  os.path.join(self.proj_dir, "sam_configs", n + ".json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.paths["sam"][n] = path

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
        """
        proj_dir = "~/github/revruns/tests"
        master = "~/github/revruns/revruns/data/rev_inputs.xlsx"
        self = Setup(master, proj_dir)
        """
        super().__init__(master, proj_dir)       

    def __repr__(self):
        mpath = self.__dict__["master"]
        ppath = self.__dict__["proj_dir"]
        msg = "master=" + mpath + "  proj_dir=" + ppath
        msg = "<Setup " + msg + ">"
        return msg    

    def generation(self):
        """Create all generation configurations."""

        # Get the needed data frames
        scendf = self.scenarios
        sysdf = self.system_configs
        projdf = self.project_control
        projdf = projdf.applymap(lambda x: x.replace(" ", ""))

        # Get scenarios
        for name, spath in self.paths["scenarios"].items():

            # Find the system name
            system = scendf["system_name"][scendf["scenario_name"] == name]
            system = system.iloc[0]

            # Create a system config dictionary
            sdf = sysdf[sysdf["system_name"] == system]
            sysdct = dict(zip(sdf["parameter"], sdf["value"]))

            # Create a proj control dictionary
            projdct = dict(zip(projdf["parameter"], projdf[name]))

            # Get configuration elements
            sam_files = self.paths["sam"][system]
            resource_files = RESOURCE_DATASETS[projdct["resource"]]
            output_request = projdct["output_request"].split(",")
            years = self.years[name].dropna().astype(int).to_list()
            tech = sysdct["compute_module"]
    
            # Create the point table
            point_path = self.paths["points"][name]
            if not os.path.exists(point_path):
                os.makedirs(os.path.dirname(point_path), exist_ok=True)
                build_points(resource_files, "default", point_path)

            # Replace template placholders
            config = TEMPLATES["gen"].copy()
            config["execution_control"]["allocation"] = projdct["allocation"]
            config["analysis_years"] = years
            config["output_request"] = output_request
            config["technology"] = tech
            config["sam_files"] = {system: sam_files}
            config["resource_file"] = resource_files
            config["project_points"] = point_path

            # This is the target path
            save = self.paths["generation"][name]
            write_config(config, save)

    def collect(self):
        """Create all collect configurations."""

        # Get the needed data frames
        projdf = self.project_control
        projdf = projdf.applymap(lambda x: x.replace(" ", ""))

        # Get scenarios
        for name, spath in self.paths["scenarios"].items():

            # Create a proj control dictionary
            projdct = dict(zip(projdf["parameter"], projdf[name]))
            output_request = projdct["output_request"].split(",")

            # Replace template placholders
            config = TEMPLATES["co"].copy()
            config["execution_control"]["allocation"] = projdct["allocation"]
            config["dsets"] = output_request

            # This is the target path
            save = self.paths["collect"][name]
            write_config(config, save)

    def multi_year(self):
        """Create all collect configurations."""

        # Get the needed data frames
        projdf = self.project_control
        projdf = projdf.applymap(lambda x: x.replace(" ", ""))

        # Get scenarios
        for name, spath in self.paths["scenarios"].items():

            # Create a proj control dictionary
            projdct = dict(zip(projdf["parameter"], projdf[name]))
            output_request = projdct["output_request"].split(",")

            # Replace template placholders
            config = TEMPLATES["my"].copy()
            config["execution_control"]["allocation"] = projdct["allocation"]
            config["groups"]["none"]["dsets"] = output_request

            # This is the target path
            save = self.paths["multi-year"][name]
            write_config(config, save)

    def pipeline(self):
        """Create a different pipeline for generation and aggregation modules.
        """

    def aggregation(self):
        """Create all collect configurations."""

        # Get the needed data frames
        scendf = self.scenarios
        sysdf = self.system_configs
        projdf = self.project_control
        exdf = self.exclusion_files
        projdf = projdf.applymap(lambda x: x.replace(" ", ""))
        for name, spath in self.paths["scenarios"].items():

            # Some items depend on how many years are run
            years = self.years[name].dropna().astype(int).to_list()
            if len(years) > 1:
                cf_dset = "cf_mean-means"
                lcoe_dset = "lcoe_fcr-means"
            else:
                cf_dset = "cf_mean"
                lcoe_dset = "lcoe_fcr"

            # Get a new template
            config = TEMPLATES["ag"].copy()
            config["execution_control"]["allocation"]
            config["cf_dset"] = cf_dset
            config["lcoe_dset"] = lcoe_dset

            # Exclusions
            layerdf = scendf[scendf["scenario_name"]== name]
            excls = layerdf["exclusion_name"][layerdf["scenario_name"] == name]
            excls = excls.dropna().values
            excl_dict = {}
            for e in excls:
                entry = {}
                exclude_values = exdf["exclude_values"][exdf["exclusion_name"] == e]
                entry["exclude_values"]
                
            

            # This is the target path
            save = self.paths["aggregation"][name]
            write_config(config, save)


    def sam(self):
        """Create all SAM configurations."""
        
        systems = self.system_configs
        for name in systems["system_name"].dropna().unique():

            # Reshape this
            sdf = systems[systems["system_name"] == name]
            sdct = dict(zip(sdf["parameter"], sdf["value"]))

            # The template depends on the module
            module = sdct["compute_module"]
            template = SAM_TEMPLATES[module].copy()
            for par, val in sdct.items():
                template[par] = val

            # If this is windspeed it needs a power curve
            if "wind_turbine_powercurve_powerout" in sdct:
                ws = self.power_curves["wind_speed"].to_list()
                pc = self.power_curves[name].to_list()
                template["wind_turbine_powercurve_windspeeds"] = ws
                template["wind_turbine_powercurve_powerout"] = pc

            # Not sure if "PLACEHOLDER" could break something
            config = {p: v for p, v in template.items() if v != "PLACEHOLDER"}

            # Save
            save = self.paths["sam"][name]
            write_config(config, save)

    def build(self):
        """Run all submodules and build configurations."""

        self.sam()
        self.generation()
        self.collect()
        self.multi_year()
        
        self.aggregation()
        self.supply_curve()
        self.rep_profiles()
        self.pipeline()


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
    setup.build()
    



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
        setup_project(file, project_dir)
        print(Fore.GREEN + "Project set up in " + project_dir + "." + 
              Style.RESET_ALL)


# if __name__ == "__main__":
#     main()
