# -*- coding: utf-8 -*-
"""
Run this to setup configuration files as templates for a reV run.
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





MODULE_NAMES = {
    "gen": "generation",
    "co": "collect",
    "my": "multi-year",
    "ag": "supply-curve-aggregation",
    "sc": "supply-curve",
    "rp": "rep-profiles",
    "ba": "batch"
}

DEFAULT_PATHS = {
    "gen": "./config_generation.json",
    "co": "./config_collect.json",
    "my": "./config_multi-year.json",
    "ag": "./config_aggregation.json",
    "sc": "./config_supply-curve.json",
    "rp": "./config_rep-profiles.json",
    "ba": "./config_batch.json",
    "points": "./project_points/project_points.csv",
    "sam": "./sam_configs/sam.json"
}

# Move to main module script
DEFAULT_YEARS = {
    "csp": [y for y in range(1998, 2019)],
    "pvwattsv5": [y for y in range(1998, 2019)],
    "pvwattsv5": [y for y in range(1998, 2019)],
    "windpower": [y for y in range(2007, 2014)]
}

PIPELINE_TEMPLATE =  {
    "logging": {
        "log_level": "INFO"
    },
    "pipeline": []
}

@click.command()
@click.option("--generation", "-gen", is_flag=True, help=AGGREGATION_HELP)
@click.option("--collect", "-co", is_flag=True, help=COLLECT_HELP)
@click.option("--multiyear", "-my", is_flag=True, help=MULTIYEAR_HELP)
@click.option("--aggregation", "-ag", is_flag=True, help=AGGREGATION_HELP)
@click.option("--supplycurve", "-sc", is_flag=True, help=SUPPLYCURVE_HELP)
@click.option("--repprofiles", "-rp", is_flag=True, help=REPPROFILES_HELP)
@click.option("--batch", "-ba", is_flag=True, help=BATCH_HELP)
@click.option("--tech", "-t", default="pvwattsv5", help=GEN_HELP)
@click.option("--full", "-f", is_flag=True, help=FULL_HELP)
@click.option("--allocation", "-alloc", default="PLACEHOLDER", help=ALLOC_HELP)
@click.option("--output_dir", "-outdir", default="./", help=OUTDIR_HELP)
@click.option("--log_dir", "-logdir", default="./logs", help=LOGDIR_HELP)
@click.option("--verbose", "-v", is_flag=True)
def main(generation, collect, multiyear, aggregation, supplycurve, repprofiles,
         batch, tech, full, allocation, output_dir, log_dir,
         verbose):
    """Write template configuration json files for each reV module specified.
    Additionaly options will set up template sam_files and default project
    point files. Files will be written to your current directory.

    In the output configuration jsons the term 'PLACEHOLDER' is SET for values
    that require user inputs. Other inputs have defaults that may or may not 
    be appropriate. To use all available points for the specified generator,
    provide the '--all-points' or '-ap' flag. To use all available years for a
    specified generator, provide the '--all-years' or '-ay' flag.

    Sample Arguments
    ----------------
        generation = False
        collect = False
        multiyear = False
        aggregation = True
        supplycurve = True
        repprofiles = True
        batch = False
        tech = "pvwattsv5"
        allpoints = True
        allyears = True
        full = False
        verbose = True
    """

    # Get requested modules as strings
    strings = np.array(["gen", "co", "my", "ag", "sc", "rp", "ba"])
    requested = np.array([generation, collect, multiyear, aggregation,
                          supplycurve, repprofiles, batch])

    # Convert module selections from booleans to key strings
    if full:
        modules = strings
    else:
        modules = strings[requested]

    # Retrieve the template objects
    templates = {m: TEMPLATES[m] for m in modules}

    # Assign all years (easier to subtract than add), and add allocation
    years = DEFAULT_YEARS[tech]
    for m in templates.keys():
        if "analysis_years" in templates[m]:
            templates[m]["analysis_years"] = years
        if "execution_control" in templates[m]:
            templates[m]["execution_control"]["allocation"] = allocation
        if "directories" in templates[m]:
            templates[m]["directories"]["output_directory"] = output_dir
            templates[m]["directories"]["log_directories"] = log_dir

    # Assign all points if specified  <---------------------------------------- Not done.
    os.makedirs("./project_points", exist_ok=True)

    # Write sam template:
    os.makedirs("./sam_configs", exist_ok=True)
    sam_template = SAM_TEMPLATES[tech]

    # Write json file for each template
    for m in templates.keys():
        config = templates[m]
        path = DEFAULT_PATHS[m]
        write_config(config, path, verbose)

    # If there are more than one module, write pipeline configuration
    if len(modules) > 1:
        pipeline = PIPELINE_TEMPLATE.copy()
        for m in modules:
            dict = {MODULE_NAMES[m]: DEFAULT_PATHS[m]}
            pipeline["pipeline"].append(dict)
        write_config(pipeline, "./config_pipeline.json", verbose)


if "__name__" == "__main__":
    main()

