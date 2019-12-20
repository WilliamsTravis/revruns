# -*- coding: utf-8 -*-
"""
Run this to setup configuration files as templates for a reV run.
"""
import click
import numpy as np
import os
import reV
from revruns import RESOURCE_DATASETS, TEMPLATES, write_config

# Help printouts
AGGREGATION_HELP = ("Setup the `aggregation` module configuration template " +
                    "and all of its required module templates. " +
                    "This module will resample the outputs to a coarser " +
                    "resolution and group results by specified regions " +
                    "after excluding specified areas and return " +
                    "a csv of results for each coordinate. (boolean)")
ALLPOINTS_HELP = ("Use all available years for the specified generator. "+
                  "(boolean)")
ALLYEARS_HELP = ("Use all available coordinates for the specified " +
                 "generator. (boolean)")
BATCH_HELP = ("Set up the `batch` module configuration template. This" +
              "module will run reV with a sequence of arguments or argument" +
              "combinations. (boolean)")
COLLECT_HELP = ("Setup the `collect` module configuration template and all " +
                "of its required module templates. When " +
                "you run the generation module with multiple nodes, " +
                "multiple files will be written for each year. This will " +
                "combine these into single yearly files HDF5 files. (boolean)")
FULL_HELP = ("Setup the full pipeline of reV module templates, from " +
             "`generation` to `rep-profiles`. (boolean)")
GEN_HELP = ("This is the generator type to simulate using the Systems " +
            "Advisor Model (SAM). Defaults to pv. Options include: \npv: " +
            "Photovoltaic \nwind: Wind Turbines \ncsp: Concentrating Solar " +
            "Power.")
MULTIYEAR_HELP = ("Setup the `multi-year` module configuration template ." +
                  "and all of its required module templates. " +
                  "This module will combine yearly HDF5 files into one " +
                  "large HDF5 file. (boolean)")
REPPROFILES_HELP = ("Setup the `rep-profiles` module configuration " +
                    "template and all of its required module templates. " +
                    "This module will select a number of " +
                    "representative profiles within each specified region " +
                    "according to a specified measure of similarity to " +
                    "all profiles within a region. (boolean)")
SUPPLYCURVE_HELP = ("Setup the `supply-curve` module configuration " +
                    "template and all of its required module templates. " +
                    "This module will calculate supply curves " +
                    "for each aggregated coordinate and return a csv. "+
                    "(boolean)")

MODULE_NAMES = {
    "ag": "aggregation",
    "ba": "batch",
    "co": "collect",
    "gen": "generation",
    "my": "multi-year",
    "rp": "rep-profiles",
    "sc": "supply-curve"
}

DEFAULT_PATHS = {
    "ag": "./config_aggregation.json",
    "ba": "./config_batch.json",
    "co": "./config_collect.json",
    "gen": "./config_generation.json",
    "my": "./config_multi-year.json",
    "points": "./project_points/project_points.csv",
    "rp": "./config_rep-profiles.json",
    "sam": "./sam_configs/sam.json",
    "sc": "./config_supply-curve.json"
}

DEFAULT_YEARS = {
    "csp": [y for y in range(1998, 2019)],
    "pv": [y for y in range(1998, 2019)],
    "wind": [y for y in range(2007, 2014)]
}

PIPELINE_TEMPLATE =  {
    "logging": {
        "log_level": "INFO"
    },
    "pipeline": [
        {
            "generation": "./config_gen.json"
        }
    ]
}

@click.command()
@click.option("--tech", "-t", default="pv", help=GEN_HELP)
@click.option("--aggregation", "-ag", is_flag=True, help=AGGREGATION_HELP)
@click.option("--allpoints", "-ap", is_flag=True, help=ALLPOINTS_HELP)
@click.option("--allyears", "-a", is_flag=True, help=ALLYEARS_HELP)
@click.option("--batch", "-ba", is_flag=True, help=BATCH_HELP)
@click.option("--collect", "-co", is_flag=True, help=COLLECT_HELP)
@click.option("--full", "-f", is_flag=True, help=FULL_HELP)
@click.option("--multiyear", "-my", is_flag=True, help=MULTIYEAR_HELP)
@click.option("--repprofiles", "-rp", is_flag=True, help=REPPROFILES_HELP)
@click.option("--supplycurve", "-sc", is_flag=True, help=SUPPLYCURVE_HELP)
@click.option("--verbose", "-v", is_flag=True)
def main(tech, aggregation, allpoints, allyears, batch, collect, full,
         multiyear, repprofiles, supplycurve, verbose):
    """Write template configuration json files for each reV module specified.
    Generation configuration templates are written automatically. The term
    'placeholder' is included for values that require user inputs. Some inputs
    are inferred from the '--generator' value. To use all available points for
    the specified generator, provide the '--all-points' or '-ap' flag. To use
    all available years for a specified generator, provide the '--all-years' or
    '-ay' flag.
    """

    # All possible module key strings
    strings = np.array(["gen", "co", "my", "ag", "sc", "rp"])

    # Convert module selections from booleans to key strings
    if full:
        modules = strings
    else:
        # They might try to input more than one module flag
        booleans = [True, collect, multiyear, aggregation, supplycurve,
                    repprofiles]

        # So, if they do, use the module farthest down the pipeline
        selected_modules = strings[booleans]
        last_module = selected_modules[-1]
        module_index = np.where(strings == last_module)[0][0]
        modules = strings[:module_index + 1]

    # Retrieve the template objects
    templates = {m: TEMPLATES[m] for m in modules}

    # Assign all years if specified
    if allyears:
        years = DEFAULT_YEARS[tech]
        for m in templates.keys():
            if "project_control" in templates[m].keys():
                if "analysis_years" in templates[m]["project_control"].keys():
                    templates[m]["project_control"]["analysis_years"] = years

    # Assign all points if specified
    os.makedirs("./project_points", exist_ok=True)
    # if allpoints:

    # Write sam template:
    os.makedirs("./sam_configs", exist_ok=True)

    # Write json file for each template
    for m in templates.keys():
        config = templates[m]
        path = DEFAULT_PATHS[m]
        write_config(config, path, verbose)

    # The pipe line configuration will always be written
    pipeline = PIPELINE_TEMPLATE.copy()
    for m in modules:
        dict = {MODULE_NAMES[m]: DEFAULT_PATHS[m]}
        pipeline["pipeline"].append(dict)
    write_config(pipeline, "./config_pipeline.json", verbose)

    # If batch is requested, write its template file
    if batch:
        config = templates["ba"]
        path = DEFAULT_PATHS["ba"]
        write_config(config, path, verbose)


if "__name__" == "__main__":
    main()
