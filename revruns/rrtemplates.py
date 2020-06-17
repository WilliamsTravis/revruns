# -*- coding: utf-8 -*-
"""
Run this to setup configuration files as templates for a reV run.
"""

import os

import click
import numpy as np
import reV

from revruns import RESOURCE_DATASETS, TEMPLATES, SAM_TEMPLATES, write_config


# Help printouts
AGGREGATION_HELP = ("Setup the `aggregation` module configuration template "
                    "and all of its required module templates. "
                    "This module will resample the outputs to a coarser "
                    "resolution and group results by specified regions "
                    "after excluding specified areas and return "
                    "a csv of results for each coordinate. (boolean)")
ALLOC_HELP = ("Eagle account to use.")
ALLPOINTS_HELP = ("Use all available coordinates for the specified generator. "
                  "(boolean)")
BATCH_HELP = ("Set up the `batch` module configuration template. This"
              "module will run reV with a sequence of arguments or argument"
              "combinations. (boolean)")
COLLECT_HELP = ("Setup the `collect` module configuration template and all "
                "of its required module templates. When "
                "you run the generation module with multiple nodes, "
                "multiple files will be written for each year. This will "
                "combine these into single yearly files HDF5 files. (boolean)")
FULL_HELP = ("Setup the full pipeline of reV module templates, from "
             "`generation` to `rep-profiles`. (boolean)")
GEN_HELP = ("This is the generator type to simulate using the Systems "
            "Advisor Model (SAM). Defaults to pvwattsv5. Options include: "
            "\pvwattsv5 or pvwattsv7: Photovoltaic \nwindpower: Wind Turbines "
            "\ncsp: Concentrating Solar Power.")
LOGDIR_HELP = ("Logging directory. Defaults to './logs'")
MULTIYEAR_HELP = ("Setup the `multi-year` module configuration template ."
                  "and all of its required module templates. "
                  "This module will combine yearly HDF5 files into one "
                  "large HDF5 file. (boolean)")
OUTDIR_HELP = ("Output directory. This is also used as the SLURM job name "
               "in the queue. Defaults to './'")
REPPROFILES_HELP = ("Setup the `rep-profiles` module configuration "
                    "template and all of its required module templates. "
                    "This module will select a number of "
                    "representative profiles within each specified region "
                    "according to a specified measure of similarity to "
                    "all profiles within a region. (boolean)")
SUPPLYCURVE_HELP = ("Setup the `supply-curve` module configuration "
                    "template and all of its required module templates. "
                    "This module will calculate supply curves "
                    "for each aggregated coordinate and return a csv. "
                    "(boolean)")

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

