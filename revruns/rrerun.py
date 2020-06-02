#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reconfigure a reV run to rerun from a specified points in the pipeline, instead
of manually rewriting the logs. 

Created on Wed Feb 19 13:07:33 2020

@author: twillia2
"""

import click
from glob import glob
import json
import os
import subprocess as sp
import time

FOLDER_HELP = ("A folder containing configurations, point data, and results "
               "from a sucessful reV run. Defaults to current directory. "
               "(str)")
MODULE_HELP = ("A module in the reV pipeline. Options include 'generation', "
               "'collect', 'multi-year', 'aggregation', 'supply-curve', or "
               "'rep_profiles'. rrerun will overwrite all results in the "
               "pipeline starting  at this point (the results of this module "
               "included).(str)")
MODULES = ['generation', 'collect', 'multi-year', 'supply-curve-aggregation',
           'supply-curve', 'rep-profiles']
MODULE_SHORTS = {"generation":"gen",
                 "collect": "collect",
                 "multi-year": "multi-year",
                 "supply-curve-aggregation": "_agg",
                 "supply-curve": "_sc",
                 "rep-profiles": "rep_profiles"}

@click.command()
@click.option("--folder", "-f", default=".", help=FOLDER_HELP)
@click.option("--module", "-m", required=True, help=MODULE_HELP)
def main(folder, module):
    """
    revrun Rerun

    Rerun reV from a specified point in the pipeline of a successful run.
    Once you have reconfigured the desired parameters, rrerun will remove
    results from the specified module and all subsequent modules in the
    pipeline, remove their entries from the logs, and rerun reV.

    Note that you can only rerun modules in one pipeline at a time. So, if
    modules in one pipeline depend on outputs from another, you must rerun the
    other first if you want to start over that far back.
    
    samples:
        folder = "/shared-projects/rev/projects/india/forecast/wind"
        module = "aggregation"
    """

    # The module syntax can be easily confused
    ag_spellings = ["ag", "agg", "aggregation", "supply-curve_aggregation"]
    if any([m == module for m in ag_spellings]):
        module = "supply-curve-aggregation"
    if module == "multi_year" or module == "supply_curve":
        module = module.replace("_", "-")
    if module == "rep_profiles":
        module = module.replace("_", "-")

    # We need to expand absolute paths, so change to directory
    os.chdir(folder)

    # List of modules to remove
    remove_longs = MODULES[MODULES.index(module):]
    remove_shorts = [MODULE_SHORTS[m] for m in remove_longs]

    # Open the pipeline configuration we need a few things from it
    with open(os.path.join(folder, "config_pipeline.json")) as file:
        config_pipeline = json.load(file)

    # Configuration spelling is different from status spelling for one module
    if module == "supply-curve-aggregation":
        module2 = "supply-curve-aggregation"
    else:
        module2 = module

    # If the requested module isn't in the pipeline, raise error
    pipeline_modules = [list(d.keys())[0] for d in config_pipeline["pipeline"]]
    if module2 not in pipeline_modules:
        raise KeyError(module + " is not in the chosen pipeline.") 

    # If the requested module is not in the pipeline raise an error
    module_index = pipeline_modules.index(module2)
    config_path = config_pipeline["pipeline"][module_index][module2]
    config_path = os.path.abspath(config_path)
    with open(config_path) as file:
        config = json.load(file)

    # Remove previous results
    output_folder = config["directories"]["output_directory"]
    output_folder = os.path.abspath(output_folder)
    outputs_h5 = glob(os.path.join(output_folder, "*h5"))
    outputs_csv = glob(os.path.join(output_folder, "*csv"))
    outputs = outputs_h5 + outputs_csv
    for o in outputs:
        if any([m in os.path.basename(o) for m in remove_shorts]):
            os.remove(o)

    # Remove previous file logs
    try:
        log_folder = config["directories"]["log_directory"]
    except KeyError:
        try:
            log_folder = config["directories"]["logging_directory"]
        except KeyError:
            log_folder = config["directories"]["logging_directories"]
    log_folder = os.path.abspath(log_folder)
    logs = glob(os.path.join(log_folder, "*log"))
    for log in logs:
        if any([m in os.path.basename(log) for m in remove_shorts]):
            os.remove(log)

    # Remove previous stdout logs
    stdout_folder = os.path.join(log_folder, "stdout")
    outs = glob(os.path.join(stdout_folder, "*o"))
    errors = glob(os.path.join(stdout_folder, "*e"))
    stdouts = outs + errors
    for s in stdouts:
        if any([m in os.path.basename(s) for m in remove_shorts]):
            os.remove(s)

    # Rewrite the old status log
    outname = os.path.basename(output_folder)
    status_file = os.path.join(output_folder, outname + "_status.json")
    with open(status_file, "r") as file:
        status = json.load(file)
    for key in remove_longs:
        status[key] = {"pipeline_index": status[key]["pipeline_index"]}
    with open(status_file, "w") as file:
        file.write(json.dumps(status))

    # Rerun pipeline
    if os.path.exists("nohup.out"):
        os.remove("nohup.out")
    sp.Popen(["nohup", "reV", "-c", "config_pipeline.json", "pipeline",
              "--monitor"],
             stdout=open('/dev/null', 'w'),
             stderr=open('nohup.out', 'a'),
             preexec_fn=os.setpgrp)
    time.sleep(2)
    initial_out = sp.check_output(["cat", "nohup.out"])
    print(initial_out.decode())

if __name__ == "__main__":
    main()
