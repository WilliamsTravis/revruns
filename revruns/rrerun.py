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
               "'rep_profiles'. rrrerun will overwrite all results in the "
               "pipeline starting  at this point (the results of this module "
               "included).(str)")
MODULES = ['generation', 'collect', 'multi-year', 'aggregation',
           'supply-curve', 'rep-profiles']
MODULE_SHORTS = {"generation":"gen",
                 "collect": "collect",
                 "multi-year": "multi-year",
                 "aggregation": "_agg",
                 "supply-curve": "_sc",
                 "rep-profiles": "rep_profiles"}

@click.command()
@click.option("--folder", "-f", default=".", help=FOLDER_HELP)
@click.option("--module", "-m", required=True, help=MODULE_HELP)
def main(folder, module):
    """
    revrun Rerun

    Rerun reV from a specified point in the pipeline of a successful run.
    Once you have reconfigured the desired parameters, rrrerun will remove
    results from the specified module and all subsequent modules in the
    pipeline, remove their entries from the logs, and rerun reV.
    
    samples:
        folder = "/projects/rev/new_projects/ipm_solar"
        module = "rep_profiles"
    """

    # The module syntax can be easily confused
    if module == "ag" or module == "agg":
        module = "aggregation"
    if module == "multi_year" or module == "supply_curve":
        module = module.replace("_", "-")
    if module == "rep_profiles":
        module = module.replace("_", "-")

    # We need to expand absolute paths, so change to directory
    os.chdir(folder)

    # List of modules to remove
    remove_longs = MODULES[MODULES.index(module):]
    remove_shorts = [MODULE_SHORTS[m] for m in remove_longs]

    # read in the config_gen dictionary
    with open(os.path.join(folder, "config_gen.json")) as file:
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
        log_folder = config["directories"]["logging_directory"]
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
    initial_out = sp.check_output(["less", "nohup.out"])
    print(initial_out.decode())

if __name__ == "__main__":
    main()
