# -*- coding: utf-8 -*-
"""Reconfigure a reV run to rerun from a specified point in the pipeline."""

import json
import os
import subprocess as sp
import time

from glob import glob

import click

from revruns.rrlogs import find_status


FOLDER_HELP = ("A folder containing configurations and results "
               "from a sucessful reV run. Defaults to current directory. "
               "(str)")
MODULE_HELP = ("A module in the reV pipeline. Options include 'generation', "
               "'collect', 'multi-year', 'aggregation', 'supply-curve', or "
               "'rep_profiles'. rrerun will overwrite all results in the "
               "pipeline after this step (the results of this module "
               "not included).(str)")
RUN_HELP = ("Rerun reV pipeline. Without this, the logging and status "
            "files will be updated and you must rerun manually. (Boolean)")
MODULES = ["generation", "collect", "econ", "offshore", "multi-year",
           "supply-curve-aggregation", "supply-curve", "rep-profiles", "qa-qc"]
MODULE_SHORTS = {"generation": "gen",
                 "collect": "collect",
                 "econ": "econ",
                 "offshore": "offshore",
                 "multi-year": "multi-year",
                 "supply-curve-aggregation": "_agg",
                 "supply-curve": "_sc",
                 "rep-profiles": "rep_profiles",
                 "qa-qc": "qa-qc"}


def find_pipeline(folder):
    """Find the pipeline configuration."""
    path = glob(os.path.join(folder, "*pipeline*json"))[0]
    with open(path, "r") as file:
        pipeline = json.load(file)
    return path, pipeline["pipeline"]


@click.command()
@click.option("--folder", "-f", default=".", help=FOLDER_HELP)
@click.option("--module", "-m", required=True, help=MODULE_HELP)
@click.option("--run", "-r", is_flag=True, help=RUN_HELP)
def main(folder, module, run):
    """
    REVRUNS - RERUN.

    Reconfigure reV to run from a specified point in the pipeline of a
    successful run.

    Once you have reconfigured the desired parameters, rrerun will remove
    results from the specified module along with all subsequent modules in the
    pipeline, and remove their entries from the logging and status files. If
    '--run' is specified, this will also resubmit the pipeline.

    Note that you can only rerun modules in one pipeline at a time. So, if
    modules in one pipeline depend on outputs from another, you must rerun the
    other first if you want to start over that far back.
    """
    # The module syntax can be easily confused
    ag_spellings = ["ag", "agg", "aggregation", "supply-curve_aggregation"]
    if any([m == module for m in ag_spellings]):
        module = "supply-curve-aggregation"
    if module == "multi_year" or module == "supply_curve":
        module = module.replace("_", "-")
    if module == "rep_profiles":
        module = module.replace("_", "-")

    # change to directory
    os.chdir(folder)

    # Find the various logging documents
    ppath, pipeline = find_pipeline(folder)
    spath, status = find_status(folder)

    # List the files to keep and drip
    ran_modules = list(status.keys())
    keepers = ran_modules[: ran_modules.index(module) + 1]

    # Remove everything in status after the module
    status = {k: v for k, v in status.items() if k in keepers}

    # Rewrite status json
    with open(spath, "w") as sfile:
        sfile.write(json.dumps(status, indent=4))

    # Rerun pipeline
    if run:
        if os.path.exists("nohup.out"):
            os.remove("nohup.out")
        sp.Popen(["nohup", "reV", "-c", "config_pipeline.json", "pipeline",
                  "--monitor"],
                 stdout=open('pipeline.out', 'a'),
                 stderr=open('pipeline.out', 'a'),
                 preexec_fn=os.setpgrp)
        time.sleep(2)
        initial_out = sp.check_output(["cat", "nohup.out"])
        print(initial_out.decode())


if __name__ == "__main__":
    main()
