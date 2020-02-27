#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checking all batch run logs for success.

Created on Wed Feb 19 07:52:30 2020

@author: twillia2
"""
import click
from glob import glob
import json
import os

FOLDER_HELP = ("Path to a folder with a completed set of batched reV runs. "
               "Defaults to current directory. (str)")
MODULE_HELP = ("The reV module logs to check: generation, collect, "
               "multi-year, aggregation, supply-curve, or rep-profiles")

MODULE_DICT = {"gen": "generation",
               "collect": "collect",
               "multi-year": "multi-year",
               "aggregation": "aggregation",
               "supply-curve": "supply-curve",
               "rep-profiles": "rep-profiles"}
CONFIG_DICT =  {"gen": "config_gen.json",
                "collect": "config_collect.json",
                "multi-year": "config_multi-year.json",
                "aggregation": "config_ag.son",
                "supply-curve": "config_supply-curve.json",
                "rep-profiles": "config_rep-profiles.json"}

@click.command()
@click.option("--folder", "-f", default=".", help=FOLDER_HELP)
@click.option("--module", "-m", default="gen", help=MODULE_HELP)
def main(folder, module):
    """
    revruns Batch Logs

    Check all logs for a reV module in a batched run directory.
    
    example:

        rrbatch_logs -f "." -m generation 
    """
    os.chdir(".")
    with open(CONFIG_DICT[module], "r") as file:
        config = json.load(file)
    module = MODULE_DICT[module]
    tech = config["technology"]
    batches = glob("{}_*".format(tech))
    for batch in batches:
        stat_file = "{}/{}_status.json".format(batch, batch)
        with open(stat_file, "r") as file:
            status = json.load(file)
            genstat = status[module]
            runkeys = [k for k in genstat.keys() if batch in k]
            failures = 0
            for k in runkeys:
                try:
                    assert genstat[k] != "failed"
                except AssertionError:
                    failures += 1
                    print("JOB " + k + " FAILED.")
    print("Logs checked with {} failures.".format(failures))

if __name__ == "__main__":
    main()