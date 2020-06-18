# -*- coding: utf-8 -*-
"""
Interacting with reV output logs.

Created on Wed Jun 17 9:00:00 2020

@author: twillia2
"""

import json
import os

from glob import glob

import click
import pandas as pd

from colorama import Fore, Style



FOLDER_HELP = ("Path to a folder with a completed set of batched reV runs. "
               "Defaults to current directory. (str)")
MODULE_HELP = ("The reV module logs to check. Defaults to all modules: gen, "
               "collect, multi-year, aggregation, supply-curve, or "
               "rep-profiles")
CHECK_HELP = ("The type of check to perform. Option include 'failure' (print "
            "which jobs failed), 'success' (print which jobs finished), and "
            "'pending' (print which jobs have neither of the other two "
            "statuses). Defaults to failure.")
ERROR_HELP = ("A job ID. This will print the error log of a job.")
OUT_HELP = ("A job ID. This will print the standard output log of a job.")
MODULE_NAMES = {
    "gen": "generation",
    "collect": "collect",
    "multi-year": "multi-year",
    "aggregation": "supply-curve-aggregation",
    "supply-curve": "supply-curve",
    "rep-profiles": "rep-profiles"
}
CONFIG_DICT = {
    "gen": "config_gen.json",
    "collect": "config_collect.json",
    "multi-year": "config_multi-year.json",
    "aggregation": "config_aggregation.son",
    "supply-curve": "config_supply-curve.json",
    "rep-profiles": "config_rep-profiles.json"
}


def find_logs(folder):
    """Find the log folders based on configs present in folder. Assumes  # <--- Create a set of dummy jsons to see if this works
    only one log directory per folder."""


    # if there is a log directory directly in this folder use that
    contents = glob(os.path.join(folder, "*"))
    possibles = [c for c in contents if "log" in c]
    if len(possibles) == 1:
        logdir = os.path.join(folder, possibles[0])
        return logdir

    # If that didn't work check the config files
    config_files = glob(os.path.join(folder, "*.json"))
    logdir = None
    try:
        for file in config_files:
            if not logdir:
                
                # The file might not open
                try:
                    config = json.load(open(file, "r"))
                except:
                    pass
    
                # The directory might be named differently
                try:
                    logdir = config["directories"]["log_directory"]
                except KeyError:
                    logdir = config["directories"]["logging_directory"]
                finally:
                    pass
    except:
        print("Could not find log directory")
        raise

    # Expand log directory
    if logdir[0] == ".":
        logdir = logdir[2:]  # "it will have a / as well
        logdir = os.path.join(folder, logdir)
    logdir = os.path.expanduser(logdir)

    return logdir


def find_outputs(folder):
    """Find the output directory based on configs present in folder. Assumes
    only one log directory per folder."""

    # Check each json till you find it
    config_files = glob(os.path.join(folder, "*.json"))
    outdir = None
    try:
        for file in config_files:
            if not outdir:
                
                # The file might not open
                try:
                    config = json.load(open(file, "r"))
                except:
                    pass
    
                # The directory might be named differently
                try:
                    outdir = config["directories"]["output_directory"]
                except KeyError:
                    pass
    except:
        print("Could not find 'output_directory'")
        raise
    
    # Expand log directory
    if outdir[0] == ".":
        outdir = outdir[2:]  # "it will have a / as well
        outdir = os.path.join(folder, outdir)
    outdir = os.path.expanduser(outdir)

    return outdir


def find_status(folder):
    """Find the job status json."""

    # Find output directory
    try:
        files = glob(os.path.join(folder, "*.json"))
        file = [f for f in files if "_status.json" in f][0]
    except:
        outdir = find_outputs(folder)
        files = glob(os.path.join(outdir, "*.json"))
        file = [f for f in files if "_status.json" in f][0]

    # Return the dictionary
    with open(file, "r") as f:
        status = json.load(f)

    return status


def module_status_dataframe(status, module="gen"):
    """Convert the status entry for a module to a dataframe."""

    # Get the module key
    mkey = MODULE_NAMES[module]

    # Get the module entry
    mstatus = status[mkey]

    # The first entry is the pipeline index
    mindex = mstatus["pipeline_index"]

    # The rest is another dictionary for each sub job
    del mstatus["pipeline_index"]
    mdf = pd.DataFrame(mstatus).T
    mdf["pipeline_index"] = mindex

    return mdf


def status_dataframe(folder, module=None):
    """Convert the status entry for a module or an enitre project to a
    dataframe."""

    # Get the status dictionary
    status = find_status(folder)

    # If just one module
    if module:
        try:
            df = module_status_dataframe(status, module)
        except KeyError:
            print(module + " not found in status file.\n")
            raise

    # Else, create a single data frame with everything
    else:
        modules = status.keys()
        names_modules = {v: k for k, v in MODULE_NAMES.items()}
        dfs = []
        for m in modules:
            m = names_modules[m]
            dfs.append(module_status_dataframe(status, m))
        df = pd.concat(dfs, sort=False)

    return df


def success(folder, module):
    """Print status of each job for a module run."""


@click.command()
@click.option("--folder", "-f", default=".", help=FOLDER_HELP)
@click.option("--module", "-m", default=None, help=MODULE_HELP)
@click.option("--check", "-c", default=None, help=CHECK_HELP)
@click.option("--error", "-e", default=None, help=ERROR_HELP)
@click.option("--out", "-o", default=None, help=OUT_HELP)
def main(folder, module, check, error, out):
    """
    revruns Batch Logs

    Check logs for a reV module in a run directory. Assumes certain standard  # <--- add a blip about naming conventions.
    naming conventions.

    folder = "~/github/revruns/tests/data"
    module = "aggregation"
    error = 3275897
    out = 3275897
    """

    # Expand folder path
    folder = os.path.expanduser(folder)
    folder = os.path.abspath(folder)

    # Open the gen config file to determine batch names
    logdir = find_logs(folder)

    # Convert module status to data frame
    status_df = status_dataframe(folder, module)
    status_df["job_name"] = status_df.index
    print_df = status_df[['job_id', 'job_status', 'pipeline_index', 'runtime']]

    # Now return the requested return type
    if check:
        if check == "failure":
            print_df = print_df[print_df["job_status"] == "failed"]
        elif check == "success":
            print_df = print_df[print_df["job_status"] == "successful"]
        elif check == "pending":
            print_df = print_df[~print_df.isin(["successful", "failed"])]

    if not error and not out:
        print(print_df)

    if error:
        errors = glob(os.path.join(logdir, "stdout", "*e"))
        elog = [e for e in errors if str(error) in e][0]
        with open(elog, "r") as file:
            elines = file.readlines()
            if len(elines) > 20:
                print("  \n   ...   \n")
            for e in elines[-20:]:
                print(e)
            print(Fore.RED + "\n error file: " + elog + Style.RESET_ALL) 
    if out:
        outs = glob(os.path.join(logdir, "stdout", "*o"))
        olog = [o for o in outs if str(out) in o][0]
        with open(olog, "r") as file:
            olines = file.readlines()
            if len(olines) > 20:
                print("  \n   ...   \n")
            for o in olines[-20:]:
                print(o)
            print(Fore.GREEN + "\n stdout file: " + olog + Style.RESET_ALL) 

    # # Not done after here...
    # with open(CONFIG_DICT["gen"], "r") as file:
    #     config = json.load(file)
    # tech = config["technology"]
    # module_name = MODULE_DICT[module]

    # # List all batch folders and check that they exist
    # batches = glob("{}_*".format(tech))
    # try:
    #     assert batches
    # except AssertionError:
    #     print(Fore.RED + "No batch runs found." + Style.RESET_ALL)
    #     return

    # # Check for "non-successes"
    # failures = 0
    # for batch in batches:
    #     stat_file = "{0}/{0}_status.json".format(batch)

    #     # Check that the file 
    #     try:
    #         with open(stat_file, "r") as file:
    #             log = json.load(file)
    #             genstat = log[module_name]
    #             runkeys = [k for k in genstat.keys() if batch in k]
                
    #             try:
    #                 assert runkeys
    #             except AssertionError:
    #                 print(Fore.RED + "No status found for " + batch +
    #                       Style.RESET_ALL)
    #                 failures += 1

    #             for k in runkeys:
    #                 try:
    #                     status = genstat[k]["job_status"]
    #                     assert status == "successful"  # <--------------------- A job_status for the last module of a pipeline might not update to successful, even if it is.
    #                 except AssertionError:
    #                     failures += 1
    #                     if status == "submitted":
    #                         print("Job '" + k + "' may or may not be fine. " +
    #                               "Status: " + Fore.YELLOW + status)
    #                     else:
    #                         print("Job status '" + k + "': " + Fore.RED +
    #                               status + ".")
    #                     print(Style.RESET_ALL)
    #     except FileNotFoundError:
    #         failures += 1
    #         print(Fore.RED)
    #         print("No log file found for '" + batch + "'.")
    #         print(Style.RESET_ALL)

    # # Print final count. What else would be useful?
    # if failures == 0:
    #     print(Fore.GREEN)
    # else:
    #     print(Fore.RED)
    # print("Logs checked with {} incompletions.".format(failures))

    # # Reset terminal colors
    # print(Style.RESET_ALL)


if __name__ == "__main__":
    main()
