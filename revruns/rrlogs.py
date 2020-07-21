# -*- coding: utf-8 -*-
"""Check with reV status, output, and error logs.
"""

import getpass
import json
import os
import subprocess as sp
import warnings

from glob import glob

import click
import numpy as np
import pandas as pd

from colorama import Fore, Style
from pandas.core.common import SettingWithCopyWarning
from tabulate import tabulate

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


FOLDER_HELP = ("Path to a folder with a completed set of batched reV runs. "
               "Defaults to current directory. (str)")
USER_HELP = ("Unix username. If another user is running a job in the current "
             "pipeline, leaving this blank might result in out-of-date "
             "values for the job status and will not display current runtimes."
             " Defaults to current user. (str)")
MODULE_HELP = ("The reV module logs to check. Defaults to all modules: gen, "
               "collect, multi-year, aggregation, supply-curve, or "
               "rep-profiles")
CHECK_HELP = ("The type of check to perform. Option include 'failure' (print "
            "which jobs failed), 'success' (print which jobs finished), and "
            "'pending' (print which jobs have neither of the other two "
            "statuses). Defaults to failure.")
ERROR_HELP = ("A job ID. This will print the first 20 lines of the error log "
              "of a job.")
OUT_HELP = ("A job ID. This will print the first 20 lines of the standard "
            "output log of a job.")
MODULE_NAMES = {
    "gen": "generation",
    "collect": "collect",
    "econ": "econ",
    "offshore": "offshore",
    "multi-year": "multi-year",
    "aggregation": "supply-curve-aggregation",
    "supply-curve": "supply-curve",
    "rep-profiles": "rep-profiles",
    "qaqc": "qa-qc"
}
CONFIG_DICT = {
    "gen": "config_gen.json",
    "collect": "config_collect.json",
    "econ": "config_econ.json",
    "offshore": "config_offshore.json",
    "multi-year": "config_multi-year.json",
    "aggregation": "config_aggregation.son",
    "supply-curve": "config_supply-curve.json",
    "rep-profiles": "config_rep-profiles.json",
    "qaqc": "config_qaqc.json"
}
FAILURE_STRINGS = ["failure", "fail", "failed", "f"]
SUCCESS_STRINGS = ["successful", "success", "s"]
PENDING_STRINGS = ["pending", "pend", "p"]
RUNNING_STRINGS = ["running", "run", "r"]
SUBMITTED_STRINGS = ["submitted", "submit", "sb"]
UNSUBMITTED_STRINGS = ["unsubmitted", "unsubmit", "u"]


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

    # Fix artifacts
    status = fix_status(status)

    return file, status


def fix_status(status):
    """Using different versions of reV can result in problematic artifacts."""

    # Aggregation vs Supply-Curve-Aggregation
    if "aggregation" in status and "supply-curve-aggregation" in status:
        ag = status["aggregation"]
        scag = status["supply-curve-aggregation"]
    
        if len(scag) > len(ag):
            del status["aggregation"]
        else:
            status["supply-curve-aggregation"] = status["aggregation"]
            del status["aggregation"]
    elif "aggregation" in status:
            status["supply-curve-aggregation"] = status["aggregation"]
            del status["aggregation"]

    return status

   
def get_squeue(user=None):
    """Return a pandas table of the SLURM squeue output."""

    if not user:
        user = getpass.getuser()
    result = sp.run(['squeue', '-u', user], stdout=sp.PIPE)
    lines = [l.split() for l in result.stdout.decode().split("\n")]
    try:
        df = pd.DataFrame(lines[1:], columns=lines[0]).dropna()
    except:
        df = pd.DataFrame([["0"] * 8], columns=lines[0], index=[0])
    return df


def module_status_dataframe(status, module="gen"):
    """Convert the status entry for a module to a dataframe."""

    # Target columns
    tcols = ["job_id", "hardware", "fout", "dirout", "job_status", "finput",
             "runtime"]

    # Get the module key
    mkey = MODULE_NAMES[module]

    # Get the module entry
    mstatus = status[mkey]

    # The first entry is the pipeline index
    mindex = mstatus["pipeline_index"]

    # The rest is another dictionary for each sub job
    del mstatus["pipeline_index"]

    # If incomplete:
    if not mstatus:
        for col in tcols:
            if col == "job_status":
                mstatus[col] = "unsubmitted"
            else:
                mstatus[col] = None
        mstatus = {mkey: mstatus} 

    # Create data frame
    mdf = pd.DataFrame(mstatus).T
    mdf["pipeline_index"] = mindex

    return mdf


def convert_time(runtime):
    """Convert squeue time to minutes."""
    mults = [0.016666666666666666, 1, 60]
    time = [int(t) for t in runtime.split(":")][::-1]
    minute_list = [t * mults[i] for i, t in enumerate(time)]
    minutes = sum(minute_list)
    return minutes


def sq_adjust(df, user):
    """Retrieve run time and jobstatus from squeue since these aren't always
    accurate in the status log."""

    # Get the squeue data frame
    sqdf = get_squeue(user)

    # Some entries might not even have a run time column
    if not "runtime" in df.columns:
        df["runtime"] = np.nan

    # Replace runtime and status
    for i, row in sqdf.iterrows():
        jid = row["JOBID"]
        status = row["ST"]
        runtime = convert_time(row["TIME"])
        df["job_status"][df["job_id"] == jid] = status
        df["runtime"][df["job_id"] == jid] = runtime

    return df


def status_dataframe(folder, module=None, user=None):
    """Convert the status entry for a module or an enitre project to a
    dataframe."""

    # Get the status dictionary
    _, status = find_status(folder)

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

    # Now let's borrow some information from squeue
    df = sq_adjust(df, user)

    # And refine this down for the printout
    df["job_name"] = df.index
    df = df[['job_id', 'job_name', 'job_status', 'pipeline_index', 'runtime']]

    return df


def color_print(df):
    """Print each line of a data frame in red for failures and green for
    success."""


    def color_string(string):
        if string == "failed":
            string = Fore.RED + string + Style.RESET_ALL
        elif string == "successful":
            string = Fore.GREEN + string + Style.RESET_ALL
        else:
            string = Fore.YELLOW + string + Style.RESET_ALL
        return string

    df["job_status"] = df["job_status"].apply(color_string)

    print(tabulate(df, showindex=False, headers=df.columns,
                   tablefmt="simple"))


def check_entries(print_df, check):
    if check in FAILURE_STRINGS:
        print_df = print_df[print_df["job_status"] == "failed"]
    elif check in SUCCESS_STRINGS:
        print_df = print_df[print_df["job_status"] == "successful"]
    elif check in PENDING_STRINGS:
        print_df = print_df[print_df["job_status"] == "PD"]
    elif check in RUNNING_STRINGS:
        print_df = print_df[print_df["job_status"] == "R"]    
    elif check in SUBMITTED_STRINGS:
        print_df = print_df[print_df["job_status"] == "submitted"] 
    elif check in UNSUBMITTED_STRINGS:
        print_df = print_df[print_df["job_status"] == "unsubmitted"] 
    else:
        print("Could not find status filter.")

    return print_df


@click.command()
@click.option("--folder", "-f", default=".", help=FOLDER_HELP)
@click.option("--user", "-u", default=None, help=USER_HELP)
@click.option("--module", "-m", default=None, help=MODULE_HELP)
@click.option("--check", "-c", default=None, help=CHECK_HELP)
@click.option("--error", "-e", default=None, help=ERROR_HELP)
@click.option("--out", "-o", default=None, help=OUT_HELP)
def main(folder, user, module, check, error, out):
    """
    revruns - logs

    Check log files of a reV run directory. Assumes certain standard
    naming conventions:

    Configuration File names: \n
    "gen": "config_gen.json" \n
    "econ": "config_econ.json" \n
    "offshore": "config_offshore.json" \n
    "collect": "config_collect.json" \n 
    "multi-year": "config_multi-year.json", \n
    "aggregation": "config_aggregation.son", \n
    "supply-curve": "config_supply-curve.json", \n
    "rep-profiles": "config_rep-profiles.json" \n
    "qaqc": "config_qaqc.json"
    """

    # Expand folder path
    folder = os.path.expanduser(folder)
    folder = os.path.abspath(folder)

    # Find the logging directoy
    try:
        logdir = find_logs(folder)
    except:
        print(Fore.YELLOW + "Could not find log directory" +
              Style.RESET_ALL)
        return

    # Convert module status to data frame
    status_df = status_dataframe(folder, module, user)

    # Now return the requested return type
    if check:
        print_df = check_entries(status_df, check)

    if not error and not out:
        color_print(status_df)

    if error:
        errors = glob(os.path.join(logdir, "stdout", "*e"))
        try:
            elog = [e for e in errors if str(error) in e][0]
        except IndexError:
            print("Error log for job ID " + str(error) + " not found.")
            return
        with open(elog, "r") as file:
            elines = file.readlines()
            if len(elines) > 20:
                print("  \n   ...   \n")
            for e in elines[-20:]:
                print(e)
            print(Fore.YELLOW + "cat " + elog + Style.RESET_ALL) 
    if out:
        outs = glob(os.path.join(logdir, "stdout", "*o"))
        try:
            olog = [o for o in outs if str(out) in o][0]
        except IndexError:
            print("STDOUT log for job ID " + str(out) + " not found.")
            return
        with open(olog, "r") as file:
            olines = file.readlines()
            if len(olines) > 20:
                print("  \n   ...   \n")
            for o in olines[-20:]:
                print(o)
            print(Fore.YELLOW + "cat " + olog + Style.RESET_ALL) 

if __name__ == "__main__":
    main()
