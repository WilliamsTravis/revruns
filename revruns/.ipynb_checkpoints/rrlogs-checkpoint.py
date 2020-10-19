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
from revruns.rrpipeline import find_files


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
STATUS_HELP = ("Print jobs with a given status. Option include 'failed' "
               "(or 'f'), 'success' (or 's'), 'pending' (or 'p'), 'running' "
               "(or 'r'), 'submitted' (or 'sb') and 'unsubmitted (or 'usb')")
ERROR_HELP = ("A job ID. This will print the first 20 lines of the error log "
              "of a job.")
OUT_HELP = ("A job ID. This will print the first 20 lines of the standard "
            "output log of a job.")
WALK_HELP = ("Walk the given directory structure and return the status of "
             "all jobs found.")
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
    """Find the log folders based on configs present in folder. Assumes
    only one log directory per folder.

    folder = "/shared-projects/rev/projects/soco/rev/runs/reference/generation/120hh/20ps/logs/stdout"
    """
    # if there is a log directory directly in this folder use that
    contents = glob(os.path.join(folder, "*"))
    possibles = [c for c in contents if "log" in c]
    if len(possibles) == 1:
        logdir = os.path.join(folder, possibles[0])
        return logdir

    # If that didn't work, check the current path
    if "/logs/" in folder:
        logdir = os.path.join(folder[:folder.index("logs")], "logs")
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
        file = [f for f in files if "_status.json" in f]
        if not file:
            raise FileNotFoundError(Fore.RED
                                    + "No status file found."
                                    + Style.RESET_ALL)

    # Return the dictionary
    with open(file, "r") as f:
        status = json.load(f)

    # Fix artifacts
    status = fix_status(status)

    return file, status


def find_pid_dirs(folders, target_pid):
    """Check the log files and find which folder contain the target pid."""
    pid_dirs=[]
    for folder in folders:
        logs = glob(os.path.join(folder, "logs", "stdout", "*e"))
        for l in logs:
            file = os.path.basename(l)
            idx = file.rindex("_")
            pid = file[idx + 1:].replace(".e", "")
            if pid == str(target_pid):
                pid_dirs.append(folder)
    if not pid_dirs:
        raise FileNotFoundError(Fore.RED
                                + "No log files found for pid "
                                + str(target_pid) + Style.RESET_ALL)
    return pid_dirs


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


def get_scontrol(jobid):
    """Return the full status printout of a specific job, override status
    json if this is available.

    Notes:
        I haven't seen enough outputs to know how long this is availabe.
    """

    result = sp.run(["scontrol",  "show", "jobid", "-dd", str(jobid)],
                    stdout=sp.PIPE)
    lines = [l.split() for l in result.stdout.decode().split("\n")]

    # Line 4 contains the jobstatus
    statuses = {
        "FAILED": "failed",
        "COMPLETED": "successful",
        "PENDING": "PD",
        "RUNNING": "R"
    }
    status = lines[3][0].split("=")[1]
    if status in statuses:
        status = statuses[status]

    # Runtime is in line 7
    runtime = lines[6][0].split("=")[1]

    return status, runtime


def checkout(logdir, pid, output="stdout"):
    """Print out the first 20 lines of an error or stdout log file."""
    
    if output == "stdout":
        pattern = "*e"
        name = "Error"
    else:
        pattern = "*o"
        name = "STDOUT"
        outs = glob(os.path.join(logdir, "stdout", pattern))

    try:
        log = [o for o in outs if str(pid) in o][0]
    except IndexError:
        print(Fore.RED + name + " log for job ID " + str(pid)
              + " not found." + Style.RESET_ALL)
        return

    with open(log, "r") as file:
        lines = file.readlines()
        if len(lines) > 20:
            print("  \n   ...   \n")
        for l in lines[-20:]:
            print(l)
        print(Fore.YELLOW + "cat " + log + Style.RESET_ALL) 


def convert_time(runtime):
    """Convert squeue time to minutes."""
    mults = [0.016666666666666666, 1, 60]
    time = [int(t) for t in runtime.split(":")][::-1]
    minute_list = [t * mults[i] for i, t in enumerate(time)]
    minutes = round(sum(minute_list), 4)
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
        elif string == "R":
            string = Fore.BLUE + string + Style.RESET_ALL
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


def logs(folder, user, module, status, error, out):
    """Print status and job pids for a single project directory."""
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
    if status:
        print_df = check_entries(status_df, check)

    if not error and not out:
        color_print(status_df)

    if error:
        checkout(logdir, error, output="stderr")
    if out:
        checkout(logdir, pid, output="stdout")


@click.command()
@click.option("--folder", "-f", default=".", help=FOLDER_HELP)
@click.option("--user", "-u", default=None, help=USER_HELP)
@click.option("--module", "-m", default=None, help=MODULE_HELP)
@click.option("--status", "-s", default=None, help=STATUS_HELP)
@click.option("--error", "-e", default=None, help=ERROR_HELP)
@click.option("--out", "-o", default=None, help=OUT_HELP)
@click.option("--walk", "-w", is_flag=True, help=WALK_HELP)
def main(folder, user, module, status, error, out, walk):
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

    folder = "/shared-projects/rev/projects/soco/rev/runs/aggregation"
    error  = 4183851
    out = None
    user = None
    module = None
    status = None
    walk = True
    """
    # If walk find all project directories with a 
    if walk or error or out:
        folders = [os.path.dirname(f) for f in find_files(folder, file="logs")]
    else:
        folders = [folder]

    # If an error our stdout logs is requested, only run the containing folder
    if error:
        folders = find_pid_dirs(folders, error)
    if out:
        folders = find_pid_dirs(folders, out)

    # Run logs for each
    for folder in folders:
        print("\n" + Fore.CYAN + folder + ": " + Style.RESET_ALL)
        logs(folder, user, module, status, error, out)


if __name__ == "__main__":
    main()
