# -*- coding: utf-8 -*-
"""Check with reV status, output, and error logs.

Catch the scenario when a run just starts and it's time is INVALID

See if we can't use pyslurm to speed up the squeue call

rrpipeline is outputting all logs to the working directory, fix that or handle
it here.
"""
import json
import os
import warnings

from glob import glob

import click

from colorama import Fore, Style
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


FOLDER_HELP = ("Path to a folder with a completed set of batched reV runs. "
               "Defaults to current directory. (str)")
MODULE_HELP = ("The reV module logs to check. Defaults to all modules: gen, "
               "collect, multi-year, aggregation, supply-curve, or "
               "rep-profiles.")
STATUS_HELP = ("Print jobs with a given status. Option include 'failed' "
               "(or 'f'), 'success' (or 's'), 'pending' (or 'p'), 'running' "
               "(or 'r'), 'submitted' (or 'sb') and 'unsubmitted (or 'usb').")
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


def check_entries(print_df, check):
    """Check for a specific status."""
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


def checkout(logdir, pid, output="error"):
    """Print out the first 20 lines of an error or stdout log file."""
    if output == "error":
        pattern = "*e"
        name = "Error"
        outs = glob(os.path.join(logdir, "stdout", pattern))
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
        for line in lines[-20:]:
            print(line)
        print(Fore.YELLOW + "cat " + log + Style.RESET_ALL)


def color_print(df, print_folder):
    """Color the status portion of the print out."""
    from tabulate import tabulate

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

    name = "\n" + Fore.CYAN + "/" + print_folder + Style.RESET_ALL + ":"
    df["job_status"] = df["job_status"].apply(color_string)
    pdf = tabulate(df, showindex=False, headers=df.columns, tablefmt="simple")
    print(name)
    print(pdf)


def find_file(folder, file="config_pipeline.json"):
    """Check/return the config_pipeline.json file in the given directory."""
    path = os.path.join(folder, file)
    if not os.path.exists(path):
        msg = ("No {} files found. If you were looking for nested files, try "
               "running the with --walk option.").format(file)
        raise ValueError(Fore.RED + msg + Style.RESET_ALL)
    return path


def find_files(folder, file="config_pipeline.json"):
    """Walk the dirpath directories and find all file paths."""
    paths = []
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            if name == file:
                paths.append(os.path.join(root, file))
        for name in dirs:
            if name == file:
                paths.append(os.path.join(root, file))
    if not paths:
        msg = "No {} files found.".format(file)
        raise ValueError(Fore.RED + msg + Style.RESET_ALL)
    return paths


def find_logs(folder):  # <---------------------------------------------------- Speed this up or use find_files
    """Find the log directory, assumes one per folder."""
    # If there is a log directory directly in this folder use that
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
            except:  # <------------------------------------------------------- What would this/these be?
                pass

            # The directory might be named differently
            try:
                logdir = config["directories"]["log_directory"]
            except KeyError:
                logdir = config["directories"]["logging_directory"]

    # Expand log directory
    if logdir[0] == ".":
        logdir = logdir[2:]
        logdir = os.path.join(folder, logdir)
    logdir = os.path.expanduser(logdir)

    return logdir


def find_outputs(folder):
    """Find the output directory, assumes one per folder."""
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
        print("Could not find 'reV output directory'")
        raise

    # Expand log directory
    if outdir[0] == ".":
        outdir = outdir[2:]  # "it will have a / as well
        outdir = os.path.join(folder, outdir)
    outdir = os.path.expanduser(outdir)

    return outdir


def find_pid_dirs(folders, target_pid):
    """Check the log files and find which folder contain the target pid."""
    pid_dirs = []
    for folder in folders:
        logs = glob(os.path.join(folder, "logs", "stdout", "*e"))
        for line in logs:
            file = os.path.basename(line)
            idx = file.rindex("_")
            pid = file[idx + 1:].replace(".e", "")
            if pid == str(target_pid):
                pid_dirs.append(folder)
    if not pid_dirs:
        msg = "No log files found for pid {}".format(target_pid)
        raise FileNotFoundError(Fore.RED + msg + Style.RESET_ALL)
    return pid_dirs


def find_runtime(job):
    """Find the runtime for a specific job (dictionary entry)."""
    import datetime as dt

    dirout = job["dirout"]
    fout = job["fout"]

    # We will be looking for the logs files associated with fout
    if "_node" in fout:
        fout = fout.replace("node", "")
    jobname = fout.replace(".h5", "")

    # Find all the output logs for this jobname
    logdir = find_logs(dirout)
    stdout = os.path.join(logdir, "stdout")
    logs = glob(os.path.join(stdout, "*{}*.o".format(jobname)))

    # Take the last, will also work if there were multiple attempts
    logpath = logs[-1]

    # We can't get file creation in Linux
    with open(logpath, "r") as file:
        loglines = [line.replace("\n", "") for line in file.readlines()]
    for line in loglines:
        if "INFO" in line or "DEBUG" in line:
            date = line.split()[2]
            time = line.split()[3][:8]
            break

    # Format the start time from these
    time_string = " ".join([date, time])
    stime = dt.datetime.strptime(time_string, "%Y-%m-%d %H:%M:%S")

    # Get the modification time from the file stats
    fstats = os.stat(logpath)
    etime = dt.datetime.fromtimestamp(fstats.st_mtime)

    # Take the difference
    minutes = round((etime - stime).seconds / 60, 3)
    return minutes


def find_runtimes(status):
    """Find runtimes if missing from the main status json."""
    for module, entry in status.items():
        for label, job in entry.items():
            if "pipeline_index" != label:
                if isinstance(job, dict):
                    if "job_id" in job and "runtime" not in job:
                        job["runtime"] = find_runtime(job)
                        status[module][label] = job
    return status


def find_status(sub_folder):
    """Find the job status json."""
    # Find output directory
    try:
        files = glob(os.path.join(sub_folder, "*.json"))
        file = [f for f in files if "_status.json" in f][0]
    except IndexError:
        outdir = find_outputs(sub_folder)
        files = glob(os.path.join(outdir, "*.json"))
        file = [f for f in files if "_status.json" in f]
        if not file:
            return None, None

    # Return the dictionary
    with open(file, "r") as f:
        status = json.load(f)

    # Fix artifacts
    status = fix_status(status)

    # Fill in missing runtimes
    try:
        status = find_runtimes(status)
    except IndexError:
        pass

    return file, status


def fix_status(status):
    """Fix problematic artifacts from older reV versions."""
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


def module_status_dataframe(status, module="gen"):
    """Convert the status entry for a module to a dataframe."""
    import pandas as pd

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


def rrlogs(args):
    """Print status and job pids for a single project directory."""
    folder, sub_folder, module, status, error, out = args

    # Expand folder path
    sub_folder = os.path.abspath(os.path.expanduser(sub_folder))

    # Convert module status to data frame
    status_df = status_dataframe(sub_folder, module)

    # This might return None
    if status_df is not None:
        # Now return the requested return type
        if status:
            status_df = check_entries(status_df, status)

        if not error and not out:
            print_folder = os.path.relpath(sub_folder, folder)
            color_print(status_df, print_folder)

        # If a specific status was requested
        if error or out:
            # Find the logging directoy
            try:
                logdir = find_logs(sub_folder)
            except:
                print(Fore.YELLOW
                      + "Could not find log directory"
                      + Style.RESET_ALL)
                return
            if error:
                checkout(logdir, error, output="error")
            if out:
                checkout(logdir, out, output="stdout")
    else:
        print(Fore.RED + "No status file found for " + sub_folder
              + Style.RESET_ALL)
    return sub_folder


def status_dataframe(sub_folder, module=None):
    """Convert a status entr into dataframe."""
    import pandas as pd

    # Get the status dictionary
    _, status = find_status(sub_folder)

    # There might be a log file with no status data frame
    if status:
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

        # Here, let's improve the time estimation somehow
        if "runtime" not in df.columns:
            df["runtime"] = "nan"

        # And refine this down for the printout
        df["job_name"] = df.index
        df = df[['job_id', 'job_name', 'job_status', 'pipeline_index',
                 'runtime']]

        return df
    else:
        return None


@click.command()
@click.option("--folder", "-f", default=".", help=FOLDER_HELP)
@click.option("--module", "-m", default=None, help=MODULE_HELP)
@click.option("--status", "-s", default=None, help=STATUS_HELP)
@click.option("--error", "-e", default=None, help=ERROR_HELP)
@click.option("--out", "-o", default=None, help=OUT_HELP)
@click.option("--walk", "-w", is_flag=True, help=WALK_HELP)
def main(folder, module, status, error, out, walk):
    """REVRUNS - Check Logs.

    Check log files of a reV run directory. Assumes certain standard
    naming conventions:

    Configuration File names:\n
    "gen": "config_gen.json" \n
    "econ": "config_econ.json" \n
    "offshore": "config_offshore.json" \n
    "collect": "config_collect.json" \n
    "multi-year": "config_multi-year.json" \n
    "aggregation": "config_aggregation.son" \n
    "supply-curve": "config_supply-curve.json" \n
    "rep-profiles": "config_rep-profiles.json" \n
    "qaqc": "config_qaqc.json" \n
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

    # Run rrlogs for each
    args = [(folder, f, module, status, error, out) for f in folders]
    for arg in args:
        _ = rrlogs(arg)


if __name__ == "__main__":
    main()
