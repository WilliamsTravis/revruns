# -*- coding: utf-8 -*-
"""Run reV pipeline configs.

Created on Sat Sep  5 17:03:21 2020

@author: twillia2
"""

import click
import json
import os
import shlex
import subprocess as sp

from colorama import Fore, Style
from glob import glob
from pathlib import Path
from reV.pipeline import Pipeline
from rex.utilities.execution import SubprocessManager


DIR_HELP = ("The directory containing one or more config_pipeline.json "
            "files. Defaults to current directory. (str)")
WALK_HELP = ("Walk the directory structure and run all config_pipeline.json "
             "files. (boolean)")
LOG_HELP = "Check pipeline status files instead. (boolean)"
PRINT_HELP = "Print the path to all pipeline configs found instead. (boolean)"


def find_file(dirpath, file="config_pipeline.json"):
    """Check/return the config_pipeline.json file in the given directory."""
    config_path = os.path.join(dirpath, file)
    if not os.path.exists(config_path):
        raise ValueError(ValueError(Fore.RED
                         + "No "
                         + file
                         + " files found."
                         + Style.RESET_ALL))
    return config_path


def find_files(dirpath, file="config_pipeline.json"):
    """Walk the dirpath directories and finall confi_pipeline.json paths.
    """
    config_paths = []
    for root, dirs, files in os.walk(dirpath, topdown=False):
        for name in files:
            if name == file:
                config_paths.append(os.path.join(root, file))
        for name in dirs:
            if name == file:
                config_paths.append(os.path.join(root, file))
    if not config_paths:
        raise ValueError(Fore.RED
                         + "No "
                         + file
                         + " files found."
                         + Style.RESET_ALL)
    return config_paths


@click.command()
@click.option("--dirpath", "-d", default=".", help=DIR_HELP)
@click.option("--walk", "-w", is_flag=True, help=WALK_HELP)
@click.option("--print_paths", "-p", is_flag=True, help=PRINT_HELP)
def rrpipeline(dirpath, walk, print_paths):
    """Run one or all reV pipelines in a directory."""
    dirpath = os.path.expanduser(dirpath)
    dirpath = os.path.abspath(dirpath)
    if walk:
        config_paths = find_files(dirpath)
    else:
        config_paths = [find_file(dirpath)]

    for path in config_paths:
        if print_paths:
            print(Fore.CYAN + path + Style.RESET_ALL)
        else:
            # Check if its already done first here
            # ...

            print(Fore.CYAN + "Submitting " + path + "..." + Style.RESET_ALL)
            name = "_".join(os.path.dirname(path).split("/")[-3:])
            cmd = "nohup reV -c " + path + " -n " + name + " pipeline --monitor"
            cmd = shlex.split(cmd)
            output = os.path.join(os.path.dirname(path), "pipeline.out")
            process = sp.Popen(cmd,
                               stdout=open(output, "a"),
                               stderr=open(output, "a"),
                               preexec_fn=os.setpgrp)
            if process.returncode == 1:
                raise OSError("Submission failed: check {}".format(output))

            # with open(output, "r") as file:
            #     lines = file.readlines()
            #     for line in lines:
            #         print(line)

            SubprocessManager.submit(cmd, background=True,
                                     background_stdout=False)

    return config_paths


if __name__ == "__main__":
    rrpipeline()
