# -*- coding: utf-8 -*-
"""
A CLI to quickly check if HDF files have potentially anomalous values. This
will check all files with an 'h5' extension in the current directory.
The checks, so far, only include whether the minimum and maximum values are
within the ranges defined in the "VARIABLE_CHECKS" object.

Created on Mon Nov 25 08:52:00 2019

@author: twillia2
"""
import click
from glob import glob
import os
import pandas as pd
from revruns import Check_Variables

dir_help = "The directory from which to read the hdf5 files. Defaults to '.'"
write_help = "Write output to file (checkvars.csv)."

@click.command()
@click.option("-directory", default=".", help=dir_help)
@click.option("--write", default=False, help=write_help)
def main(directory, write):
    """Checks hdf5 files in current directory for threshold values in data
    sets."""

    # Get and open files. 
    files = glob("*h5")
    checks = Check_Variables(files)
    flags = checks.check_variables()

    # If there are issues either print or write out the messages.
    if len(flags) > 0:
        if not write:
            for flag, message in flags.items():
                print(flag, message)
        else:
            datasets = [os.path.abspath(flag) for flag in flags.keys()]
            messages = [message for _, message in flags.items()]
            out = pd.DataFrame({"datasets": datasets, "messages": messages})
            out.to_csv("checkvars.csv", index=False)

if __name__ == "__main__":
    main()
