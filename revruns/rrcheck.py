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
import json
import os
from osgeo import gdal
import pandas as pd
from revruns import VARIABLE_CHECKS
from tqdm import tqdm

# Help printouts
dir_help = "The directory from which to read the hdf5 files (Defaults to '.')."
write_help = "Write output to file (saved as 'checkvars.csv')."

# The command
@click.command()
@click.option("--directory", "-d", default=".", help=dir_help)
@click.option("--write", "-w", is_flag=True, help=write_help)
def main(directory, write):
    """Checks all hdf5 files in a current directory for threshold values in data
    sets. This uses GDALINFO and also otputs an XML file with summary statistics
    and attribute information for each hdf5 file.
    """
    # Get and open files. 
    files = glob(os.path.join(directory, "*h5"))
    flagged = {}
    for file in tqdm(files, position=0):
        # Open the file
        pointer = gdal.Open(file)

        # Get the list of sub data sets in each file
        subds = pointer.GetSubDatasets()

        # For each of these sub data sets, get an info dictionary
        for sub in subds:      
            info_str = gdal.Info(sub[0], options=["-stats", "-json"])      
            info = json.loads(info_str)
            filename = info["files"][0]
            desc = info["description"]
            var = desc[desc.index("//") + 2: ]
            max_data = info["bands"][0]["maximum"]
            min_data = info["bands"][0]["minimum"]
            max_threshold = VARIABLE_CHECKS[var][1]
            min_threshold = VARIABLE_CHECKS[var][0]

            # Check the thresholds. Could add more for mean and stdDev. 
            if max_data > max_threshold:
                filename = os.path.basename(filename)
                flag = ":".join([filename, var])
                message = (" - maximum value is greater than " +
                           max_threshold)
                flagged[flag] = message
            if min_data < min_threshold:
                filename = os.path.basename(filename)
                flag = ":".join([filename, var])
                message = (" - minimum value is less than " +
                           min_threshold)
                flagged[flag] = message


    # If there are issues either print or write out the messages.
    if len(flagged) > 0:
        if not write:
            for flag, message in flagged.items():
                print(flag, message)
        else:
            datasets = [os.path.abspath(flag) for flag in flagged.keys()]
            messages = [message for _, message in flagged.items()]
            out = pd.DataFrame({"datasets": datasets, "messages": messages})
            out.to_csv("checkvars.csv", index=False)
    else:
        print("All files passed.")

if __name__ == "__main__":
    main()
