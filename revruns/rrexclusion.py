#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Append a data set to an exclusions layer.

Created on Wed Feb 19 12:27:23 2020

@author: twillia2
"""
import os
import click
import h5py
import tempfile
from reVX.utilities.exclusions_converter import ExclusionsConverter  # <------- /lustre/eaglefs/shared-projects/rev/modulefiles/gitrepos/reVX/reVX/utilities


EXCL_HELP = ("Path to an HDF5 exclusions file to add to. (str)")
DESC_HELP = ("A description of the data set being added. (str)")
ADD_HELP = ("Path to a 2D GeoTiff file with a data set to be added. (str)")
NAME_HELP = ("The name of the added data set in the HDF5 file. (str)")


@click.command()
@click.option("--excl_file", "-e", required=True, help=EXCL_HELP)
@click.option("--add_file", "-a", required=True, help=ADD_HELP)
@click.option("--name", "-n", required=True, help=NAME_HELP)
@click.option("--desc", "-d", required=True, help=DESC_HELP)
@click.option("--verbose", "-v")
def main(excl_file, add_file, name, desc, verbose):
    """Append a data set to an exlusion file.
    
    Issues:
        
        This is a rather long process, could it be run in background or
        quickened?
    
    """
    
    # Create a dataset name mapping with the path to the geotiff
    data_map = {name: add_file}
    desc_map = {name: desc}

    # Create a temporary hdf5 file path
    if verbose:
        print("Creating Temporary HDF5 file from " + add_file)
    temp_h5 = tempfile.mkstemp(suffix=".h5")[1]
    os.remove(temp_h5)

    # Create a temporary h5 file
    ExclusionsConverter.layers_to_h5(temp_h5, data_map, descriptions=desc_map)

    # Append to excl_file
    if verbose:
        print("Appending dataset to " + excl_file)
    with h5py.File(temp_h5, "r") as new_file:
        with h5py.File(excl_file, "r+") as old_file:
            old_file.create_dataset(name=name, data=new_file[name])
            for k in new_file[name].attrs.keys():
                old_file[name].attrs[k] = new_file[name].attrs[k]

    # Delete temp file
    os.remove(temp_h5)

if __name__ == "__main__":
    main()
