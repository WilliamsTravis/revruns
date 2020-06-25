# -*- coding: utf-8 -*-
"""Append a data set to an exclusions layer.
"""

import os

import click
import h5py

from reVX.utilities.exclusions_converter import ExclusionsConverter


EXCL_HELP = ("Path to an HDF5 exclusions file to add to. (str)")
DESC_HELP = ("A description of the data set being added. (str)")
ADD_HELP = ("Path to a 2D GeoTiff file with a data set to be added. (str)")
NAME_HELP = ("The name of the added data set in the HDF5 file. Defaults to "
             "the file name of the GeoTiff with no extension. (str)")


@click.command()
@click.option("--excl_file", "-e", required=True, help=EXCL_HELP)
@click.option("--add_file", "-a", required=True, help=ADD_HELP)
@click.option("--name", "-n", default=None, help=NAME_HELP)
@click.option("--desc", "-d", required=True, help=DESC_HELP)
@click.option("--verbose", "-v", is_flag=True)
def main(excl_file, add_file, name, desc, verbose):
    """Append a data set to an exlusion file.
    
    Issues:
        This is a rather long process.
    excl_file = "/shared-projects/rev/exclusions/ATB_Exclusions.h5"
    add_file = "/projects/rev/data/conus/friction_surface_102008/friction_surface_conus.tif"
    desc = "Social resistance/friction index"
    name = "friction"
    verbose = True
    """

    # set the name with the tiff file if not specified
    if not name:
        name = os.path.basename(add_file).split(".")[0]

    # Create a dataset name mapping with the path to the geotiff
    layer_map = {name: add_file}
    desc_map = {name: desc}

    # Create a temporary hdf5 file path
    if verbose:
        print("Creating Temporary HDF5 file from " + add_file)
    temp_h5 = os.path.join(os.path.dirname(excl_file), name + ".h5")

    # Create a temporary h5 file
    ExclusionsConverter.layers_to_h5(temp_h5, layer_map, descriptions=desc_map)

    # Append to excl_file
    if verbose:
        print("Appending dataset to " + excl_file)

    with h5py.File(temp_h5, "r") as new:
        with h5py.File(excl_file, "r+") as old:
            data = new[name]
            if len(data.shape) == 3:
                old.create_dataset(name=name, data=data, chunks=(1, 128, 128))
            else:
                old.create_dataset(name=name, data=data, chunks=(128, 128))

            for k, value in new[name].attrs.items():
                old[name].attrs[k] = value

    # Delete temp file
    os.remove(temp_h5)


if __name__ == "__main__":
    main()