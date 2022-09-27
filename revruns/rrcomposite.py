# -*- coding: utf-8 -*-
"""Build a singular composite inclusion layer and write to geotiff.

Created on Wed Feb  9 09:10:26 2022

@author: twillia2
"""
import click
import json
import os

import h5py
import rasterio as rio

from reV.supply_curve.exclusions import ExclusionMaskFromDict
from rex import init_logger


HELP = {
    "config": ("Path to reV aggregation JSON configuration file containing an "
               "exclusion dictionary ('excl_dict')."),
    "dst": ("Destination path to output GeoTiff. Defaults to current directory"
            "using the name of aggregation config file."),
}


def composite(config, dst=None):
    """Combine exclusion layers into one composite inclusion raster.
    
    Parameters
    ----------
    config: str
        Path to reV aggregation JSON configuration file containing an exclusion
        dictionary ('excl_dict').
    dst : str
        Destination path to output GeoTiff. Defaults to current directory
        using the name of containing directory of the aggregation config file.
    """
    # Create job name and destination path
    config = os.path.abspath(config)
    if not dst:
        dst = "./" + os.path.dirname(config).split("/")[-1] + ".tif"
    name = os.path.dirname(config).split("/")[-1]

    init_logger("rrcomposite", log_level="DEBUG", log_file=name + ".log")

    # Open a config_aggregation.json file with the exlcusion logic.
    with open(config, "r") as file:
        conf = json.load(file)
    excl_dict = conf["excl_dict"]
    excl_h5 = conf["excl_fpath"]

    # Run reV to merge
    masker = ExclusionMaskFromDict(excl_h5, layers_dict=excl_dict)
    mask = masker.mask
    mask = mask.astype("uint8")

    # Get a raster profile from the h5 dataset
    if isinstance(excl_h5, list):
        template = excl_h5[0]
    else:
        template = excl_h5
    with h5py.File(template, "r") as ds:
        profile = json.loads(ds.attrs["profile"])
    profile["dtype"] = str(mask.dtype)
    profile["nodata"] = None
    profile["blockxsize"] = 256
    profile["blockysize"] = 256
    profile["tiled"] = "yes"
    profile["compress"] = "lzw"

    # Save to a single raster
    # os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"Saving output to {dst}...")
    with rio.open(dst, "w", **profile) as file:
        file.write(mask, 1)


@click.command()
@click.option("--config", "-c", required=1, help=HELP["config"])
@click.option("--dst", "-d", required=0, default=None, help=HELP["dst"])
def main(config, dst):
    """Combine exclusion layers into one composite inclusion raster."""
    composite(config, dst)


if __name__ == "__main__":
    config = "/shared-projects/rev/projects/puerto_rico/fy22/pr100/data/exclusions/config_excl_open.json"
    dst = "/shared-projects/rev/projects/puerto_rico/fy22/pr100/data/exclusions/exclusions_open.tif"
