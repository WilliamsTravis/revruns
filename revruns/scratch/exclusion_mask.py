"""Combine exlcusions layers in config into a single file."""
import json
import os

import h5py
import rasterio as rio

from revruns import rr
from reV.supply_curve.exclusions import ExclusionMaskFromDict


DP = rr.Data_Path("/shared-projects/rev/projects/weto/fy20/task_1/aggregation")
SP = rr.Data_Path("/projects/rev/data/conus/exclusion_masks/fy21", mkdir=True)
AG_DIRS = {
    # "limited_access": DP.join("01_na_n_mid"),
    "reference_access": DP.join("05_b_b_mid")
}


def mask(ag_dir, dst):
    """Use a supply-curve config to aggregate exclusion layers into one."""
    if os.path.exists(dst):
        return

    # Get the aggreation configuration file
    path = os.path.join(ag_dir, "config_aggregation.json")
    with open(path, "r") as file:
        config = json.load(file)

    # Extract the needed elements from the confgi
    excl_h5 = config["excl_fpath"]
    layers_dict = config["excl_dict"]
    if "min_area" in config:
        min_area = config["min_area"]
    else:
        min_area = None
    if "area_filter_kernel" in config:
        kernel = config["area_filter_kernel"]
    else:
        kernel = "queen"

    # Create a mask converter
    masker = ExclusionMaskFromDict(excl_h5, layers_dict=layers_dict,
                                   min_area=min_area, kernel=kernel)

    # Get the mask and the georeferencing
    mask = masker.mask
    mask = mask.astype("float32")
    try:
        profile = masker.excl_h5.profile
    except KeyError:
        with h5py.File(excl_h5, "r") as ds:
            for key in ds.keys():
                if "profile" in ds[key].attrs.keys():
                    profile = ds[key].attrs["profile"]
                    profile = json.loads(profile)
    profile["dtype"] = str(mask.dtype)

    # Save
    with rio.Env():
        with rio.open(dst, "w", **profile) as file:
            file.write(mask, 1)


def masks():
    """Write an aggregated exclusion layer for each scenario to a geotiff."""
    for scenario, ag_dir in AG_DIRS.items():
        file = scenario + ".tif"
        dst = SP.join(file)
        if not os.path.exists(dst):
            print("Processing " + scenario + "...")
            mask(ag_dir, dst)
        else:
            print(scenario + "already processed, skipping.")



if __name__ == "__main__":
    masks()
