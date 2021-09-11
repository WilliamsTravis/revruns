#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 12:00:14 2021

@author: twillia2
"""
import json
import os

from glob import glob

os.chdir("/shared-projects/rev/projects/weto/fy21/atb/rev/aggregation/onshore")


def fixit(file):
    """Set the status to success for aggregation."""
    with open(file, "r") as f:
        config = json.load(f)
    if "job_status" in config["supply-curve-aggregation"]:
        del config["supply-curve-aggregation"]["job_status"]
    if "rep-profiles" in config:
        del config["rep-profiles"]
    fname = os.path.basename(file)
    cname = "_".join(fname.split("_")[:-1]) + "_agg"
    config["supply-curve-aggregation"][cname]["job_status"] = "successful"

    with open(file, "w") as f:
        f.write(json.dumps(config, indent=4))


def main():
    """Reset status files."""
    files = glob("*/*status*")
    for file in files:
        fixit(file)


if __name__ == "__main__":
    main()
