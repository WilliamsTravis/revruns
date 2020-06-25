#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a point file from an exclusion h5. 

Created on Thu Jun 25 11:11:45 2020

@author: twillia2
"""


import h5py
import pandas as pd



# Get just the array and the lat lons
def to_raster():
    """Take a 2D exclusion point file and turn it into a raster."""

    with h5py.File(file, "r") as ds:
        keys = list(ds.keys())
        array = ds[dataset][:]
        lats = ds["latitude"][:]
        lons = ds["longitude"][:]