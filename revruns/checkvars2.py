#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attempting to find min and max data set values in reasonable amount of time
for large hdf5 files.

Created on Tue Nov 26 07:46:53 2019

@author: twillia2
"""

import dask.array as da
import h5py
from revruns import Check_Variables

# Testing with a tiny sample ~ 10 MB
file = "set1_sf0_gen_2015_node24.h5"

# Benchmark this against gdalinfo strategy
check = Check_Variables([file])
%timeit check.checkvars()  # 2.34 ms

# One data set - Chunk size for a 2018392 X 17520 (35,362,227,840 point) ds?
def dask_check(file):
    ds = h5py.File(file)["cf_profile"]
    array = da.from_array(ds, chunks=(1000, 1000))
    array.max().compute()
