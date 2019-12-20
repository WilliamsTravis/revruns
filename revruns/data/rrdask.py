"""
The use of DASK would be useful for many reV output file operations.

The first thing I want to do is just find all of the unique values of a file.
"""
import dask.array as da
import dask.dataframe as ddf
import h5py as hp
import numpy as np
import pandas as pd
from dask.dianostics import ProgressBar

file = "/projects/rev/data/exclusions/CONUS_Exclusions.h5"
dataset = "naris_wind"

f = hp.File(file)
ds = f[dataset]
shape = ds.shape
keys = f.keys()
array = da.from_array(ds, chunks=ds.chunks)
f.close()
