# -*- coding: utf-8 -*-
"""
This script will create some standard output graphs and figures from reV output
hdf5 files.

Move to functions at some point.

Created on Thu Nov 14 10:28:59 2019

@author: twillia2
"""
import datetime as dt
import h5py as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# set wd temporarily
os.chdir("/Users/twillia2/github/data/revruns/run_1")

# From run_1
files = ["pvwattsv5_fixed_2016.h5", "pvwattsv5_tracking_2016.h5",
         "pvwattsv5_fixed_2017.h5", "pvwattsv5_tracking_2017.h5"]

# 2016 fixed pv
sample = files[0]

# Get year
year = int(sample[sample.index(".") - 4: sample.index(".")])

# Open file
file = hp.File(files[0], mode="r")

# Get keys?
keys = file.keys()

# Get the three data sets
cf_profile = file["cf_profile"][:]
cf_mean = file["cf_mean"][:]
poa = file["poa"][:]
time = file["time_index"]

# Get the date-times
t1 = dt.datetime(year, 1, 1, 0, 0).strftime("%Y-%m-%d %H:%M")
t2 = dt.datetime(year, 12, 31, 23, 59).strftime("%Y-%m-%d %H:%M")
dates = pd.date_range(t1, t2, freq="30 min")



# Capacity Factor Means
plt.hist(cf_mean, bins=50)


# Capacity Factor Profile
