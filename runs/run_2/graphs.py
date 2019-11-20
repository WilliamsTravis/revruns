# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:38:49 2019

@author: twillia2
"""

from glob import glob
import os
from revruns import compare_profiles, extract_arrays, show_colorbars

# set wd temporarily
os.chdir("/Users/twillia2/github/data/revruns/run_2")

# Profiles
savefolder = "graphs"
dpi = 1000

# POA For 2016
files = glob("*h5")
datasets = {f.replace(".h5", ""): extract_arrays(f) for
            f in files if "2015" in f}
compare_profiles(datasets, dataset="poa",
                 units="W $\mathregular{m^{-2}}$",
                 title="Point of Array Irradiance - Los Angeles County (2015)",
                 cmap="viridis",
                 savefolder=savefolder,
                 dpi=dpi)


# Capacity Factor Profile For 2015
files = glob("*h5")
datasets = {f.replace(".h5", ""): extract_arrays(f) for
            f in files if "2015" in f}
for key in datasets.keys():
    datasets[key]['cf_profile'] = datasets[key]['cf_profile'] / 1000
compare_profiles(datasets, dataset="cf_profile", units="Ratio",
                 title="Capacity Factor - Los Angeles County (2015)",
                 cmap="plasma",
                 savefolder=savefolder,
                 dpi=dpi)

# Check the value distributiond of each


