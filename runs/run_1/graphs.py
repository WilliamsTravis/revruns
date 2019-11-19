# -*- coding: utf-8 -*-
"""
This script will create some standard output graphs and figures from reV output
hdf5 files.

Careful with this on normal machines (memory)

Move to functions at some point.

If we standardized the file name formats we could further automate this:
    i.e. "module_groupingfeature_year.h5"

Created on Thu Nov 14 10:28:59 2019

@author: twillia2
"""
from glob import glob
import os
from revruns import compare_profiles, extract_arrays, show_colorbars

# set wd temporarily
os.chdir("/Users/twillia2/github/data/revruns/run_1")

# Profiles
savefolder = "graphs"
dpi = 1000

# POA For 2016
files = glob("*h5")
datasets = {f.replace(".h5", ""): extract_arrays(f) for
            f in files if "2016" in f}
compare_profiles(datasets, dataset="poa",
                 units="W $\mathregular{m^{-2}}$",
                 title="Point of Array Irradiance - Denver Area (2016)",
                 cmap="viridis",
                 savefolder=savefolder,
                 dpi=dpi)

# POA for 2017
datasets = {f.replace(".h5", ""): extract_arrays(f) for
            f in files if "2017" in f}
compare_profiles(datasets, dataset="poa",
                 units="W $\mathregular{m^{-2}}$",
                 title="Point of Array Irradiance - Denver Area (2017)",
                 cmap="viridis",
                 savefolder=savefolder,
                 dpi=dpi)

# Capacity Factor Profile For 2016
files = glob("*h5")
datasets = {f.replace(".h5", ""): extract_arrays(f) for
            f in files if "2016" in f}
for key in datasets.keys():
    datasets[key]['cf_profile'] = datasets[key]['cf_profile'] / 1000
compare_profiles(datasets, dataset="cf_profile", units="Ratio",
                 title="Capacity Factor - Denver Area (2016)",
                 cmap="plasma",
                 savefolder=savefolder,
                 dpi=dpi)

# Capacity Factor for 2017
datasets = {f.replace(".h5", ""): extract_arrays(f) for
            f in files if "2017" in f}
for key in datasets.keys():
    datasets[key]['cf_profile'] = datasets[key]['cf_profile'] / 1000
compare_profiles(datasets, dataset="cf_profile", units="Ratio",
                 title="Capacity Factor - Denver Area (2017)",
                 cmap="plasma",
                 savefolder=savefolder,
                 dpi=dpi)


# Check the value distributiond of each
# This will be panel that 
