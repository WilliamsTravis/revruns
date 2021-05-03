# -*- coding: utf-8 -*-
"""Return a diurnal average version of a reV generation file.

Created on Sun Apr 25 18:38:26 2021

@author: twillia2
"""
import h5py
import numpy as np
import pandas as pd
import pathos.multiprocessing as mp

from revruns import rr
from tqdm import tqdm


DP = rr.Data_Path("/shared-projects/rev/projects/weto/fy21/transition/rev")
SAMPLE_H5 = DP.join("scenario_01/scenario_01_multi-year.h5")
SAMPLE_STATES = ["Rhode Island"]


class Diurnal:
    """Methods for building diurnal generation profile files."""

    def __init__(self, file):
        """Initialize Diurnal object."""
        self.file = file

    def __repr__(self):
        """Print representation for a Diurnal object."""
        msg = f"<Diurnal object: file = {self.file}"
        return msg

    def build(self, dst, states=None):
        """Write a sample dataset of a resource file with a subset of US states."""
        # Open the source datset
        # with h5py.File(self.file, "r") as ds:
        ds = h5py.File(self.file, "r")
        smeta = self._state_meta(ds, states)
        idx = np.array(smeta.index)

        # Get top level attributes
        # with h5py.File(dst, "w") as trgt:
        trgt = h5py.File(dst, "w")
        for key, attr in ds.attrs.items():
            trgt.attrs[key] = attr

        # Our possibly subetted meta is our new meta
        smeta = smeta.reset_index(drop=True)
        smeta, dtypes = smeta.rr.to_sarray()
        trgt.create_dataset(name="meta", data=smeta, dtype=dtypes)

        # Loop through each dataset, pass through the 2D datasets
        for key, data in ds.items():
            # Get none time-series elements
            if any([k in key for k in ["means", "stdev"]]):
                trgt.create_dataset(name=key, data=ds[key][idx])
                for akey, attrs in ds[key].attrs.items():
                    trgt[key].attrs[akey] = attrs

        # Now build the mult-year dirunal profiles
        # diurnal_sets = self._diurnal_sets(ds, idx)


    def _diurnal_sets(self, ds, idx):
        """Convert full timeseries to daily averages by hour."""
        # Filter out the non-2d dataset keys
        keys = list(ds.keys())
        keys.sort()
        drops = ["time_index", "meta", "mean", "stdev"]
        for drop in drops:
            keys = [k for k in keys if drop not in k]

        # Group the 3D dataset keys
        profile_groups = {}
        for key in keys:
            gkey = key.split("-")[0]
            if gkey not in profile_groups:
                profile_groups[gkey] = []
            profile_groups[gkey].append(key)

        # Build the diurnal array for each 3D group
        for skey, skeys in profile_groups.items():
            data = self._diurnal_set(ds, idx, skeys)

    def _diurnal_set(self, ds, idx, skeys):
        """Convert a full timeseries to daily averages by hour."""
        n = len(skeys)
        arg_list = [(ds, skey, idx) for skey in skeys]
        profiles = []
        # with mp.Pool(n) as pool:
        #     for profile in tqdm(pool.imap(self._subset, arg_list), total=n):
        #         profiles.append(profile)
        for skey in tqdm(skeys):
            profile = ds[skey][:, idx]
            profiles.append(profile)

        data = self._diurnal(profiles)

        return data

    def _diurnal(self, profiles):
        """Return the average hourly value."""
        # Make sure this is a numpy array
        profiles = np.array(profiles)

        # Let's start with a single time series
        profiles = np.vstack(profiles)

        # We want 24-hour averages at each location  # <----------------------- Indexing will change depending on timestep here
        hour_layers = []
        for hour in range(0, 24):
            avg = np.mean(profiles[hour::23], axis=0)
            hour_layers.append(avg)
        hour_profiles = np.vstack(hour_layers)

        return hour_profiles

    def _state_meta(self, ds, states=None):
        """Return the index position associated with a single or list of states."""
        meta = pd.DataFrame(ds["meta"][:])
        meta.rr.decode()
        if states:
            meta = meta[meta["state"].isin(states)]
        return meta


if __name__ == "__main__":
    file = SAMPLE_H5
    states = SAMPLE_STATES
    dst = "/scratch/twillia2/scenario_01_utco.h5"
    self = Diurnal(file)
