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
SAMPLE_H5 = ("/projects/rev/new_projects/reeds_solar/outputs/"
             "outputs_multi-year.h5")

SAMPLE_STATES = ["Rhode Island"]


class Diurnal:
    """Methods for building diurnal generation profile files."""

    def __init__(self, file):
        """Initialize Diurnal object."""
        self.file = file
        self._set_timestep()

    def __repr__(self):
        """Print representation for a Diurnal object."""
        msg = f"<Diurnal object: file = {self.file}"
        return msg

    def build(self, dst, states=None):
        """Write a sample dataset of a resource file with a subset of US states."""
        # Open the source datset
        ds = h5py.File(self.file, "r")
        meta = self._meta(ds, states)
        idx = np.array(meta.index)

        # Get top level attributes
        trgt = h5py.File(dst, "w")
        for key, attr in ds.attrs.items():
            trgt.attrs[key] = attr

        # Our possibly subetted meta is our new meta
        smeta = meta.reset_index(drop=True)
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
        self._diurnal_sets(ds, trgt, idx)

        # Close datasets
        ds.close()
        trgt.close()

    def _diurnal_sets(self, ds, trgt, idx):
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
            if len(ds[skeys[0]].shape) == 2:
                data = self._diurnal_set(ds, idx, skeys)
                trgt.create_dataset(name=skey, data=data)

    def _diurnal_set(self, ds, idx, skeys):
        """Convert a full timeseries to daily averages by hour."""
        profiles = []
        # arg_list = [(ds, skey, idx) for skey in skeys]
        # with mp.Pool(n) as pool:
        #     for profile in tqdm(pool.imap(self._subset, arg_list),
        #                         total=len(arg_list)):
        #         profiles.append(profile)
        for skey in tqdm(skeys):
            profile = ds[skey][:, idx]
            profiles.append(profile)

        data = self._diurnal(profiles)

        return data

    def _diurnal(self, profiles):
        """Return the average hourly value."""
        # Make sure this is a numpy array
        array = np.vstack(profiles)

        # We want the time step resoltuion averages each location for each day
        hour_layers = []
        for hour in range(0, self.nsteps):
            avg = np.mean(array[hour::self.nsteps], axis=0)
            hour_layers.append(avg)
        data = np.vstack(hour_layers)

        return data

    def _meta(self, ds, states=None):
        """Format meta object."""
        meta = pd.DataFrame(ds["meta"][:])
        meta.rr.decode()
        if states:
            meta = meta[meta["state"].isin(states)]
        return meta

    def _set_timestep(self):
        """Set the right timestep based on the time index."""
        with h5py.File(self.file, "r") as ds:
            time = [t for t in ds.keys() if "time" in t][0]
            self.nsteps = int(ds[time].shape[0] / 8760) * 24


if __name__ == "__main__":
    file = SAMPLE_H5
    states = SAMPLE_STATES
    dst = "/scratch/twillia2/reeds_solar_ri.h5"
    self = Diurnal(file)
    self.build(dst, states)
