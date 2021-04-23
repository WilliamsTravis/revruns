# -*- coding: utf-8 -*-
"""Setup reV configurations from master table.

Things to do:
    - Update master config sheet
        - Their are several incomplete entries
        - Some datasets are binary, some are continuous, and some have a
          lookup dictionary because they are derived from character values.
          So, perhaps it would be easiest to enter the proper values for
          each exclusion. i.e. include [0, 1, 2], or exclude [1, 2, 3, 4, 5]
        - Relatedly, some will be ranges (distance to shore, 3nm state water
                                          exlcusion)



Created on Fri Mar 26 13:56:34 2021

@author: twillia2
"""
import os

import h5py
import pandas as pd

from revruns import rr


EXCL = "/projects/rev/data/exclusions/Offshore_Exclusions.h5"
RESOURCE = "/datasets/WIND/conus/v1.0.0/wtk_conus_{}.h5"
DP = rr.Data_Path("/shared-projects/rev/projects/weto/fy21/offshore")
MASTER = DP.join("data/tables/FY21 WETO Offshore.xlsx")
TECHS = ["mid", "adv"]
ACCESSES = ["limited", "open"]


class Setup:
    """Class methods for setting up WETO FY21 Offshore reV runs."""

    @property
    def scenarios(self):
        """Return folder dictionary for all scenarios."""
        scenarios = {}
        ag_number = 0
        gen_number = 0
        for tech in TECHS:
            gen_key = f"{gen_number}_{tech}"
            gen_folder = DP.join("rev/generation", gen_key, mkdir=True)
            gen_number += 1
            for access in ACCESSES:
                key = f"{ag_number}_{access}_{tech}"
                ag_folder = DP.join("rev/aggregation", key, mkdir=True)
                scenarios[key] = {}
                scenarios[key]["gen"] = gen_folder
                scenarios[key]["agg"] = ag_folder
                ag_number += 1
        return scenarios

    @property
    def project_points(self):
        """Return only offshore points for now."""
        dst = DP.join("project_points/project_points_offshore.csv", mkdir=True)
        if not os.path.exists(dst):
            with h5py.File(RESOURCE.format(2007), "r") as ds:
                meta = pd.DataFrame(ds["meta"][:])
            meta.rr.decode()
            meta["gid"] = meta.index
            meta = meta[meta["offshore"] == 1]
            meta["config"] = "default"
            meta.to_csv(dst, index=False)
        return dst

    @property
    def master_tech(self):
        """Return formatted master technology sheet."""
        return rr.get_sheet(MASTER, "technology_costs")

    @property
    def master_trans(self):
        """Return formatted master transmission sheet."""
        return rr.get_sheet(MASTER, "transmission")

    @property
    def master_access(self):
        """Return formatted master access sheet."""
        df = rr.get_sheet(MASTER, "exclusions")
        df = df[["dataset_name", "rev_name", "category", "limited_access",
                 "reference_access", "open_access"]]
        df = df[~pd.isnull(df["rev_name"])]
        df = df.fillna(0)
        return df


def main():
    """Configure entire project pipeline."""
    # Setup scenario folders

    # Project points

    # Sam

    # Generation

    # Collect

    # Multiyear

    # Aggregation

    # Supply Curve
