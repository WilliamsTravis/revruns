# -*- coding: utf-8 -*-
"""revruns datasets.

Methods for accessing and updating versions commonly used datasets.

Author: twillia2
Date: Fri Apr 15 09:40:28 MDT 2022
"""
import os
import shutil

from zipfile import ZipFile

import geopandas as gpd
import requests

from revruns import Paths
from revruns.rr import isint


LINKS = {
    "boem_lease_areas": 
        "https://www.boem.gov/BOEM-Renewable-Energy-Shapefiles.zip",
    "wtdb": "",
    "atb": ""
}


class ATB:
    """Methods for retrieving data from the Annual Technology Baseline."""

    def __init__(self, table_fpath=None, year=2030, tech="wind"):
        """Initialize ATB object."""
        self.table_fpath = table_fpath
        self.url = "https://data.openei.org/submissions/4129"  # <------------- Will it consistently be stored at 4129?
        self.tech = tech
        self.year = year

    def __repr__(self):
        """Return ATB representation string."""
        msgs = [f"{k}={v}" for k, v in self.__dict__.items()]
        msg = ", ".join(msgs)
        return f"<ATB object: {msg}>"

    @property
    def sheets(self):
        """Return list of available sheet names."""
        if not self.table_fpath:
            # Get from website
            raise OSError("File not found...")
        else:
            names = get_sheet(self.table_fpath)
        return names

    @property
    def technologies(self):
        """Return lookup of technology - ATB names."""
        lookup = {
            "wind": "Land-Based Wind",
            "solar": "Solar - Utility PV"
        }
        return lookup

    @property
    def table(self):
        """Read in/download ATB data sheets."""
        sheet = self.technologies[self.tech]
        if not self.table_fpath:
            # Get from website
            raise OSError("File not found...")
        else:
            df = get_sheet(self.table_fpath, sheet_name=sheet)
            df = self._parse_table(df)
        return df

    @property
    def capex(self):
        """Return capex for given tech and year."""

    @property
    def opex(self):
        """Return opex for given tech and year."""
        sheet = self.technologies[self.tech]
        table = get_sheet(self.table_fpath, sheet)

    def _parse_table(self, df):
        # Get the ATB sheet into a useable form
        # Find starting y-position and filter
        iy, _ = np.where(df.values == "Base Year")
        assert len(iy) == 1, "_parse_table failed to parse."
        iy = iy[0] + 1
        df = df.iloc[iy:, :]

        # Find starting x-position and filter
        target_col = "Techno-Economic Cost and Performance Parameters"
        _, ix = np.where(df.values == target_col)
        assert len(ix) == 1, "_parse_table failed to parse."
        ix = ix[0] + 2
        df = df.iloc[:, ix:]

        # Adjust the columns
        years = [str(int(y)) for y in df.iloc[0, 2:]]
        cols = ["variable", "scenario", *years]
        df = df.iloc[1:]
        df.columns = cols

        # Fill in missing variable entries
        variables = []
        variable = None
        for value in df["variable"].values:
            df["variable"]
    
        return df


class BOEM:
    """Methods for retrieving BOEM datasets."""

    def lease_areas(self, update=False):
        """Retrieve shapefiles of BOEM Offshore RE Areas."""
        # Update if requested
        if update:
            self.update()
        else:
            # Use existing file if present
            path = None
            if "boem" in Paths.paths and not update:
                file = list(Paths.paths["boem"].glob("*lease*gpkg"))
                if file:
                    path = file

            # If no file, we need to update
            if not path:
                self.update()

        # Now grab the path
        path = list(Paths.paths["boem"].glob("*lease*gpkg"))[0]

        return path

    def date(self, fpath):
        """Return a date if available."""
        date = None
        name = os.path.basename(fpath).replace(".shp", "").lower()
        if any([isint(c) for c in name]):
            date = name = name.split("_", 1)[-1]
            parts = [f"{int(p):02d}" for p in date.split("_")[:3]]
            date = "".join(parts)
        return date

    def name(self, fpath):
        """Build a more reasonable name for each file."""
        name = os.path.basename(fpath).replace(".shp", "").lower()
        date = self.date(fpath)
        if date:
            name = name.split("_", 1)[0]
            name = f"{name}_{date}"
        return name

    def update(self):
        """Update files."""
        # Set up target paths
        url = LINKS["boem_lease_areas"]
        name = os.path.basename(url)
        zdst = Paths.data.joinpath(f"boem/{name}")
        zdstdir = zdst.parent.joinpath("shapefiles")
        zdst.parent.mkdir(exist_ok=True)

        # Download zipfile
        with requests.get(url) as r:
            with open(zdst, "wb") as file:
                file.write(r.content)

        # Unzip zipfile
        with ZipFile(zdst, "r") as zfile:
            zfile.extractall(zdstdir)

        # Convert each to geopackage
        for fpath in zdstdir.glob("*shp"):
            name = self.name(fpath)
            df = gpd.read_file(fpath)
            dst = Paths.paths["boem"].joinpath(name + ".gpkg")
            date = self.date(fpath)
            if date:
                df.attrs["date"] = date  # Does not work, want it to
            df.to_file(dst)

        # Remove zip file and original shapefiles
        os.remove(zdst)
        shutil.rmtree(zdstdir)

    def name_date(self, file):
        """Build a more reasonable name for each file."""
        name = os.path.basename(file).replace(".shp", "").lower()
        if any([isint(c) for c in name]):
            date = name.split("_", 1)[-1]
            name = name.split("_", 1)[0]
            parts = [f"{int(p):02d}" for p in date.split("_")[:3]]
            date = "".join(parts)
        else:
            date = None


if __name__ == "__main__":
    self = BOEM()
