# -*- coding: utf-8 -*-
"""Create a transmission connection table for an aggregation factor.

@author travis
"""
import warnings

from functools import cached_property

import pandas as pd

from tqdm import tqdm

from revruns.rrdb import TechPotential


warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
tqdm.pandas()

FEATURE_NAMES = {
    "conus": "",
    "canada": "",
    "mexico": ""
}



class Transmission(TechPotential):
    """Methods for connecting supply curve points to transmission."""

    def __init__(self, resolution=128, country="conus"):
        """Initialize Transmission object."""
        super().__init__(schema="transmission", country=country)
        self.resolution = resolution
        self.country = country

    def __repr__(self):
        """Return representation string."""
        attrs = [f"{key}={attr}" for key, attr in self.__dict__.items()]
        return f"<{self.__class__.__name__} object: {', '.join(attrs)}>"

    @cached_property
    def points(self):
        """Build supply curve point table."""

    @cached_property
    def template(self):
        """Retrieve template raster for the given country."""

    @cached_property
    def features(self):
        """Retrieve transmission feature."""
        

if __name__ == "__main__":
    self = Transmission()
