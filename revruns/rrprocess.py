"""Standard post processing steps for reV supply-curve runs."""
import os

from functools import lru_cache

import geopandas as gpd
import pandas as pd

COUNTIES = ("https://www2.census.gov/geo/tiger/TIGER2021/COUNTY/"
            "tl_2021_us_county.zip")  # Update this periodically, or just grab whatever is in the folder each time
STATES = ("https://www2.census.gov/geo/tiger/TIGER2021/STATE/"
          "tl_2021_us_state.zip")  # Update this periodically
HOME = "."  # Will be click main argument
ONSHORE_FULL = ("/projects/rev/data/transmission/build/agtables/"
                "build_{:03d}_agg.csv")
# ONSHORE_FULL = ("/Users/twillia2/Desktop/review_datasets/"
#                 "build_128_agg.csv")
REGIONS = {
    "Pacific": ["Oregon",   "Washington"],
    "Mountain": ["Colorado", "Idaho", "Montana", "Wyoming"],
    "Great Plains": ["Iowa", "Kansas", "Missouri", "Minnesota", "Nebraska",
                     "North Dakota", "South Dakota"],
    "Great Lakes": ["Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin"],
    "Northeast": ["Connecticut", "New Jersey", "New York", "Maine",
                  "New Hampshire", "Massachusetts", "Pennsylvania",
                  "Rhode Island", "Vermont"],
    "California": ["California"],
    "Southwest": ["Arizona", "Nevada", "New Mexico", "Utah"],
    "South Central": ["Arkansas", "Louisiana", "Oklahoma", "Texas"],
    "Southeast": ["Alabama", "Delaware", "District of Columbia", "Florida",
                  "Georgia", "Kentucky", "Maryland", "Mississippi",
                  "North Carolina", "South Carolina", "Tennessee", "Virginia",
                  "West Virginia"]
}

RESOURCE_CLASSES = {
    "windspeed": {
        "onshore": {
            1: [9.01, 100],
            2: [8.77, 9.01],
            3: [8.57, 8.77],
            4: [8.35, 8.57],
            5: [8.07, 8.35],
            6: [7.62, 8.07],
            7: [7.10, 7.62],
            8: [6.53, 7.10],
            9: [5.90, 6.53],
            10: [0, 5.90],
        },
        "offshore": {
            "fixed": {
                1: [9.98, 100],
                2: [9.31, 9.98],
                3: [9.13, 9.31],
                4: [8.85, 9.13],
                5: [7.94, 8.85],
                6: [7.07, 7.94],
                7: [0, 7.07]
            },
            "floating": {
                1: [10.30, 1000],
                2: [10.01, 10.30],
                3: [9.60, 10.01],
                4: [8.84, 9.60],
                5: [7.43, 8.84],
                6: [5.98, 7.43],
                7: [0, 5.98]
            }
        }
    }
}

# This will be a parameter used to convert pixel sums to area
PIXEL_SUM_FIELDS = ["shadow_flicker_120m", "shadow_flicker_135m"]


# Implement this to calculate capex and unit capex if possible
def capex(df):
    """Recalculate capital costs if needed input columns are present."""
    capacity = df["capacity"]
    capacity_kw = capacity * 1000

    fcr = df["mean_fixed_charge_rate"]

    unit_om = df["mean_fixed_operating_cost"] / df["mean_system_capacity"]
    om = unit_om * capacity_kw

    mean_cf = df["mean_cf"]
    lcoe = df["mean_lcoe"]
    if "raw_lcoe" in df:
        raw_lcoe = df["raw_lcoe"]
    else:
        raw_lcoe = lcoe.copy()

    cc = ((lcoe * (capacity * mean_cf * 8760)) - om) / fcr  # Watch out for economies of scale here
    unit_cc = cc / capacity_kw  # $/kw

    raw_cc = ((raw_lcoe * (capacity * mean_cf * 8760)) - om) / fcr  # Watch out for economies of scale here
    raw_unit_cc = raw_cc / capacity_kw  # $/kw

    df["capex"] = cc
    df["unit_capex"] = unit_cc
    df["raw_capex"] = raw_cc
    df["raw_unit_capex"] = raw_unit_cc

    return df


class Process:
    """Methods for performing standard post-processing steps on reV outputs."""

    import addfips
    import geopandas as gpd
    import numpy as np
    import pandas as pd

    from revruns import rr
    from tqdm import tqdm

    def __init__(self, home=".", file_pattern="*_sc.csv", files=None,
                 pixel_sum_fields=PIXEL_SUM_FIELDS, resolution=90):
        """Initialize Post_Process object.

        Parameters
        ----------
        home : str
            Path to directory containing reV supply-curve tables.
        file_pattern : str
            Glob pattern to filter files in home directory. Defaults to
            "*_sc.csv" (reV supply curve table pattern) and finds all such
            files in the driectory tree.
        pixel_sum_fields : list
            List of field name representing pixel sum characterizations to be
            converted to area and percent of available area fields.
        resolution : int
            The resolution in meters of the exclusion/characterization raster.
        """
        self.home = self.rr.Data_Path(home)
        self._files = files
        self.file_pattern = file_pattern
        self.pixel_sum_fields = pixel_sum_fields

    def __repr__(self):
        """Return representation string."""
        msg = f"<Post_Process object: home='{self.home.data_path}'>"
        return msg

    def process(self):
        """Run all post-processing steps on all files."""
        self.assign_regions()
        self.assign_classes()
        self.assign_areas()
        if os.path.exists(ONSHORE_FULL):
            self.assign_counties()

    def assign_area(self, file):
        """Assign area to pixel summed characterizations for a file."""  # Some what inefficient, reading and saving twice but covers all cases
        cols = self._cols(file)
        area_fields = [f"{f}_sq_km" for f in self.pixel_sum_fields]
        pct_fields = [f"{f}_pct" for f in self.pixel_sum_fields]
        target_fields = area_fields + pct_fields
        if any([f not in cols for f in target_fields]):
            for field in self.pixel_sum_fields:
                if field in cols:
                    acol = f"{field}_sq_km"
                    pcol = f"{field}_pct"
                    df = self.pd.read_csv(file, low_memory=False)
                    df[acol] = (df[field] * 90 * 90) / 1_000_000
                    df[pcol] = (df[acol] / df["area_sq_km"]) * 100
                    df.to_csv(file, index=False)

    def assign_areas(self):
        """Assign area to pixel summed characterizations for all files."""
        for file in self.files:
            self.assign_area(file)

    def assign_class(self, file, field="windspeed"):
        """Assign a particular resource class to an sc df."""
        col = f"{field}_class"
        cols = self._cols(file)
        if col not in cols:
            df = self.pd.read_csv(file, low_memory=False)
            rfield = self.resource_field(file, field)
            onmap = RESOURCE_CLASSES[field]["onshore"]
            offmap = RESOURCE_CLASSES[field]["offshore"]

            if "offshore" in cols and "wind" in field and "sub_type" in cols:
                # onshore
                ondf = df[df["offshore"] == 0]
                ondf[col] = df[rfield].apply(self.map_range, range_dict=onmap)

                # offshore
                offdf = df[df["offshore"] == 1]

                # Fixed
                fimap = offmap["fixed"]
                fidf = offdf[offdf["sub_type"] == "fixed"]
                clss = fidf[rfield].apply(self.map_range, range_dict=fimap)

                # Floating
                flmap = offmap["floating"]
                fldf = offdf[offdf["sub_type"] == "floating"]
                clss = fldf[rfield].apply(self.map_range, range_dict=flmap)
                fldf[col] = clss

                # Recombine
                offdf = self.pd.concat([fidf, fldf])
                df = self.pd.concat([ondf, offdf])
            else:
                df[col] = df[rfield].apply(self.map_range, range_dict=onmap)
            df.to_csv(file, index=False)

    def assign_classes(self):
        """Assign resource classes if possible to an sc df."""
        for file in self.files:
            for field in RESOURCE_CLASSES.keys():
                self.assign_class(file, field)

    def assign_county(self, file):
        """Assign the appropriate county FIPS to each point."""
        df = self.pd.read_csv(file)

        # Split in on/offshore
        offdf = df[df["offshore"] == 1]
        ondf = df[df["offshore"] == 0]

        # Fix onshore state/counties assign fips
        if ondf.shape[0] > 0:
            ondf = self._fix_fips(ondf)

        # Assign offshore states, counties, and fips
        if offdf.shape[0] > 0:
            fdf = self.pd.read_csv(ONSHORE_FULL.format(128))  # <-------------- This should technically depend on config agg factor
            fdf = self._fix_fips(fdf)

            offdf = offdf.rr.nearest(fdf,
                                     fields=["cnty_fips", "state", "county"])
            del offdf["dist"]
            del offdf["geometry"]

        df = self.pd.concat([ondf, offdf])
        df = df.sort_values("sc_gid")
        if "geometry" in df:
            del df["geometry"]
            df = pd.DataFrame(df)

        df.to_csv(file, index=False)

    def assign_counties(self):
        """Assign the nearest county FIPS to each point for each file."""
        for file in self.files:
            self.assign_county(file)

    def assign_region(self, file):
        """Assign each point an NREL region."""
        if "nrel_region" not in self._cols(file):
            df = self.pd.read_csv(file)
            df["nrel_region"] = df["state"].map(self.nrel_regions)
            df.to_csv(file, index=False)

    def assign_regions(self):
        """Assign each point an NREL region for each file."""
        for file in self.files:
            self.assign_region(file)

    @property
    @lru_cache()
    def counties(self):
        """Read in and format county dataframe."""
        dst = "/projects/rev/data/conus/shapefiles/counties_2021.gpkg"  # This is not future-proof
        if not os.path.exists(dst):
            df = self.gpd.read_file(COUNTIES)
            sdf = self.gpd.read_file(STATES)
            statefps = dict(zip(sdf["STATEFP"], sdf["NAME"]))
            df["county"] = df["NAME"]
            df["state"] = df["STATEFP"].map(statefps)
            df["cnty_fips"] = df["STATEFP"] + df["COUNTYFP"]
            df = df[["county", "state", "cnty_fips", "geometry"]]
            df = df.to_crs("epsg:4326")
            df.to_file(dst, "GPKG")
        else:
            df = self.gpd.read_file(dst)
        return df

    def map_range(self, x, range_dict):
        """Return class for a given value."""
        for clss, rng in range_dict.items():
            if x > rng[0] and x <= rng[1]:
                return clss

    @property
    def files(self):
        """Return all supply-curve files in home directory."""
        if self._files is None:
            rpattern = f"**/{self.file_pattern}"
            files = self.home.files(rpattern, recursive=True)
        else:
            files = self._files
        return files

    @property
    def nrel_regions(self):
        """Return state, NREL region dictionary."""
        regions = {}
        for region, states in REGIONS.items():
            for state in states:
                regions[state] = region
        return regions

    @lru_cache()
    def resource_field(self, file, field="windspeed"):
        """Return the resource field for a data frame."""
        # There is a new situation we have to account for
        if field == "windspeed":
            df = self.pd.read_csv(file, low_memory=False)
            if all([self.np.isnan(v) for v in df["mean_res"]]):
                if "mean_ws_mean-means" in df.columns:
                    field = "mean_ws_mean-means"
                elif "mean_ws_mean" in df.columns:
                    field = "mean_ws_mean"
                else:
                    field = "mean_res"
            else:
                field = "mean_res"
        return field

    def _fix_fips(self, df):
        """Reassign counties, states, and fips to fix bug."""
        counties = self.counties
        if not isinstance(df, gpd.GeoDataFrame):
            if "geometry" in df:
                del df["geometry"]
            df = df.rr.to_geo()
        if "cnty_fips" in df:
            del df["cnty_fips"]
        columns = list(df.columns) + ["cnty_fips"]
        del df["state"]
        del df["county"]
        df = self.gpd.sjoin(df, counties)
        df = df[columns]
        return df

    def _cols(self, file):
        """Return only the columns of a csv file."""
        return self.pd.read_csv(file, index_col=0, nrows=0).columns


if __name__ == "__main__":
    file = "/shared-projects/rev/projects/weto/fy21/atb/rev/aggregation/onshore/fips_run/fips_run_agg.csv"
    self = Process()