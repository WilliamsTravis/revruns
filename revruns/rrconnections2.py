# -*- coding: utf-8 -*-
"""Create a transmission connection table for an aggregation factor.

TODO: 
    - Upload template rasters to database.

@author travis
"""
import json
import os
import sys
import time
import warnings

from functools import cached_property, lru_cache

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import pathos.multiprocessing as mp

from multiprocessing import Manager
from pandarallel import pandarallel as pdl
from pathlib import Path
from reV.supply_curve.extent import SupplyCurveExtent
from shapely.ops import nearest_points
from tqdm import tqdm

from revruns.rrdb import TechPotential
from tqdm import tqdm

pdl.initialize(progress_bar=True, verbose=False)
tqdm.pandas()
warnings.filterwarnings("ignore", category=FutureWarning)


BOUNDARIES = ("/lustre/eaglefs/projects/rev/transmission/data/"
              "ne_10m_admin_0_countries.shp")
DISTANCE_BUFFER = 5_000
HOME = "/projects/rev/transmission/"
INPUTS = {
    "conus": {
        "name": "United States of America",
        "feature": "rev_conus_trans_lines",
        "excl_fpath": "/projects/rev/data/exclusions/CONUS_Exclusions.h5",
        "crs": "esri:102003"
    },
    "canada": {
        "name": "Canada",
        "feature": "rev_can_trans_lines",
        "excl_fpath": "/projects/rev/data/exclusions/Canada_Exclusions.h5",
        "crs": "esri:102001"
    },
    "mexico": {
        "name": "Mexico",
        "feature": "rev_mex_trans_lines",
        "excl_fpath": "/projects/rev/data/exclusions/Mexico_Exclusions.h5",
        "crs": ("+proj=aea +lat_1=14.5 +lat_2=32.5 +lat_0=24 +lon_0=-105 "
                "+x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs ")  # SR-ORG:28
    }
}

def add_coordinates(sc, points):
    """Add average supply curve area coordinates to point table."""
    # Get column and row indices
    sc_cols, sc_rows = np.meshgrid(
        np.arange(sc.n_cols),
        np.arange(sc.n_rows)
    )
    rows = sc_rows.flatten()
    cols = sc_cols.flatten()

    # Calculate the coordinates for each point
    latlons = self._get_coords(sc, rows, cols)
    points.loc[:, "latitude"] = latlons[:, 0]
    points.loc[:, "longitude"] = latlons[:, 1]

    return points


def build_points(country, resolution, dst):
    """"Build and save a new point data set."""
    # Build initial point indices
    sc = SupplyCurveExtent(
        f_excl=INPUTS[country]["excl_fpath"],
        resolution=resolution
    )
    points = sc.points
    points["sc_point_gid"] = points.index
    points = points.rename({"row_ind": "sc_row_ind",
                            "col_ind": "sc_col_ind"},
                            axis=1)

    # Chunk a list of sc_cols and rows
    points = add_coordinates(sc, points)

    # Convert to target geodataframe
    profile = get_profile(country)
    points = points.rr.to_geo()
    points = clip_land(points, profile, resolution)

    # Save
    dst.parent.mkdir(parents=True, exist_ok=True)
    points.to_file(dst, driver="GPKG")

    return points

def clip_land(points, profile, resolution):
    """Clip out land, subset for CONUS if needed."""
    # Get country geometry
    df = gpd.read_file(BOUNDARIES)
    df = df[["NAME", "geometry"]]
    df = df[df["NAME"] == INPUTS[self.country]["name"]]

    # If CONUS
    if self.country == "conus":
        # Just keep the biggest polygon
        geoms = df["geometry"].iloc[0]
        areas = np.array([g.area for g in geoms])
        idx = np.where(areas == max(areas))[0][0]
        df["geometry"] = geoms[idx]

    # Reproject to local
    df = df.to_crs(INPUTS[self.country]["crs"])
    points = points.to_crs(INPUTS[self.country]["crs"])

    # Clip points file
    buffer = (profile["transform"][0] * resolution) / 2
    points["geometry"] = points["geometry"].buffer(buffer).envelope
    points = points[points["latitude"] != np.inf]
    points = gpd.overlay(points, df, how="intersection")        

    # Reset geometry
    del points["geometry"]
    points = pd.DataFrame(points)
    points = points.rr.to_geo()

    return points


def get_profile(country):
    """Retrieve the georeferencing profile from an exclusion file."""
    with h5py.File(INPUTS[country]["excl_fpath"]) as ds:
        profile = json.loads(ds.attrs["profile"])
    return profile


class Transmission():
    """Methods for connecting supply curve points to transmission."""

    def __init__(self, home=".", country="conus", resolution=128):
        """Initialize Transmission object."""
        self.country = country
        self.db = TechPotential(schema="transmission")
        self.resolution = resolution
        self.home = Path(home).expanduser().absolute()
        self._preflight()

    def __repr__(self):
        """Return representation string."""
        attrs = []
        excludes = ["points", "sc_points", "features"]
        for key, attr in self.__dict__.items():
            if "country" in key:
                print(f"{key}, {attr}")
            if not key.startswith("_") and key not in excludes:
                attrs.append(f"{key}={attr}")
        return f"<{self.__class__.__name__} object: {', '.join(attrs)}>"

    def distance(self, row):
        """Find the closest point on the closest line to the target point."""
        # Grab the supply curve point and features
        point = row["geometry"]
        features = self.features

        # Find the distance of the closest PCA load center and add 5km
        pcadist = self.search_distance(point)
        buffer = row["geometry"].buffer(pcadist)
        close_features = features[features.intersects(buffer)]

        # Only search for features within the PCA radius
        dist_m = []
        dependencies = []
        for i, frow in close_features.iterrows():
            line = frow["geometry"]
            dist = point.distance(line)
            dist_m.append(dist)

            # If it's a substation, we need those lines
            if frow["category"] == "Substation":
                deps = frow["trans_gids"]
                for dep in deps:
                    if dep not in close_features["trans_gid"].values:
                        dependencies.append(dep)

        # Assign distances and convert to miles
        close_features["dist_m"] = dist_m
        close_features["dist_mi"] = close_features["dist_m"] / 1_609.34
        del close_features["dist_m"]

        # Add in substation dependencies with large artifial distance
        missing = features[features["trans_gid"].isin(dependencies)]
        missing["dist_mi"] = 9_999_999
        if missing.shape[0] > 0:
            close_features = pd.concat([close_features, missing])

        # Attach the supply curve point identifiers
        close_features["sc_point_gid"] = row["sc_point_gid"]
        close_features["sc_point_row_id"] = row["sc_row_ind"]
        close_features["sc_point_col_id"] = row["sc_col_ind"]

        return close_features

    def distance_parallel(self, points):
        """Apply the distance function to ncpu chunks of point dataframe."""
        # Split points data frame into chunks
        out = []
        ncpu = mp.cpu_count()
        chunks = np.array_split(points, ncpu)
        with mp.Pool(ncpu) as pool:
            for c in tqdm(pool.imap(self._distance_chunk, chunks), total=ncpu):
                out.append(c)

        # out = points.parallel_apply(self.distance, axis=1)
        connections = pd.concat(out)
        drops = ["geometry", "bgid", "egid", "voltage"]
        connections = connections.drop(drops, axis=1)
        return connections

    def set_features(self):
        """Retrieve transmission feature."""
        table = INPUTS[self.country]["feature"]
        features = self.db.get(table=table)
        features = features.rename({"gid": "trans_gid"}, axis=1)
        features = features.to_crs(INPUTS[self.country]["crs"])
        features["trans_gids"] = features["trans_gids"].apply(json.loads)
        self.features = features

    @cached_property
    def sc_points(self):
        """Retrieve a supply curve point geodataframe."""
        df = gpd.read_file(self.point_fpath)
        df = df.to_crs(INPUTS[self.country]["crs"])
        return df

    def search_distance(self, point):
        """Find the distance of the closest PCA load center and add 5km."""
        pca_df = self.features[self.features["category"] == "PCALoadCen"]
        dists = [point.distance(line) for line in pca_df["geometry"]]
        dist = min(dists) + DISTANCE_BUFFER
        return dist

    def _distance_chunk(self, chunk):
        """Apply self.distance to a chunk of a points dataframe."""
        df = chunk.apply(self.distance, axis=1)
        df = pd.concat(df.to_list())
        return df

    def _get_coords(self, sc, rows, cols):
        """Build coordinates for the supply curve table."""
        lats = []
        lons = []
        rowcols = zip(rows, cols)
        for r, c in tqdm(rowcols, total=rows.shape[0]):
            r = sc.excl_row_slices[r]
            c = sc.excl_col_slices[c]
            lats.append(sc.exclusions["latitude", r, c].mean())
            lons.append(sc.exclusions["longitude", r, c].mean())
        return np.array([lats, lons]).T

    def _find_missing_lines(self, connections):
        """Find missing lines for substations in connection table.

        Parameters
        ----------
        connections : pd.core.fram.DataFrame
            A connections table output containing initial sc point to feature
            connections.

        Returns
        -------
        list : A list of dictionaries with sc_point_gid's as keys and a list of 
               trans_gid's as values. 
        """
        # Defining this here because of laziness
        def _find_missing(row):
            # Find missing line dependencies for a single entry
            sc_gid = row["sc_point_gid"]
            point_conns = connections[connections["sc_point_gid"] == sc_gid]
            line_gids = row["trans_gids"]
            lines = []
            for line_gid in line_gids:
                if line_gid not in point_conns["trans_gid"].values:
                    lines.append(line_gid)
            missing = {sc_gid: lines}
            return missing

        # Subset for unique substations
        substations = connections[connections["category"] == "Substation"]

        # If a point connects to substation, it must also connect to its lines
        values = substations.parallel_apply(_find_missing, axis=1).values
        values = [m for m in values if len(list(m.values())[0]) > 0]
        missing = {}
        for entry in values:
            key = list(entry.keys())[0]
            lines = entry[key]
            if key in missing:
                for line in lines:
                    if line not in missing[key]:
                        missing[key].append(line)
            else:
                missing[key] = lines

        # Set as an attribute for later
        self.missing = missing

        return missing

    def _fix_missing_lines(self, connections):
        """Check for and find fix missing substation line dependencies.

        Parameters
        ----------
        connections: pd.core.frame.DataFrame
            Data frame of supply curve point to transmission line connections.

        Returns
        -------
        pd.core.frame.DataFrame
            The same data frame with added entries if missing line
            dependencies were found.
        """
        # Get lines
        missing = self.missing
        features = self.features()

        # for each Find those features and reformat
        new_rows = []
        for sc_gid, trans_gids in missing.items():
            row = connections[connections["sc_point_gid"] == sc_gid].iloc[0]
            lines = features[features["trans_gid"].isin(trans_gids)]
            for i, line in lines.iterrows():
                new_rows.append({
                    "ac_cap": line["ac_cap"],
                    "cap_left": line["cap_left"],
                    "category": line["category"],
                    "trans_gid": line["trans_gid"],
                    "dist_mi": 9_999_990,
                    "sc_point_gid": row["sc_point_gid"],
                    "sc_point_row_id": row["sc_point_row_id"],
                    "sc_point_col_id": row["sc_point_col_id"]
                })
            mdf = pd.DataFrame(new_rows)
            mdf = mdf.replace("null", np.nan)

        # Append these to the table
        connections = pd.concat([connections, mdf])

        return connections

    def _preflight(self):
        """Initialize more attributes."""
        # Get profile from h5 file
        self.profile = get_profile(self.country)
        self.crs = self.profile["crs"]

        # Build supply curve points if needed
        fname = f"build_{self.resolution:03d}_agg.gpkg"
        self.point_fpath = self.home.joinpath(f"{self.country}/agtables/{fname}")
        if not self.point_fpath.exists():
            build_points(self.country, self.resolution, self.point_fpath)

        # Let's try setting the features as an attribute to avoid mp caching
        self.set_features()

    def main(self, overwrite=False):
        """Build a connection table for a given country and resolution"""
        # Build destination path
        fname = f"connections_{self.resolution:03d}.csv"
        dst = self.home.joinpath(f"{self.country}/{fname}")

        # Check if it exists and needs to be overwritten
        if dst.exists() and overwrite:
            os.remove(dst)       
        if dst.exists():
            print(f"{dst} exists, skipping. Use overwrite option to rebuild.")

        # Build the table
        else:
            # Get the supply curve point geodataframe
            start = time.time()
            points = self.sc_points

            # Apply the distance function to each row
            # print("USING SAMPLE POINTS!")
            # points = points.iloc[:1_000]
            connections = self.distance_parallel(points)
 
            # Add missing substation dependencies
            # n_missing = len(self._find_missing_lines(connections))
            # if n_missing > 0:
            #     print(f"\n{(n_missing)} substations with missing lines, "
            #           "fixing...")
            #     connections = self._fix_missing_lines(connections)

            # Write to file
            connections.to_csv(dst, index=False)

        # Done.
        end = time.time()
        seconds = round(end - start, 3)
        print(f"\nDone: {seconds} seconds.")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        country = sys.argv[1]
    else:
        country = "canada"
    builder = self = Transmission(country=country, home=HOME)
    builder.main(overwrite=True)
