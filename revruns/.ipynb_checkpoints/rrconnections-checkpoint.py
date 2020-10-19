# -*- coding: utf-8 -*-
"""Create a transmission connection table for a particular supply-curve
 aggregation factor.
"""

import click
import copy
import io
import json
import multiprocessing as mp
import os
import shutil
import subprocess as sp
import warnings

from contextlib import redirect_stdout

import geopandas as gpd
import getpass
import numpy as np
import pandas as pd
import pgpasslib
import psycopg2 as pg
import revruns as rr

from reV.pipeline import Pipeline
from revruns.constants import TEMPLATES
from shapely.geometry import Point
from scipy.spatial import cKDTree
from tqdm import tqdm


warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
tqdm.pandas()


AEA_CRS = ("+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 "
           "+ellps=GRS80 +datum=NAD83 +units=m +no_defs ")
GEN_PATH = "/projects/rev/data/transmission/build/sample_gen_2013.h5"
RES_PATH = "/datasets/WIND/conus/v1.0.0/wtk_conus_2013.h5"
EXCL_PATH = "/projects/rev/data/exclusions/ATB_Exclusions.h5"
TM_DSET = "techmap_wtk"
ALLCONNS_PATH = ("/projects/rev/data/transmission/shapefiles/"
                 "conus_allconns.gpkg")

RES_HELP = ("The resolution factor (or aggregation factor) to use to build "
            "the supply curve points. (int)")
DIR_HELP = ("Destination directory. For consistency the output files will be "
            "named according to their aggregation factor. Defaults to "
            " the current directory. (str)")  # <------------------------------- We should also include the exclusion resolution
ALLOC_HELP = ("The Eagle account/allocation to use to generate the supply "
              "curve points used in the connection table. (str)")
MEM_HELP = ("The amount of memory in GB needed to generate the supply "
            "curve points used in the connection table. If the default doesn't "
            " work try 179. Defaults to 90 GB. (int)")
TIME_HELP = ("The amount of time in hours needed to generate the supply "
             "curve points used in the connection table. Defaults to 1. (int)")
EXCL_HELP = ("The exclusion dataset to use for the supply curve point "
             "aggregation. Defaults to " + EXCL_PATH + ". (str)")
SIMPLE_HELP = ("Use the simple minimum distance connection criteria for each "
               "connection. (boolean)")
JA_HELP = ("Run the aggregation table and quit. This is here because I haven't "
           "worked out how to submit the connection half of this process "
           "through SLURM after the reV supply curve is completed. (boolean)")


def get_allconns(dst):
    """The file is stored on the PostGres data base. How to manage other 
    countries than CONUS?"""

    # Write to file for later
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True) 

        # Setup Postgres Connection Paramters
        user = getpass.getuser()
        host = "gds_edit.nrel.gov"
        dbase = "tech_potential"
        port = 5432
        pw = pgpasslib.getpass(host, port, dbase, user)

        # The user might need to set up their password
        if not pw:
            msg = ("No password found for the PostGres database needed to retrieve "
                   "the transmission lines dataset."
                   " Please install pgpasslib and add this line to ~/.pgpass: \n "
                   "gds_edit.nrel.gov:5432:tech_potential:<user_name>:<password>")
            raise LookupError(msg)

        # Open a connection
        con = pg.connect(user=user, host=host, database=dbase, password=pw,
                         port=port)

        # Retrieve dataset
        cmd = """select * from transmission.rev_conus_trans_lines;"""
        df = gpd.GeoDataFrame.from_postgis(cmd, con, geom_col="geom",
                                           crs=AEA_CRS)
        con.close()

        # Can't have lists in geopackage columns
        for c in df.columns:
            first_value = df[c][~pd.isnull(df[c])].iloc[0]
            if isinstance(first_value, list):
                df[c] = df[c].apply(lambda x: json.dumps(x))

        # Save
        df.to_file(dst, driver="GPKG")

        return df

    else:
        df = gpd.read_file(dst)
        return df


def get_reeds(dst):
    """We need this to join to the multipliers table. It's by reeds region."""

    # Write to file for later
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True) 

        # Setup Postgres Connection Paramters
        user = getpass.getuser()
        host = "gds_edit.nrel.gov"
        dbase = "tech_potential"
        port = 5432
        pw = pgpasslib.getpass(host, port, dbase, user)

        # The user might need to set up their password
        if not pw:
            msg = ("No password found for the PostGres database needed to "
                   "retrieve the transmission lines dataset."
                   " Please install pgpasslib and add this line to "
                   "~/.pgpass: \n "
                   "gds_edit.nrel.gov:5432:tech_potential:<user_name>:<password>")
            raise LookupError(msg)

        # Open a connection
        con = pg.connect(user=user, host=host, database=dbase, password=pw,
                         port=port)

        # Retrieve dataset
        cmd = """select * from transmission.conus_nrel_reeds_region;"""
        df = gpd.GeoDataFrame.from_postgis(cmd, con, geom_col="geom",
                                           crs=AEA_CRS)
        con.close()

        # Can't have lists in geopackage columns
        for c in df.columns:
            first_value = df[c][~pd.isnull(df[c])].iloc[0]
            if isinstance(first_value, list):
                df[c] = df[c].apply(lambda x: json.dumps(x))

        # Save
        df.to_file(dst, driver="GPKG")

        return df

    else:
        df = gpd.read_file(dst)
        return df


def get_linedf():

    # Get the transmission table and set ids
    transdf = get_allconns(ALLCONNS_PATH)
    transdf["trans_line_gid"] = transdf["gid"]

    # Apply point_line to each row in the dataframe
    linedf = transdf[transdf["geometry"].notna()]

    return  linedf


# def pipeline(config_path):
#     import logging
#     import sys
#     logging.basicConfig(stream=sys.stdout, level=logging.INFO)

#     f = io.StringIO()
#     with contextlib.suppressdout(f):
#         p = Pipeline.run(config_path, monitor=True, verbose=False)
#     out = f.getvalue()
#     return out


def simple_ag(dstdir, resolution, allocation, memory, time):
    """Run aggregation with no exclusions to get full set of sc points."""

    # Setup the path to the point file
    dstdir = os.path.expanduser(dstdir)
    dstdir = os.path.abspath(dstdir)
    resolution = int(resolution)
    point_dir = "{}_{:03d}".format(os.path.basename(dstdir), resolution)
    point_file = point_dir + "_agg.csv"
    tabledir = os.path.join(dstdir, "agtables")
    point_path = os.path.join(tabledir, point_dir, point_file)
    rundir = os.path.dirname(point_path)
    final_path = os.path.join(tabledir, point_file)

    # We'll only need to do this once for each
    if not os.path.exists(final_path):

        os.makedirs(rundir, exist_ok=True)

        # Create a simple configuration file for aggregation
        config = copy.deepcopy(TEMPLATES["ag"])

        logdir = os.path.join(rundir, "logs")
        config_agpath = os.path.join(rundir, "config_aggregation.json")

        config["execution_control"]["memory"] = memory
        config["execution_control"]["walltime"] = time
        config["data_layers"]= {
            "cnty_fips": {
                "dset": "cnty_fips",
                "method": "mode"
            }
        }
        config["excl_dict"] = {
            "albers": {
                "include_values": [
                    1
                ]
            }
        }

        if "res_class_bins" in config:
            del config["res_class_bins"]

        config["directories"]["logging_directories"] = logdir
        config["directories"]["output_directory"] = rundir
        config["excl_fpath"] = EXCL_PATH
        config["execution_control"]["allocation"] = allocation
        config["lcoe_dset"] = "lcoe_fcr"
        config["cf_dset"] = "cf_mean"
        config["res_class_dset"] = "ws_mean"
        config["resolution"] = resolution
        config["tm_dset"] = TM_DSET
        config["gen_fpath"] = GEN_PATH
        config["res_fpath"] = RES_PATH
        config["power_density"] = 3

        # Write config to file
        with open(config_agpath, "w") as cfile:
            cfile.write(json.dumps(config, indent=4))

        # Create a pipeline config for just aggregation (need it to monitor)
        config_pipath = os.path.join(rundir, "config_pipeline.json")
        config = copy.deepcopy(TEMPLATES["pi"])
        config["pipeline"].append({"supply-curve-aggregation": config_agpath})
        with open(config_pipath, "w") as cfile:
            cfile.write(json.dumps(config, indent=4))

        # Call reV on this file
        # out = pipeline(config_pipath)
        Pipeline.run(config_pipath, monitor=True, verbose=False)
        shutil.move(point_path, final_path)
        shutil.rmtree(os.path.dirname(point_path))

    return final_path


# Find the closest point on the closest line to the target point
def single_dist(row, linedf, simple=False):

    # Grab the supply curve point
    point = row["geometry"]

    # Find the distance of the closest PCA load center and add 5km
    pcadf = linedf[linedf["category"] == "PCALoadCen"]
    pcadists = [point.distance(l) for l in pcadf["geometry"]]
    pcadist = min(pcadists) + 5000 

    # We are only searching for transmission features within this radius
    buffer = row["geometry"].buffer(pcadist)
    finaldf = linedf[linedf.intersects(buffer)] 
    finaldf["dist_m"] = [point.distance(l) for l in finaldf["geometry"]]

    # If simple, we only need the closest transmission structure
    if simple:
        dmin = finaldf["dist_m"].min()
        idx = np.where(finaldf["dist_m"] == dmin)[0][0]
        finaldf = finaldf.loc[idx]

    # Convert to miles
    finaldf["dist_mi"] = finaldf["dist_m"] / 1609.34
    del finaldf["dist_m"]

    # Attach the supply curve point identifiers
    finaldf["sc_point_gid"] = row["sc_point_gid"]
    finaldf["sc_point_row_id"] = row["sc_row_ind"]
    finaldf["sc_point_col_id"] = row["sc_col_ind"]

    return finaldf


def par_dist(arg):

    # global linedf  # <-------------------------------------------------------- I can't define this within the connections, but these are specific to it :/
    # global scdf

    idx, ppath, simple = arg
    linedf = get_linedf()
    scdf = pd.read_csv(ppath, low_memory=False)
    scdf = scdf.loc[idx]
    crs = linedf.crs.to_wkt()
    scdf = scdf.rr.to_geo()
    scdf = scdf.to_crs(crs)

    condf = scdf.progress_apply(single_dist, linedf=linedf, simple=simple,
                                axis=1)
    condfs = [condf.iloc[i] for i in range(condf.shape[0])]
    condf = pd.concat(condfs)

    return condf


def connections(ppath, tpath, simple=False):
    """Find distances to nearby transmission lines or stations."""

    if not os.path.exists(tpath):

        scdf = pd.read_csv(ppath)
        linedf = get_linedf()
        crs = linedf.crs.to_wkt()
        scdf = scdf.rr.to_geo()
        scdf = scdf.to_crs(crs)

        ncpu = mp.cpu_count()
        chunks = np.array_split(scdf.index, ncpu)
        args = [(list(idx), ppath, simple) for idx in chunks]
        condfs = []
        with mp.Pool(ncpu) as pool:
            for condf in pool.imap(par_dist, args):
                condfs.append(condf)
        condf = pd.concat(condfs)

        # It's too big
        condf = condf.drop(["geometry" ,"bgid", "egid", "gid", "voltage"],
                           axis=1)

        print("Saving connections table to " + tpath)
        condf.to_csv(tpath, index=False)

    else:
        condf = pd.read_csv(tpath)

    return condf


def multipliers():
    """Use the connection data frame to create the regional multipliers."""



@click.command()
@click.option("--resolution", "-r", required=True, help=RES_HELP)
@click.option("--dstdir", "-d", default=".", help=DIR_HELP)
@click.option("--allocation", "-a", required=True, help=ALLOC_HELP)
@click.option("--memory", "-m", default=90, help=MEM_HELP)
@click.option("--time", "-t", default=1, help=TIME_HELP)
@click.option("--exclusions", "-e", default=EXCL_PATH, help=EXCL_HELP)
@click.option("--simple", "-s", is_flag=True, help=SIMPLE_HELP)
@click.option("--just_agg", "-ja", is_flag=True, help=JA_HELP)
def main(resolution, dstdir, allocation, memory, time, exclusions, simple, 
         just_agg):
    """
    rrconnections

    Create a table with transmission feature connections and distance for a
    given supply curve aggregation factor.

    Exclusion file input not implemented yet.

    Sample Parameters:
    resolution = 77
    dstdir = "/projects/rev/data/transmission/build"
    allocation = "setallwind"
    memory = 90
    time = 1
    exclusions = EXCL_PATH
    simple = False
    """

    dstdir = os.path.abspath(os.path.expanduser(dstdir))

    # Build the point file and retrieve the file path
    ppath = simple_ag(dstdir, resolution, allocation, memory, time)

    if not just_agg:

        # Create the target path
        resolution = int(resolution)
        tpath = os.path.join(dstdir, "connections_{:03d}.csv".format(resolution))

        # And run transmission connections
        condf = connections(ppath, tpath, simple)

        # And lastly, the multipliers table (just using 1 for everything for now)
        mpath = os.path.join(dstdir, "multipliers_{:03d}.csv".format(resolution))
        mdf = condf[["sc_point_gid"]].drop_duplicates()
        mdf["transmission_multiplier"] = 1
        mdf.to_csv(mpath, index=False)

    else:
        print("A full supply curve point table has been saved to " + ppath)
        print("You may now check out a node to generate a " + str(resolution) +
              " resolution connections table.")


if __name__ == "__main__":
    main()
