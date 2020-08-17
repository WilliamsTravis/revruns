# -*- coding: utf-8 -*-
"""
Create a transmission connection table for a particular supply-curve
aggregation factor.

@author: twillia2
"""

import copy
import json
import multiprocessing as mp
import os
import shutil
import subprocess as sp

import geopandas as gpd
import numpy as np
import pandas as pd
import revruns as rr

from revruns.constants import TEMPLATES
from shapely.geometry import Point
from scipy.spatial import cKDTree
from tqdm import tqdm

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
tqdm.pandas()


AEA_CRS = ("+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 "
           "+ellps=GRS80 +datum=NAD83 +units=m +no_defs ")


def dist_apply(scdf, linedf):
    """To apply the distance function in parallel."""  # <------------------ Not quite done

    ncpu = mp.cpu_count()
    chunks = np.array_split(scdf.index, ncpu)
    args = [(scdf.loc[idx], linedf) for idx in chunks]
#     args = [(idx, idx * 2) for idx in chunks]
    distances = []
    with mp.Pool() as pool:
        for dists in pool.imap(point_line, args):
            distances.append(dists)
    return distances


def get_allconns(dst):
    """The file is stored on the PostGres data base. How to manage other 
    countries than CONUS?"""

    import getpass
    import pgpasslib
    import psycopg2 as pg

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


def pipeline(config_path):
    cmd = "reV -c " + config_path + " pipeline --monitor"
    p = sp.Popen([cmd], stdout=sp.PIPE, shell=True)
    output, err = p.communicate()  
    print("reV stdout: " + output.decode())
    if err:
        print("reV errors: " + err.decode())
    return output, err

            
def point_line(arg):
    """
    Find the closest transmission line to a point and return a gid and distance.

    Parameters
    ----------
    row : pd.
        A pandas series with a "geometry" column.
    linedf : geopandas.geodataframe.GeoDataFrame
        A geodataframe with a trans_line_gid and shapely geometry objects.

    Returns
    -------
    tuple
        A tuple with the transmission gid, the distance from the point to it,
        and the category of the transmission connection structure.
    """

    # To check that it's working
    print("{} pid running...\n".format(os.getpid()))
    df, linedf = arg

    # Find the closest point on the closest line to the target point
    def single_row(point, linedf):
        distances = [point.distance(l) for l in linedf["geometry"]]
        dmin = np.min(distances)
        idx = np.where(distances == dmin)[0][0]

        # We need this gid and this category
        gid = linedf.index[idx]
        category = linedf["category"].iloc[idx]

        return gid, dmin, category

    distances = df["geometry"].apply(single_row, linedf=linedf)

    return distances


def simple_ag(gen_fpath, dstdir, resolution, res_fpath, allocation, excl_fpath,
              tm_dset):
    """Run aggregation with no exclusions to get full set of sc points.

    gen_fpath = "/lustre/eaglefs/shared-projects/rev/projects/soco/rev/runs/reference/generation/120hh/150ps/150ps_multi-year.h5"
    dstdir = "/scratch/twillia2/connections"
    resolution = 64
    res_fpath = "/datasets/WIND/conus/v1.0.0/wtk_conus_{}.h5"
    allocation = "setallwind"
    excl_fpath = "/projects/rev/data/exclusions/ATB_Exclusions.h5"
    tm_dset = "techmap_wtk"
    """

    # Setup the path to the point file
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
        
        del config["res_class_bins"]

        config["directories"]["logging_directories"] = logdir
        config["directories"]["output_directory"] = rundir
        config["excl_fpath"] = excl_fpath
        config["execution_control"]["allocation"] = allocation
        config["lcoe_dset"] = "lcoe_fcr-means"
        config["cf_dset"] = "cf_mean-means"
        config["res_class_dset"] = "ws_mean-means"
        config["resolution"] = resolution
        config["tm_dset"] = tm_dset
        config["gen_fpath"] = gen_fpath
        config["res_fpath"] = res_fpath
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
        print("Building reV supply curve points...")
        out, err = pipeline(config_pipath)
        if not err:
            shutil.move(point_path, final_path)
        else:
            raise Exception("Aggregation table build failed.")

    return final_path


def transmission(ppath, tpath, allconns_path, crs):

    # Get Points
    scdf = pd.read_csv(ppath, low_memory=False)
    cols = ["sc_point_gid", "sc_row_ind", "sc_col_ind", "latitude",
            "longitude"]
    new_cols = ["sc_point_gid", "sc_point_row_ind", "sc_point_col_ind",
                "latitude", "longitude"]
    scdf = scdf[cols]
    scdf.columns = new_cols
    scdf = scdf.rr.to_geo()

    # Get the transmission table and set ids
    transdf = get_allconns(allconns_path)
    transdf["trans_line_gid"] = transdf.index

    # Match CRS
    crs = transdf.crs.to_wkt()
    scdf = scdf.to_crs(crs)

    # Apply point_line to each row in the dataframe
    linedf = transdf[transdf["geometry"].notna()]
    linedf = linedf[["trans_line_gid", "geometry", "category"]]
    scdf = scdf.iloc[:500]
    distances = dist_apply(scdf, linedf)

    gids = [d[0] for d in distances]
    dists = [d[1] / 1609.34 for d in distances]
    cats = [d[2] for d in distances]

    # Add the new columns
    scdf["trans_line_gid"] = gids
    scdf["dist_mi"] = dists 
    scdf["category"] = cats
    scdf["ac_cap"] = 9999

    # Write to file?
    scdf.to_csv(dst, index=False)

    return dst


# def main(gen_fpath, dstdir, resolution, res_fpath, allocation, excl_fpath,
#          tm_dset, allconns_path, crs):
# """
gen_fpath = "/lustre/eaglefs/shared-projects/rev/projects/soco/rev/runs/reference/generation/120hh/150ps/150ps_multi-year.h5"
dstdir = "/scratch/twillia2/connections"
resolution = 90
res_fpath = "/datasets/WIND/conus/v1.0.0/wtk_conus_{}.h5"
allocation = "setallwind"
excl_fpath = "/projects/rev/data/exclusions/ATB_Exclusions.h5"
tm_dset = "techmap_wtk"
allconns_path = "/projects/rev/data/transmission/shapefiles/conus_allconns.gpkg"
# """

# Build the point file and retrieve the file path
ppath = simple_ag(gen_fpath, dstdir, resolution, res_fpath, allocation,
                  excl_fpath, tm_dset)

# Create the target path
tpath = os.path.join(dstdir, "connections_{:02d}.csv".format(resolution))

# And run transmission
# tpath = transmission(ppath, tpath, allconns_path)

print("Transmission connections table saved to " + tpath)


# if __name__ == "__main__":
#     main()
