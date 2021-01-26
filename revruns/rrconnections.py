# -*- coding: utf-8 -*-
"""Create a transmission connection table for an aggregation factor.

To Do:
    - Implement SLURM submission
    - Catch missing line dependencies and add them to the table.
    - Refactor into class methods

@author travis
"""
import ast
import click
import copy
import json
import multiprocessing as mp
import os
import shutil
import warnings

import geopandas as gpd
import getpass
import numpy as np
import pandas as pd
import pgpasslib
import psycopg2 as pg
import rasterio as rio

from pyproj import Proj
from rasterio.sample import sample_gen
from reV.handlers.transmission import TransmissionFeatures
from reV.pipeline import Pipeline
from revruns.constants import TEMPLATES
from revruns import rr
from rex.utilities.hpc import SLURM
from scipy.spatial import cKDTree
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
tqdm.pandas()


AEA_CRS = ("+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 "
           "+ellps=GRS80 +datum=NAD83 +units=m +no_defs ")
DISTANCE_BUFFER = 5000
GEN_PATH = "/projects/rev/data/transmission/build/sample_gen_2013.h5"
RES_PATH = "/datasets/WIND/conus/v1.0.0/wtk_conus_2013.h5"
TM_DSET = "techmap_wtk"

ALLCONNS_PATH = ("/projects/rev/data/transmission/shapefiles/"
                 "conus_allconns.gpkg")
MULTIPLIER_PATHS = {
    "conus": ("/projects/rev/data/transmission/transmults/"
              "conus_trans_multipliers.csv")
}
EXCL_PATHS = {
    "conus": "/projects/rev/data/exclusions/ATB_Exclusions.h5",
    "canada": "/projects/rev/data/exclusions/Canada_Exclusions.h5"
}

ALLOC_HELP = ("The Eagle account/allocation to use to generate the supply "
              "curve points used in the connection table. (str)")
DIR_HELP = ("Destination directory. For consistency the output files will be "
            "named according to their aggregation factor. Defaults to "
            "the current directory. (str)")
COUNTRY_HELP = ("County for which to create connections table. Only 'CONUS' "
                "and 'Canada' are available so far.")

JA_HELP = ("Run the aggregation table and quit. This is here because I "
           "haven't worked out how to submit the connection half of this "
           "process through SLURM after the reV supply curve is completed. "
           "(boolean)")
MEM_HELP = ("The amount of memory in GB needed to generate the supply "
            "curve points used in the connection table. If the default "
            "doesn't work try 179. Defaults to 90 GB. (int)")
RES_HELP = ("The resolution factor (or aggregation factor) to use to build "
            "the supply curve points. (int)")
SIMPLE_HELP = ("Use the simple minimum distance connection criteria for each "
               "connection. (boolean)")
TIME_HELP = ("The amount of time in hours needed to generate the supply "
             "curve points used in the connection table. Defaults to 1. (int)")


def get_allconns(dst, country="conus"):
    """Get file stored on the PostGres data base."""
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
                   "retrieve the transmission lines dataset. Please install "
                   "pgpasslib and add this line to ~/.pgpass: \n "
                   "gds_edit.nrel.gov:5432:tech_potential:<user_name>:"
                   "<password>")
            raise LookupError(msg)

        # Open a connection
        con = pg.connect(user=user, host=host, database=dbase, password=pw,
                         port=port)

        # Retrieve dataset
        if country.lower() == "conus":
            cmd = """select * from transmission.rev_conus_trans_lines;"""
        elif country.lower() == "canada":
            cmd = """select * from transmission.rev_can_trans_lines;"""
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
                   "gds_edit.nrel.gov:5432:tech_potential:<user_name>:"
                   "<password>")
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
    """Return the transmission line data frame."""
    # Get the transmission table and set ids
    transdf = get_allconns(ALLCONNS_PATH)
    transdf["trans_line_gid"] = transdf["gid"]

    # Apply point_line to each row in the dataframe
    linedf = transdf[transdf["geometry"].notna()]

    return linedf


def list_pgtables(con):
    """List all available tables in a postgres connection."""
    cursor = con.cursor()
    cursor.execute("select relname from pg_class where relkind='r' "
                   "and relname !~ '^(pg_|sql_)';")
    tables = []
    for lst in cursor.fetchall():
        table = lst[0]
        tables.append(table)
    return tables


def simple_ag(dstdir, resolution, allocation, memory, time, country):
    """Run aggregation with no exclusions to get full set of sc points."""
    # Setup the path to the point file
    dstdir = os.path.abspath(os.path.expanduser(dstdir))
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
        del config["data_layers"]
        # config["data_layers"] = {
        #     "cnty_fips": {
        #         "dset": "cnty_fips",
        #         "method": "mode"
        #     }
        # }
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
        config["excl_fpath"] = EXCL_PATHS[country.lower()]
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

        # Call reV on this file  <--------------------------------------------- I want this to stay running until its done so I can run this and the connections in sequence, but it seems to run in the back ground still from the cli
        Pipeline.run(config_pipath, monitor=True, verbose=False)
        shutil.move(point_path, final_path)
        shutil.rmtree(os.path.dirname(point_path))

    return final_path


def single_dist(row, linedf, simple=False):
    """Find the closest point on the closest line to the target point."""
    # Grab the supply curve point
    point = row["geometry"]

    # Find the distance of the closest PCA load center and add 5km
    pcadf = linedf[linedf["category"] == "PCALoadCen"]
    pcadists = [point.distance(line) for line in pcadf["geometry"]]
    pcadist = min(pcadists) + DISTANCE_BUFFER

    # We are only searching for transmission features within this radius
    buffer = row["geometry"].buffer(pcadist)
    finaldf = linedf[linedf.intersects(buffer)]
    finaldf["dist_m"] = [point.distance(line) for line in finaldf["geometry"]]

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
    """Find the closest point on the closest line to the target points."""
    # Unpack arguments
    idx, ppath, simple = arg

    # Get the transmission feature data frame
    linedf = get_linedf()

    # Get the supply curve points and select the indices of this chunk
    scdf = pd.read_csv(ppath, low_memory=False)
    scdf = scdf.loc[idx]
    crs = linedf.crs.to_wkt()
    scdf = scdf.rr.to_geo()
    scdf = scdf.to_crs(crs)

    # Apply the distance function to each row and reshape into a data frame
    condf = scdf.progress_apply(single_dist, linedf=linedf, simple=simple,
                                axis=1)
    condfs = [condf.iloc[i] for i in range(condf.shape[0])]
    condf = pd.concat(condfs)

    return condf


# Make sure all line dependencies are present
def fix_missing_dependencies(condf, linedf, scdf, simple=False):
    """Check for and find fix missing feature line dependencies.

    Parameters
    ----------
    condf: pd.core.frame.DataFrame
        A data frame of supply curve point to transmission line connections.
    linedf:  pd.core.frame.DataFrame
        A data frame of transmission line features.

    Returns
    -------
    pd.core.frame.DataFrame
        The same data frame with added entries if missing line dependencies
        were found.
    """
    # This requires an extra function
    def find_point(row, scdf):
        """Find the closest supply curve point to a line."""
        # Find all distances to this line
        line = row["geometry"]
        scdists = [point.distance(line) for point in scdf["geometry"]]

        # Find the closest point from the distances
        dist_m = np.min(scdists)
        point_idx = np.where(scdists == dist_m)[0][0]
        point_row = scdf.iloc[point_idx]

        # These have different field names
        fields = {
            "sc_point_gid": "sc_point_gid",
            "sc_row_ind": "sc_point_row_id",
            "sc_col_ind": "sc_point_col_id"
        }

        # We need that points identifiers
        for key, field in fields.items():
            row[field] = point_row[key]

        # Finally add in distance in miles
        row["dist_mi"] = dist_m / 1609.34

        # We only need these fields
        keepers = ['ac_cap', 'cap_left', 'category', 'trans_gids',
                   'trans_line_gid', 'dist_mi', 'sc_point_gid',
                   'sc_point_row_id', 'sc_point_col_id']
        row = row[keepers]

        return row

    # Get missings dependencies - catching error, how do they do this so fast?
    missing_dependencies = []
    try:
        TransmissionFeatures(condf)._check_feature_dependencies()
    except RuntimeError as e:
        error = str(e)
        missing_str = error[error.index("dependencies:") + 14:]
        missing_dict = ast.literal_eval(missing_str)
        for gids in missing_dict.values():
            for gid in gids:
                missing_dependencies.append(gid)

    # Find those features and reformat (distance, sc_points, etc don't matter)
    mdf = linedf[linedf["gid"].isin(missing_dependencies)]
    mdf = mdf.replace("null", np.nan)
    mdf = mdf.apply(find_point, scdf=scdf, axis=1)

    # Append these to the table
    condf = pd.concat([condf, mdf])

    return condf


def connections(ppath, dst, simple=False):
    """Find distances to nearby transmission features.

    Parameters
    ----------
    ppath : str
        Path to input supply curve point file.
    dst : str
        Path to output transmission feature file.
    simple : logical
        Use simple connection method

    Returns
    -------
    None
    """
    if not os.path.exists(dst):
        # Get the supply curve points
        scdf = pd.read_csv(ppath)
        linedf = get_linedf()
        crs = linedf.crs.to_wkt()
        scdf = scdf.rr.to_geo()
        scdf = scdf.to_crs(crs)

        # Split data frame into chunks and build args for connection function
        ncpu = mp.cpu_count()
        chunks = np.array_split(scdf.index, ncpu)
        args = [(list(idx), ppath, simple) for idx in chunks]

        # Run the connection functions on each chunk
        condfs = []
        with mp.Pool(ncpu) as pool:
            for condf in pool.imap(par_dist, args):
                condfs.append(condf)
        condf = pd.concat(condfs)

        # It's too big
        condf = condf.drop(["geometry", "bgid", "egid", "gid", "voltage"],
                           axis=1)

        # There might be missing dependencies
        condf = fix_missing_dependencies(condf, linedf, scdf, simple)

        # Save to file
        print("Saving connections table to " + dst)
        condf.to_csv(dst, index=False)


def multipliers(ppath, dst, country="conus"):
    """Use the connection data frame to create the regional multipliers.

    Parameters
    ----------
    ppath : str
        Path to input supply curve point file.
    dst : str
        Path to output multiplier file.
    country : str
        String representation for the country the point path represents. So far
        only 'conus' is available.

    Returns
    -------
    None
    """
    # Get supply curve points and multipliers
    pnts = pd.read_csv(ppath)
    mult_lkup = pd.read_csv(MULTIPLIER_PATHS[country])

    # Get the projected coordinates of the points to match the reeds geotiff
    with rio.open('/projects/rev/data/conus/reeds_regions.tif') as fin:
        proj = Proj(fin.crs.to_proj4())
    eastings, northings = proj(pnts.longitude.values, pnts.latitude.values)
    pnts['eastings'] = eastings
    pnts['northings'] = northings

    # Get the reeds regions associated with each point
    with rio.open('/projects/rev/data/conus/reeds_regions.tif') as fin:
        generator = sample_gen(fin, pnts[['eastings', 'northings']].values)
        results = [x[0] for x in generator]
    pnts['reeds_demand_region'] = results
    pnts_mults = pd.merge(pnts,
                          mult_lkup,
                          on='reeds_demand_region',
                          how='left')

    # Make sure the multiplier dimensions match the points
    try:
        assert pnts_mults.shape[0] == pnts.shape[0]
    except AssertionError:
        raise("Supply curve and multiplier point dimensions do not match.")

    # Find points with no multipliers and assign nearest neighbors
    misses = pnts_mults[pd.isnull(pnts_mults.trans_multiplier)]
    hits = pnts_mults[~pd.isnull(pnts_mults.trans_multiplier)]
    hits_tree = cKDTree(hits[['eastings', 'northings']].values)
    dist, idx = hits_tree.query(misses[['eastings', 'northings']].values)
    nearests = hits.iloc[idx].trans_multiplier.values
    pnts_mults.loc[misses.index.values, 'trans_multiplier'] = nearests
    try:
        n_missing = len(pnts_mults[pd.isnull(pnts_mults.trans_multiplier)])
        assert n_missing == 0
    except AssertionError:
        raise("Nearest neighbor search to fill in missing mutlipliers failed.")

    # Save
    cols = ['sc_point_gid', 'trans_multiplier']
    df = pnts_mults[cols]
    df.to_csv(dst)


def slurm(resolution, allocation, time, memory, country, dstdir):
    """Submit this job to slurm.

    This isn't worked in yet. I have to make it so this won't submit itself.
    """
    # Build command
    name = "transmission_{:03d}".format(resolution)
    template = "rrconnections -r {} -a {} -t {} -m {} -c {} -d {}"
    cmd = template.format(
        resolution,
        allocation,
        time,
        memory,
        country,
        dstdir
    )

    # Submit command to slurm - there redundancy because of aggregation step
    slurm = SLURM()
    slurm.sbatch(cmd=cmd, alloc=allocation, walltime=time, memory=memory,
                 name=name, stdout_path='./stdout',
                 keep_sh=False, conda_env="revruns", module=None,
                 module_root=None)


@click.command()
@click.option("--resolution", "-r", required=True, help=RES_HELP)
@click.option("--allocation", "-a", required=True, help=ALLOC_HELP)
@click.option("--country", "-c", default="CONUS", help=COUNTRY_HELP)
@click.option("--dstdir", "-d", default=".", help=DIR_HELP)
@click.option("--memory", "-m", default=90, help=MEM_HELP)
@click.option("--time", "-t", default=1, help=TIME_HELP)
@click.option("--simple", "-s", is_flag=True, help=SIMPLE_HELP)
@click.option("--just_agg", "-ja", is_flag=True, help=JA_HELP)
def main(resolution, allocation, country, dstdir, memory, time, simple,
         just_agg):
    """RRCONNECTIONS.

    Create a table with transmission feature connections and distance for a
    given supply curve aggregation factor.

    Exclusion file input not implemented yet.

    Sample Parameters:
    resolution = 64
    dstdir = "/projects/rev/data/transmission/build/canada"
    allocation = "wetosa"
    memory = 90
    time = 1
    country = "canada"
    exclusions = EXCL_PATHS[country]
    simple = False
    """
    dstdir = os.path.abspath(os.path.expanduser(dstdir))
    country = country.lower()

    # Build the point file and retrieve the file path
    ppath = simple_ag(dstdir, resolution, allocation, memory, time, country)

    if not just_agg:
        # Create the target path
        resolution = int(resolution)
        tname = "connections_{:03d}.csv".format(resolution)
        tpath = os.path.join(dstdir, tname)

        # And run transmission connections
        connections(ppath, tpath, simple)

        # And lastly, the multipliers table
        if country == "conus":
            mfile = "multipliers_{:03d}.csv".format(resolution)
            mpath = os.path.join(dstdir, mfile)
            multipliers(ppath, mpath)

    else:
        print("A full supply curve point table has been saved to " + ppath)
        print("You may now check out a node to generate a " + str(resolution) +
              " resolution connections table.")


if __name__ == "__main__":
    main()
