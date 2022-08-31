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

from cached_property import cached_property
from pathos import multiprocessing as mp
from pyproj import Proj
from rasterio.sample import sample_gen
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
ALLCONNS_PATHS = {
    "onshore": ("/projects/rev/data/transmission/shapefiles/"
                "conus_allconns.gpkg"),
    "offshore": ("/projects/rev/data/transmission/shapefiles/"
                 "conus_allconns_offshore.gpkg")
}
MULTIPLIER_PATHS = {
    "conus": ("/projects/rev/data/transmission/transmults/"
              "conus_trans_multipliers.csv")
}
EXCL_PATHS = {
    "onshore": {
        "conus": "/projects/rev/data/exclusions/ATB_Exclusions.h5",
        "canada": "/projects/rev/data/exclusions/Canada_Exclusions.h5",
        "mexico": "/projects/rev/data/exclusions/Mexico_Exclusions.h5",
    },
    "offshore": {
        "conus": "/projects/rev/data/exclusions/Offshore_Exclusions.h5"
        }
}

# Help printouts
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
OFF_HELP = ("Run for offshore. Defaults to false and onshore. (boolean)")
RES_HELP = ("The resolution factor (or aggregation factor) to use to build "
            "the supply curve points. (int)")
SIMPLE_HELP = ("Use the simple minimum distance connection criteria for each "
               "connection. (boolean)")
TIME_HELP = ("The amount of time in hours needed to generate the supply "
             "curve points used in the connection table. Defaults to 1. (int)")


def multipliers(point_path, dst, country="conus"):
    """Use the connection data frame to create the regional multipliers.

    Parameters
    ----------
    point_path : str
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
    pnts = pd.read_csv(point_path)
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
        raise("Nearest neighbor search for missing mutlipliers failed.")

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


class Features:
    """Methods for retrieving and storing transmission feature datasets."""

    def __init__(self, country="conus"):
        """Initialize Features object."""
        self.country = country

    def __repr__(self):
        """Return representation string."""
        cntry = self.country
        msg = f"<Features instance: country={cntry}>"
        return msg

    @cached_property
    def linedf(self):
        """Get transmission feature file stored on the Postgres data base."""
        # Build query
        if self.country.lower() == "canada":
            name = "rev_can_trans_lines"
        elif self.country.lower() == "mexico":
            name = "rev_mex_trans_lines"
        else:
            name = "rev_conus_trans_lines"
        cmd = f"""select * from transmission.{name};"""

        # Open a connection and retrieve dataset
        with pg.connect(**self.con_args) as con:
            df = gpd.GeoDataFrame.from_postgis(cmd, con, geom_col="geom",
                                               crs=AEA_CRS)

        # Reset geometry column to match other datasets
        df["geometry"] = df["geom"]
        df = df.set_geometry("geometry")
        del df["geom"]
        df["trans_line_gid"] = df["gid"]
        df = df[df["geometry"].notna()]

        return df

    @property
    def con_args(self):
        """Return a database connection."""
        # Setup Postgres Connection Paramters
        user = getpass.getuser()
        host = "gds_edit.nrel.gov"
        dbname = "tech_potential"
        port = 5432
        password = pgpasslib.getpass(host, port, dbname, user)

        # The user might need to set up their password
        if not password:
            msg = ("No password found for the PostGres database needed to "
                   "retrieve the transmission lines dataset. Please install "
                   "pgpasslib (pip) and add this line to ~/.pgpass: \n "
                   "gds_edit.nrel.gov:5432:tech_potential:<user_name>:"
                   "<password>")
            raise LookupError(msg)

        # Build kwargs
        kwargs = {"user": user, "host": host, "dbname": dbname, "user": user,
                 "password": password, "port": port}

        return kwargs

    @property
    def get_reeds(self):
        """Get ReEDS region table."""
        # Retrieve dataset
        cmd = """select * from boundary.conus_nrel_reeds_regions;"""
        with pg.connect(**self.con_args) as con:
            df = gpd.GeoDataFrame.from_postgis(cmd, con, crs=AEA_CRS,
                                               geom_col="the_geom_102008")

        # Clean up table
        df = df[["pca_reg", "demreg", "the_geom_102008", "raster_val", "gid",
                 "id_0", "country"]]
        df = df.rename({"the_geom_102008": "goemetry"}, axis=1)

        return df

    @property
    def table_list(self):
        """Return a list of all available tables in the postgres connection."""
        with pg.connect(**self.con_args) as con:
            with con.cursor() as cursor:
                cursor.execute("select relname from pg_class where "
                               "relkind='r' and relname !~ '^(pg_|sql_)';")
                tables = []
                for lst in cursor.fetchall():
                    print(lst)
                    table = lst[0]
                    tables.append(table)
        return tables


class Build_Points:
    """Methods to build the supply curve points."""

    def __init__(self, allocation, home_path=".", **kwargs):
        """Initialize Build_Points object."""
        self.home_path = os.path.expanduser(os.path.abspath(home_path))
        self.allocation = allocation

    def __repr__(self):
        """Return representation string."""
        home = self.home_path
        alloc = self.allocation
        msg = f"<Build_Points instance: home_path={home}, allocation={alloc}>"
        return msg

    def agg_factor(self, area_sqkm, resolution=90, verbose=False):
        """Return the closest aggregation factor needed for a given area.

        Parameters
        ----------
        area_sqkm : int | float
            An area in square kilometers.
        resolution : int
            The x/y resolution of the grid being aggregated.
        """
        area_sqm = area_sqkm * 1_000_000
        factor = np.sqrt(area_sqm) / resolution
        factor = int(np.round(factor, 0))

        # Now whats the resulting area after rounding
        if verbose:
            area = ((resolution * factor) ** 2) / 1_000_000
            print(f"Closest agg factor:{factor} \nResulting area: {area} sqkm")
        return factor

    def aggregate(self, resolution, memory, time, country, offshore=False):
        """Run aggregation with no exclusions to get full set of sc points."""
        # Setup paths
        name = self.name(resolution)
        dst2 = self.home.join("agtables", name + "_agg.csv", mkdir=True)

        # Make sure the resolution is an integer
        resolution = int(resolution)

        # We'll only need to do this once for each
        if not os.path.exists(dst2):
            dst1 = self.home.join("agtables", name, name + "_agg.csv",
                                  mkdir=True)
            pipeline_path = self.config(resolution, memory, time, country,
                                        offshore)

            # Call reV on this file  <----------------------------------------- I want this to stay running until its done so I can run this and the connections in sequence
            Pipeline.run(pipeline_path, monitor=True, verbose=False)
            shutil.move(dst1, dst2)
            shutil.rmtree(self.home.join("agtables", name))

        return dst2

    def config(self, resolution, memory, time, country, offshore):
        """Create a simple configuration setup for aggregation."""
        ag_path = self._agg_config(resolution, memory, time, country, offshore)
        pipeline_path = self._pipeline_config(ag_path)
        return pipeline_path

    @property
    def home(self):
        """Return a Data_Path about for the home directory."""
        return rr.Data_Path(self.home_path)

    def name(self, resolution):
        """Return the name of the resolution run."""
        resolution = int(resolution)
        return "{}_{:03d}".format(self.home.base, resolution)

    def _agg_config(self, resolution, memory, time, country, offshore):
        # Create the aggregation config
        config = copy.deepcopy(TEMPLATES["ag"])
        name = self.name(resolution)
        run_home = self.home.extend("agtables", mkdir=True)
        logdir = run_home.join(name, "logs")
        config_path = run_home.join(name, "config_aggregation.json")

        # Remove uneeded elements
        del config["data_layers"]
        if "res_class_bins" in config:
            del config["res_class_bins"]

        # There is a big difference between on and offshore
        if offshore:
            shore = "offshore"
            excl = {"dist_to_coast": {"exclude_values": [0]}}
        else:
            shore = "onshore"
            excl = {"albers": {"include_values": [1]}}

        # Fill in needed elements
        config["cf_dset"] = "cf_mean"
        config["directories"]["logging_directories"] = logdir
        config["directories"]["output_directory"] = run_home.join(name)
        config["execution_control"]["allocation"] = self.allocation
        config["execution_control"]["memory"] = memory
        config["execution_control"]["walltime"] = time
        config["excl_dict"] = excl
        config["excl_fpath"] = EXCL_PATHS[shore][country.lower()]
        config["gen_fpath"] = GEN_PATH
        config["lcoe_dset"] = "lcoe_fcr"
        config["power_density"] = 3
        config["res_class_dset"] = "ws_mean"
        config["res_fpath"] = RES_PATH
        config["resolution"] = resolution
        config["tm_dset"] = TM_DSET

        # Write config to file
        with open(config_path, "w") as cfile:
            cfile.write(json.dumps(config, indent=4))

        return config_path

    def _pipeline_config(self, ag_config_path):
        # Create a pipeline config for just aggregation (need it to monitor)
        run_home = self.home.extend("agtables", mkdir=True)
        res = int(os.path.dirname(ag_config_path).split("_")[-1])
        config_path = run_home.join(
            self.name(res),
            "config_pipeline.json",
            mkdir=True
        )

        config = copy.deepcopy(TEMPLATES["pi"])
        config["pipeline"].append({"supply-curve-aggregation": ag_config_path})
        with open(config_path, "w") as file:
            file.write(json.dumps(config, indent=4))

        return config_path


class Connections(Build_Points, Features):
    """Methods for connection supply curve points to transmission features."""

    def __init__(self, allocation, home_path=".", country="conus",
                 offshore=False, simple=False):
        """Initialize Connections object."""
        super(Connections, self).__init__(allocation=allocation,
                                          home_path=home_path)
        self.country=country
        self.offshore=offshore
        self.simple=simple

    def __repr__(self):
        """Return representation string."""
        home = self.home_path
        alloc = self.allocation
        msg = f"<Connections instance: home_path={home}, allocation={alloc}>"
        return msg

    def single_dist(self, row, simple=False):
        """Find the closest point on the closest line to the target point."""
        # Grab the supply curve point
        point = row["geometry"]

        # Find the distance of the closest PCA load center and add 5km
        pcadf = self.linedf[self.linedf["category"] == "PCALoadCen"]
        pcadists = [point.distance(line) for line in pcadf["geometry"]]
        pcadist = min(pcadists) + DISTANCE_BUFFER

        # We are only searching for transmission features within this radius
        buffer = row["geometry"].buffer(pcadist)
        finaldf = self.linedf[self.linedf.intersects(buffer)]
        dist_m = []
        for line in finaldf["geometry"]:
            dist_m.append(point.distance(line))
        finaldf["dist_m"] = dist_m

        # If simple, we only need the closest transmission structure
        if simple:
            dmin = finaldf["dist_m"].min()
            idx = np.where(finaldf["dist_m"] == dmin)[0][0]
            finaldf = finaldf.iloc[idx]

        # Convert to miles
        finaldf["dist_mi"] = finaldf["dist_m"] / 1_609.34
        del finaldf["dist_m"]

        # Attach the supply curve point identifiers
        finaldf["sc_point_gid"] = row["sc_point_gid"]
        finaldf["sc_point_row_id"] = row["sc_row_ind"]
        finaldf["sc_point_col_id"] = row["sc_col_ind"]

        return finaldf

    def par_dist(self, args):
        """Find the closest point on the closest line to the target points."""
        # Unpack arguments
        idx, point_path, simple = args

        # Get the supply curve points and select the indices of this chunk
        scdf = pd.read_csv(point_path, low_memory=False)  #  <----------------- Use chunksize to split these up instead of reading entire frame
        scdf = scdf.loc[idx]
        crs = self.linedf.crs.to_wkt()
        scdf = scdf.rr.to_geo()
        scdf = scdf.to_crs(crs)

        # Apply the distance function to each row and reshape into data frame
        condf = scdf.progress_apply(self.single_dist, simple=simple, axis=1)
        if not simple:
            # This is a dataframe of dataframes
            condfs = [condf.iloc[i] for i in range(condf.shape[0])]
            condf = pd.concat(condfs)
        condf["trans_gids"] = condf["trans_gids"].apply(lambda x: json.dumps(x))

        return condf

    def fix_missing_dependencies(self, condf, linedf, scdf, simple=False):
        """Check for and find fix missing feature line dependencies.

        Parameters
        ----------
        condf: pd.core.frame.DataFrame
            Data frame of supply curve point to transmission line connections.
        linedf:  pd.core.frame.DataFrame
            Data frame of transmission line features.

        Returns
        -------
        pd.core.frame.DataFrame
            The same data frame with added entries if missing line
            dependencies were found.
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

        # Get missings dependencies - catching error
        if not self.simple:
            from reV.handlers.transmission import TransmissionFeatures

            features = TransmissionFeatures(condf)
            if "_check_feature_dependencies" in features.__dict__:
                check = features._check_feature_dependencies
            else:
                check = features.check_feature_dependencies

            missing_dependencies = []
            try:
                check()
            except RuntimeError as e:
                error = str(e)
                missing_str = error[error.index("dependencies:") + 14:]
                missing_dict = ast.literal_eval(missing_str)
                for gids in missing_dict.values():
                    for gid in gids:
                        missing_dependencies.append(gid)

            # Find those features and reformat
            mdf = linedf[linedf["gid"].isin(missing_dependencies)]
            mdf = mdf.replace("null", np.nan)
            mdf = mdf.apply(find_point, scdf=scdf, axis=1)

            # Append these to the table
            condf = pd.concat([condf, mdf])

        return condf

    def connections(self, point_path, dst, simple=False, offshore=False):
        """Find distances to nearby transmission features.

        Parameters
        ----------
        point_path : str
            Path to input supply curve point file.
        dst : str
            Path to output transmission feature file.
        simple : logical
            Use simple connection method.
        offshore : logical
            Create connections for offshore points.

        Returns
        -------
        None
        """
        if not os.path.exists(dst):
            # Get the supply curve points
            print(f"building {dst}...")
            scdf = pd.read_csv(point_path)
            crs = self.linedf.crs.to_wkt()
            scdf = scdf.rr.to_geo()
            scdf = scdf.to_crs(crs)

            # Split into chunks and build args for connection function
            ncpu = mp.cpu_count()
            chunks = np.array_split(scdf.index, ncpu)
            arg_list = [(list(idx), point_path, simple) for idx in chunks]

            # Run the connection functions on each chunk
            condfs = []
            with mp.Pool(ncpu) as pool:
                for condf in pool.imap(self.par_dist, arg_list):
                    condfs.append(condf)
            condf = pd.concat(condfs)

            # It's too big
            condf = condf.drop(["geometry", "bgid", "egid", "gid", "voltage"],
                               axis=1)

            # There might be missing dependencies
            condf = self.fix_missing_dependencies(condf, scdf, simple)

            # Save to file
            print("Saving connections table to " + dst)
            condf.to_csv(dst, index=False)


@click.command()
@click.option("--resolution", "-r", required=True, help=RES_HELP)
@click.option("--allocation", "-a", required=True, help=ALLOC_HELP)
@click.option("--country", "-c", default="CONUS", help=COUNTRY_HELP)
@click.option("--home", "-d", default=".", help=DIR_HELP)
@click.option("--memory", "-m", default=90, help=MEM_HELP)
@click.option("--time", "-t", default=1, help=TIME_HELP)
@click.option("--offshore", "-o", is_flag=True, default=False, help=OFF_HELP)
@click.option("--simple", "-s", is_flag=True, help=SIMPLE_HELP)
@click.option("--just_agg", "-ja", is_flag=True, help=JA_HELP)
def main(resolution, allocation, country, home, memory, time, offshore,
         simple, just_agg):
    """RRCONNECTIONS.

    Create a table with transmission feature connections and distance for a
    given supply curve aggregation factor.

    Exclusion file input and SLURM submission not implemented yet.
    """
    home = os.path.abspath(os.path.expanduser(home))
    country = country.lower()

    # Build the point file and retrieve the file path
    builder = Connections(allocation, home, simple=simple)
    point_path = builder.aggregate(resolution, memory, time, country, offshore)

    if not just_agg:
        # Create the target path
        resolution = int(resolution)
        trans_name = "connections_{:03d}.csv".format(resolution)
        dst = os.path.join(home, trans_name)

        # And run transmission connections
        builder.connections(point_path, dst, simple, offshore)

        # And lastly, the multipliers table
        if country == "conus":
            mult_file = "multipliers_{:03d}.csv".format(resolution)
            mult_path = os.path.join(home, mult_file)
            multipliers(point_path, mult_path)

    else:
        print("A full supply curve point table has been saved to "
              f"{point_path}")
        print("You may now check out a node to generate a "
              + str(resolution)
              + " resolution connections table.")


if __name__ == "__main__":
    allocation = "wetosa"
    country = "canada"
    home = f"/projects/rev/data/transmission/build/{country}"
    resolution = 128
    allocation = "wetosa"
    memory = 90
    time = 1
    simple = False
    offshore = False
    just_agg = False
    exclusions = EXCL_PATHS["onshore"][country]
    self = Connections(allocation, home, country=country)

    # main()
