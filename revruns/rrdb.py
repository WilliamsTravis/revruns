# -*- coding: utf-8 -*-
"""RRdb

Access features from the GDS database.

Created on Wed Apr 27 10:50:54 2022

@author: twillia2
"""
import json
import os

from functools import lru_cache

import geopandas as gpd
import getpass
import numpy as np
import pgpasslib
import psycopg2 as pg
import pyproj
import rasterio as rio

from tqdm import tqdm

from revruns.rr import isint



class Rasters:
    """Methods for extracting rasters from the GDS_EDIT database."""

    def __init__(self, schema=None, table=None):
        """Initialize Rasters object."""
        self.schema = schema
        self.table = table

    @lru_cache(1)
    def _raster_metas(self, schema=None, table=None):
        """Get raster meta data, need to find out how to distinguish."""
        schema = self._schema(schema)
        table = self._table(table)
        cmd = (f"SELECT rid, (ST_MetaData(rast)).* FROM {schema}.{table};")
        with pg.connect(**self.con_args) as con:
            with con.cursor() as cursor:
                cursor.execute(cmd)
                out = cursor.fetchall()
        fields = ["xmin", "ymax", "nx", "ny", "xres", "yres", "xrot",
                  "yrot", "epsg", "nbands"]
        rids = [o[0] for o in out]
        metas = [dict(zip(fields, o[1:])) for o in out]
        metas = dict(zip(rids, metas))
        return metas

    def _raster_points(self, schema=None, table=None):
        """Return x and y coordinate lists for a raster."""
        meta = self._raster_meta(schema, table)
        xs = [meta["xmin"] + (meta["rx"] * i) for i in range(0, meta["nx"])]
        ys = [meta["ymax"] + (meta["ry"] * i) for i in range(0, meta["ny"])]
        return xs, ys

    def _raster(self, schema=None, table=None):
        """Read in a raster."""
        # Infer schema and table
        schema = self._schema(schema)
        table = self._table(table)

        # Read in raster chunks
        cmd = f"SELECT rid, (ST_DumpValues(rast)).* FROM {schema}.{table};"
        with pg.connect(**self.con_args) as con:
            with con.cursor() as cursor:
                cursor.execute(cmd)
                out = cursor.fetchall()

        # Sort by rids
        values = {v[0]: v[1:] for v in out}
        metas = self._raster_metas()
        values = [values[key] for key in metas.keys()]

        # Write to scratch for now
        self._combine(values, table)

    def _combine(self, values, table):
        """Combine raster array chunks into single array, write to raster."""
        # Get the raster ids and arrays
        chunks = [v[-1] for v in values]
        chunks = np.array(chunks)

        # Get the template for the full array
        template, profile = self._template(str(chunks[0].dtype))

        # Insert chunks into array
        xmin = profile["transform"][2]
        ymax = profile["transform"][-1]
        metas = self._raster_metas()
        missing = []
        for i, chunk in enumerate(chunks):
            meta = list(metas.values())[i]
            xix = round((meta["xmin"] - xmin) / meta["xres"])
            yix = round((meta["ymax"] - ymax) / meta["yres"])
            if xix < template.shape[1]:
                template[yix: yix + meta["ny"], xix: xix + meta["nx"]] = chunk
            else:
                print(f"missing chunk: {i}")
                missing.append(i)

        dst = f"/scratch/twillia2/{self.table}.tif" 
        if os.path.exists(dst):
            os.remove(dst)
        with rio.open(dst,"w", **profile) as file:
            file.write(template, 1)

    def _template(self, dtype="float64"):
        """Build a template and rasterio profile the full raster."""
        metas = self._raster_metas()
        meta = metas[next(iter(metas))]
        xs = [value["xmin"] for value in metas.values()]
        ys = [value["ymax"] for value in metas.values()]

        xres = abs(meta["xres"])
        yres = abs(meta["yres"])

        width = round((max(xs) - min(xs)) / xres) + meta["nx"]
        height = round((max(ys) - min(ys)) / yres) + meta["ny"]

        if "float" in dtype:
            na = np.finfo(dtype).max
        else:
            na = np.iinfo(dtype).max

        template = np.zeros((height, width)) + na

        transform = (
            meta["xres"],
            meta["xrot"],
            min(np.unique(xs)),
            meta["yrot"],
            meta["yres"],
            max(np.unique(ys))
        )
        profile = {
            "driver": "GTiff",
            "dtype": dtype,
            "nodata": na,
            "width": template.shape[1],
            "height": template.shape[0],
            "count": 1,
            "crs": pyproj.CRS.from_epsg(meta["epsg"]),
            "transform": transform,
            "blockxsize": 128,
            "blockysize": 128,
            "tiled": True,
            "compress": "lzw",
            "interleave": "band"
        }

        return template, profile


class TechPotential(Rasters):
    """Methods for retrieving and storing transmission feature datasets."""

    def __init__(self, schema=None, table=None, country="conus"):
        """Initialize Features object."""
        super().__init__(schema, table)
        self.country = country

    def __repr__(self):
        """Return representation string."""
        attrs = ", ".join([f"{k}={v}" for k, v in self.__dict__.items()])
        name = self.__class__.__name__
        msg = f"<{name} instance: {attrs}>"
        return msg

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

    def schemas(self, grep=None):
        """Return a list of all available schemas in database."""
        with pg.connect(**self.con_args) as con:
            with con.cursor() as cursor:
                cursor.execute("select schema_name "
                               " from information_schema.schemata;")
                schemas = []
                for lst in cursor.fetchall():
                    schema = lst[0]
                    schemas.append(schema)

        if grep:
            schemas = [schema for schema in schemas if grep in schema]

        return schemas

    def tables(self, schema=None, grep=None):
        """Return a list of all available tables in the postgres connection."""
        schema = self._schema(schema)
        with pg.connect(**self.con_args) as con:
            with con.cursor() as cursor:
                cursor.execute("select * from pg_tables "
                               f"where schemaname = '{schema}';")
                tables = []
                for lst in cursor.fetchall():
                    table = lst[1]
                    tables.append(table)

        if grep:
            tables = [table for table in tables if grep in table]

        return tables

    def get(self, schema=None, table=None, crs=None):
        """Get a dataset given a schema and table name."""
        # Check/get the schema and table names
        schema = self._schema(schema)
        table = self._table(table)

        # Find a good geometry column
        geom = self.get_geom(table)
        crs = self.get_crs(schema, table, crs)

        # Get the table
        cmd = f"select * from {schema}.{table};"""
        with pg.connect(**self.con_args) as con:
            df = gpd.GeoDataFrame.from_postgis(cmd, con, crs=crs,
                                                geom_col=geom)

        # Rename geometry field to something more reasonable
        df = df.rename({geom: "geometry"}, axis=1)
        df = df.set_geometry("geometry")

        # Remove other geom columns
        for gcol in self.get_geoms(table):
            if gcol in df:
                del df[gcol]

        # Jsonify any illegal types (just lists for now)
        for col in df.columns:
            if any([isinstance(r, list) for r in df[col].values]):
                df[col] = df[col].apply(lambda x: json.dumps(x))

        return df

    def get_crs(self, schema=None, table=None, crs=None):
        """Find the coordinate reference system."""
        schema = self._schema(schema)
        table = self._table(table)
        geom = self.get_geom(table)
        cmd = f"select ST_SRID({geom}) from {schema}.{table} limit 1;"
        with pg.connect(**self.con_args) as con:
            with con.cursor() as cursor:
                cursor.execute(cmd)
                srs = cursor.fetchall()[0][0]

        # It breaks when this one crs is specified as an epsg code
        # Probably someone did that because GDAL used to recongize it
        if srs == 102008:
            code = "esri"
        else:
            code = "epsg"

        try:
            pyproj.CRS(f"{code}:{srs}")
            srs = f"{code}:{srs}"
        except pyproj.exceptions.CRSError:
            if crs:
                srs = crs
            else:
                raise pyproj.exceptions.CRSError(f"Can't find srid {srs}")

        return srs

    def get_geom(self, table=None):
        """Find a good geometry column from a table."""
        # Get all geometry columns
        gcols = self.get_geoms(table)

        # There might be a code in the column name, which is best
        code_gcols = []
        for gcol in gcols:
            if "centroid" not in gcol:
                code = [part for part in gcol.split("_") if isint(part)]
                if code:
                    code_gcols.append(gcol)

        # Apply priority selection
        if code_gcols:
            if len(code_gcols) == 1:
                geom = code_gcols[0]
        elif "geom" in gcols:
            geom = "geom"
        elif "the_geom" in gcols:
            geom = "the_geom"
        else: 
            geom = gcols[0] 

        return geom

    def get_geoms(self, table=None):
        """Find all geometry columns."""
        table = self._table(table)
        cmd = ("select column_name from information_schema.columns "
               f"where table_name = '{table}';")
        with pg.connect(**self.con_args) as con:
            with con.cursor() as cursor:
                cursor.execute(cmd)
                columns = []
                for lst in cursor.fetchall():
                    column = lst[0]
                    columns.append(column)
        gcols = [col for col in columns if "geom" in col]
        return gcols

    def _schema(self, schema=None):
        """Use schema attribute if no argument given."""
        if not schema:
            if not self.schema:
                raise KeyError("No schema provided. Set as attribute or "
                               "argument.")
            else:
                schema = self.schema
        return schema

    def _table(self, table=None):
        """Use table attribute if no argument given."""
        if not table:
            if not self.table:
                raise KeyError("No table provided. Set as attribute or "
                               "argument.")
            else:
                table = self.table
        return table


if __name__ == "__main__":
    schema = "land_use"
    table = "can_esa_globcover30"
    country = "canada"
    self = TechPotential(schema, table, country)
    # _ = self._raster()
