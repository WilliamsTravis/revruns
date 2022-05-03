# -*- coding: utf-8 -*-
"""RRdb

Access features from the GDS database.

Created on Wed Apr 27 10:50:54 2022

@author: twillia2
"""
import geopandas as gpd
import getpass
import pgpasslib
import psycopg2 as pg


class Features:
    """Methods for retrieving and storing transmission feature datasets."""

    def __init__(self, schema=None, table=None,country="conus"):
        """Initialize Features object."""
        self.schema = schema
        self.table = table
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

    def get(self, schema=None, table=None):
        """Get a dataset given a schema and table name."""
        # Check/get the schema and table names
        schema = self._schema(schema)
        table = self._table(table)

        # Find a good geometry column
        geom = self.get_geom(table)
        crs = self.get_crs(schema, table)

        # Get the table
        cmd = f"select * from {schema}.{table};"""
        with pg.connect(**self.con_args) as con:
            df = gpd.GeoDataFrame.from_postgis(cmd, con, crs=crs,
                                               geom_col=geom)

        # Rename geometry field to something more reasonable
        df = df.rename({geom: "geometry"}, axis=1)
        df = df.set_geometry("geometry")

        return df

    def get_crs(self, schema=None, table=None):
        """Find the coordinate reference system."""
        schema = self._schema(schema)
        table = self._table(table)
        geom = self.get_geom(table)
        cmd = f"select ST_SRID({geom}) from {schema}.{table} limit 1;"
        with pg.connect(**self.con_args) as con:
            with con.cursor() as cursor:
                cursor.execute(cmd)
                crs = cursor.fetchall()[0][0]

        # It breaks when this one crs is specified as an epsg code
        # Probably someone did that because GDAL used to recongize it
        if str(crs) == "102008":  
            crs = "esri:102008"
        else:
            crs = f"epsg:{crs}"

        return crs

    def get_geom(self, table):
        """Find a good geometry column from a table."""
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
        if "geom" in gcols:
            geom = "geom"
        else: 
            # Why are there multiple geometry columns in these?
            geom = gcols[0] 
        return geom

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
