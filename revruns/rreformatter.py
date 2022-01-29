# -*- coding: utf-8 -*-
"""Reformat a shapefile or raster into a reV raster using a template.

Ideally, any raster or vector format will work with this.

Created on Tue Apr 27 16:47:10 2021

@author: twillia2
"""
TEMPLATE = "/projects/rev/data/conus/wind_deployment_potential/albers.tif"
FILE = ("/projects/rev/data/conus/offshore/dist_to_coast_offshore.tif")
DST = ("/projects/rev/data/conus/dist_to_coast.tif")


class Exclusions:
    """Build or add to an HDF5 Exclusions dataset."""

    import h5py
    import numpy as np
    import rasterio as rio

    from pyproj import Transformer
    from rasterio.errors import RasterioIOError

    def __init__(self, excl_fpath, string_values={}):
        """Initialize Exclusions object.

        Parameters
        ----------
            excl_fpath : str
                Path to target HDF5 reV exclusion file.
            string_values : str | dict
                Dictionary or path dictionary of raster value, key pairs
                derived from shapefiles containing string values (optional).
        """
        super().__init__(self, **kwargs)
        self.excl_fpath = excl_fpath
        self.string_values = string_values
        self._preflight()
        self._initialize_h5()

    def __repr__(self):
        """Print the object representation string."""
        msg = "<Exclusions Object:  excl_fpath={}>".format(self.excl_fpath)
        return msg

    def add_layer(self, dname, file, description=None, overwrite=False):
        """Add a raster file and its description to the HDF5 exclusion file."""
        # Open raster object
        try:
            raster = self.rio.open(file)
        except Exception:
            raise self.RasterioIOError("file " + file + " does not exist")

        # Get profile information
        profile = raster.profile
        profile["crs"] = profile["crs"].to_proj4()
        dtype = profile["dtype"]
        profile = dict(profile)

        # We need a 6 element geotransform, sometimes we recieve three extra
        profile["transform"] = profile["transform"][:6]

        # Add coordinates and else check that the new file matches everything
        self._set_coords(profile)
        self._check_dims(raster, profile, dname)

        # Add everything to target exclusion HDF
        array = raster.read()
        with self.h5py.File(self.excl_fpath, "r+") as hdf:
            keys = list(hdf.keys())
            if dname in keys:
                if overwrite:
                    del hdf[dname]
                    keys.remove(dname)

            if dname not in keys:
                hdf.create_dataset(name=dname, data=array, dtype=dtype,
                                   chunks=(1, 128, 128))
                hdf[dname].attrs["file"] = os.path.abspath(file)
                hdf[dname].attrs["profile"] = json.dumps(profile)
                if description:
                    hdf[dname].attrs["description"] = description
                if dname in self.string_values:
                    string_value = json.dumps(self.string_values[dname])
                    hdf[dname].attrs["string_values"] = string_value

    def add_layers(self, file_dict, desc_dict=None, overwrite=False):
        """Add multiple raster files and their descriptions."""
        from tqdm import tqdm

        # Make copies of these dictionaries?
        file_dict = file_dict.copy()
        if desc_dict:
            desc_dict = desc_dict.copy()

        # If descriptions are provided make sure they match the files
        if desc_dict:
            try:
                dninf = [k for k in desc_dict if k not in file_dict]
                fnind = [k for k in file_dict if k not in desc_dict]
                assert not dninf
                assert not fnind
            except Exception:
                mismatches = self.np.unique(dninf + fnind)
                msg = ("File and description keys do not match. "
                       "Problematic keys: " + ", ".join(mismatches))
                raise AssertionError(msg)
        else:
            desc_dict = {key: None for key in file_dict.keys()}

        # Let's remove existing keys here
        if not overwrite:
            with self.h5py.File(self.excl_fpath, "r") as h5:
                keys = list(h5.keys())
                for key in keys:
                    if key in file_dict:
                        del file_dict[key]
                        del desc_dict[key]

        # Should we parallelize this?
        for dname, file in tqdm(file_dict.items(), total=len(file_dict)):
            description = desc_dict[dname]
            self.add_layer(dname, file, description, overwrite=overwrite)

    def techmap(self, res_fpath, dname, max_workers=None, map_chunk=2560,
                distance_upper_bound=None, save_flag=True):
        """Build a mapping grid between exclusion resource data.

        Parameters
        ----------
        res_fpath : str
            Filepath to HDF5 resource file.
        dname : str
            Dataset name in excl_fpath to save mapping results to.
        max_workers : int, optional
            Number of cores to run mapping on. None uses all available cpus.
            The default is None.
        distance_upper_bound : float, optional
            Upper boundary distance for KNN lookup between exclusion points and
            resource points. None will calculate a good distance based on the
            resource meta data coordinates. 0.03 is a good value for a 4km
            resource grid and finer. The default is None.
        map_chunk : TYPE, optional
          Calculation chunk used for the tech mapping calc. The default is
            2560.
        save_flag : boolean, optional
            Save the techmap in the excl_fpath. The default is True.
        """
        from reV.supply_curve.tech_mapping import TechMapping

        # If saving, does it return an object?
        arrays = TechMapping.run(self.excl_fpath, res_fpath, dname,
                                 max_workers=None, sc_resolution=2560)
        return arrays

    def _preflight(self):
        """More initializing steps."""
        if self.string_values:
            if not isinstance(self.string_values, dict):
                with open(self.string_values, "r") as file:
                    self.string_values = json.load(file)

    def _check_dims(self, raster, profile, dname):
        # Check new layers against the first added raster
        old = self.profile
        new = profile

        # Check the CRS
        if not crs_match(old["crs"], new["crs"]):
            raise AssertionError("CRS for " + dname + " does not match"
                                 " exisitng CRS.")

        # Check the transform
        try:
            # Standardize these
            old_trans = old["transform"][:6]
            new_trans = new["transform"][:6]
            assert old_trans == new_trans
        except Exception:
            raise AssertionError("Geotransform for " + dname + " does "
                                 "not match geotransform.")

        # Check the dimesions
        try:
            assert old["width"] == new["width"]
            assert old["height"] == new["height"]
        except Exception:
            raise AssertionError("Width and/or height for " + dname +
                                 " does not match exisitng " +
                                 "dimensions.")

    def _convert_coords(self, xs, ys):
        # Convert projected coordinates into WGS84
        mx, my = self.np.meshgrid(xs, ys)
        transformer = self.Transformer.from_crs(self.profile["crs"],
                                                "epsg:4326", always_xy=True)
        lons, lats = transformer.transform(mx, my)
        return lons, lats

    def _get_coords(self):
        # Get x and y coordinates (One day we'll have one transform order!)
        geom = self.profile["transform"]  # Ensure its in the right order
        xres = geom[0]
        ulx = geom[2]
        yres = geom[4]
        uly = geom[5]

        # Not doing rotations here
        xs = [ulx + col * xres for col in range(self.profile["width"])]
        ys = [uly + row * yres for row in range(self.profile["height"])]

        # Let's not use float 64
        xs = self.np.array(xs).astype("float32")
        ys = self.np.array(ys).astype("float32")

        return xs, ys

    def _initialize_h5(self):
        import datetime as dt

        # Create an empty hdf file if one doesn't exist
        date = format(dt.datetime.today(), "%Y-%m-%d %H:%M")
        self.excl_fpath = os.path.expanduser(self.excl_fpath)
        self.excl_fpath = os.path.abspath(self.excl_fpath)
        if not os.path.exists(self.excl_fpath):
            os.makedirs(os.path.dirname(self.excl_fpath), exist_ok=True)
            with self.h5py.File(self.excl_fpath, "w") as ds:
                ds.attrs["creation_date"] = date

    def _set_coords(self, profile):
        # Add the lat and lon meshgrids if they aren't already present
        with self.h5py.File(self.excl_fpath, "r+") as hdf:
            keys = list(hdf.keys())
            attrs = hdf.attrs.keys()

            # Set profile if needed
            if "profile" not in attrs:
                hdf.attrs["profile"] = json.dumps(profile)
            self.profile = profile

            # Set coordinates if needed
            if "latitude" not in keys or "longitude" not in keys:
                # Get the original crs coordinates
                xs, ys = self._get_coords()

                # Convert to geographic coordinates
                lons, lats = self._convert_coords(xs, ys)

                # Create grid and upload
                hdf.create_dataset(name="longitude", data=lons)
                hdf.create_dataset(name="latitude", data=lats)

class Reformatter:
    """Reformat any file into a reV-shaped raster."""
    import os
    import subprocess as sp
    import numpy as np
    import geopandas as gpd
    import rasterio as rio
    from tqdm import tqdm
    from pathos import multiprocessing as mp
    from rasterio import features
    import h5py
    import datetime

    def __init__(self, template, vectorfile_fields):
        """Initialize Reformatter object.

        Parameters
        ----------
        template : str
            Path to a raster to use as a template.
        """
        self.template = template
        self.vectorfile_fields = vectorfile_fields
        self.string_lookup = {}

    def __repr__(self):
        """Return object representation string."""
        return f"<Reformatter: template={self.template}>"

    def reformat_raster(self, file, dst):  # <------------------------------------------ Use rio python bindings here, the compression isnt' working
        """Resample and re-project a raster."""
        self.sp.call(["rio", "warp", file, dst,
                      "--like", self.template,
                      "--co", "compress=lzw",  # <----------------------------- Might not be working?
                      "--co", "blockysize=128",
                      "--co", "blockxsize=128",
                      "--co", "tiled=yes"]) # <----------- do we need this function? 

    def reformat_batch_vectorfiles(self,dst_folder):
        """Batch process the shapefiles indicated by the shapefile_fields"""
        # get the number of cpu
        n_cpu = self.os.cpu_count()
        
        # create output raster path
        vectorfile_fields = self.vectorfile_fields

        # sequential processing
        for key, value_list in vectorfile_fields.items():
            for value in value_list:
                # create dst path
                dst_name  = self.os.path.basename(key).split(".")[0] + "_" + value + ".tif"
                dst = self.os.path.join(dst_folder,dst_name)
                
                # single cpu process
                raseter_path = self.reformat_vectorfile(file = key,dst = dst,field = value)
                # print(dst)
                # h5_path = dst.replace(".tif",".h5")
                # data_setname = self.os.path.basename(key).split(".")[0] + "_" + value
                # self.reformat_to_h5(raseter_path,h5_path,data_setname)



        # parallel process the vector files <-------------------------haven't implemented
        # dsts=[]
        # with self.mp.Pool(n_cpu) as pool:
        #     for t in self.tqdm(pool.map(self.reformat_vectorfile,vector_files,dst,field_names),
        #                     total=n_cpu):
        #         dsts.append(t)
        # return dsts
    
    
    def reformat_vectorfile(self, file, dst, field=None, buffer=None, overwrite=False):
        """Preprocess, re-project, and rasterize a vector."""
        # Read and process file
        gdf = self._process_vectorfile(file, field, buffer)
        meta = self.meta

        # Skip if overwrite
        if not overwrite and self.os.path.exists(dst):
            return dst
        else:
            # Rasterize
            elements = gdf[["raster_value", "geometry"]].values
            shapes = [(g, r) for r, g in elements]

            out_shape = [meta["height"], meta["width"]]
            transform = meta["transform"]
            with self.rio.Env():
                array = self.features.rasterize(shapes, out_shape,
                                                transform=transform)
            
            dtype = str(array.dtype)
            if "int" in dtype:
                nodata = self.np.iinfo(dtype).max
            else:
                nodata = self.np.finfo(dtype).max
            meta["dtype"] = dtype
            meta["nodata"] = nodata
            
            # Write to a raster
            with self.rio.Env():
                with self.rio.open(dst, "w", **meta) as rio_dst:
                    rio_dst.write(array, 1)
            
        # reformat to h5
        h5_file = dst.replace(".tif",".h5")
        self.reformat_to_h5(file,dst,h5_file)


    
    def reformat_to_h5(self,vector_file,raster_path,h5_path,dataset_name=None):
        """ reformat a raster to a h5 file"""
        # open raster
        with self.rio.open(raster_path) as raster:
            raster_data = raster.read(1)
            print(raster_data.dtype)
            # meta = str(raster.meta.copy())

        # if dataset_name is none, use the input raster name 
        if dataset_name is None:
            dataset_name = self.os.path.basename(h5_path).split(".")[0]
        
        # create h5 and write dataset
        with self.h5py.File(h5_path,"w") as h5_f:
            data_set = h5_f.create_dataset(dataset_name, data=raster_data)
            # print(self.string_lookup)
            try:
                data_set.attrs['value lookup'] =self.string_lookup[self.os.path.basename(vector_file).split["."][0]]
            except:
                print("no lookup table")
            # data_set.attrs['create date'] = self.datetime.date.today()
            data_set.attrs['projection'] = self.meta
        h5_f.close()
    

    def guided(self, dirname="."):
        """Guide the processing of shapefiles with prompts and user inputs."""

    @property
    def meta(self):
        """Return the meta information from the template file."""
        with self.rio.open(self.template) as raster:
            meta = raster.meta
        return meta

    def _map_strings(self, file, gdf, field):
        """Map string values to integers and save a lookup dictionary."""
        # Assing integers to unique string values
        strings = gdf[field].unique()
        string_map = {i + 1: v for i, v in enumerate(strings)}
        value_map = {v: k for k, v in string_map.items()}

        # Replace strings with integers
        gdf["raster_value"] = gdf[field].map(value_map)

        # Update the string lookup dictionary <------- a file may have multiple fields in string type 
        file_name = self.os.path.basename(file).split(".")[0]  # <------------------ Adjust incase people use .'s in there file names
        field_map  = {field:string_map}
        
        # created a nested string_loopup
        if file_name not in self.string_lookup:
            self.string_lookup[file_name] = field_map
        else:
            self.string_lookup[file_name].update(field_map)
        
        print(self.string_lookup)

        return gdf

    def _process_vectorfile(self, file, field=None, buffer=None):
        """Process a single file. It includes following steps:
        1. check if the a field exists in a shapefile;
            to make sure the field name and its associated shapefile's path is correct; 
            otherwise raise an error
        2. check if the shapefile has a the same projetion as the template raster
            if not, re-project
        3. if buffer isn't non, then buffer the shapefile 
         """

        # Read in file and check the path
        if not self.os.path.exists(file):
            raise Exception("This shapefile path doesn't exist:{}".format(file))
        else:
            gdf = self.gpd.read_file(file)

        # Check the projection;
        if str(gdf.crs).upper() != str(self.meta["crs"]).upper():
            gdf = gdf.to_crs(self.meta["crs"])
        

        # Check if the field value in the shapefile and Assign raster value
        if field:
            if field not in gdf.columns:
                raise Exception("Field '{}' not in '{}'".format(field,file)) 
            gdf["raster_value"] = gdf[field]
        else:
            gdf["raster_value"] = 1

        # Account for string values
        if isinstance(gdf["raster_value"].iloc[0], str):
            gdf = self._map_strings(file, gdf, field)

        # Reduce to two fields
        gdf = gdf[["raster_value", "geometry"]]

        # Buffering
        if buffer:
            gdf["geometry"] = gdf["geometry"].buffer(buffer)
        return gdf
    

        


if __name__ == "__main__":
    
    # shapefile_fields = {
    #     "/Users/jgu3/GDS/01-Project/02-Luma/01-Rawdata/puerto_rico_wind_hr_2019.gpkg": "boundary_layer_height",
    #     "/Users/jgu3/GDS/01-Project/02-Luma/01-Rawdata/offshore/PR_Tropical_Cyclone_Wind_Exposure.geojson": "leaseBlock",
    # }
    # TEMPLATE = r"/Users/jgu3/GDS/01-Project/02-Luma/01-Rawdata/template/pr_template_32161.tif"
    # self = Reformatter(TEMPLATE, vectorfile_fields=shapefile_fields)
    
    # # single shapefile test
    # # 1. numerical example
    # shape_file = "/Users/jgu3/GDS/01-Project/02-Luma/01-Rawdata/puerto_rico_wind_hr_2019.gpkg"
    # test_field = "boundary_layer_height"
    # dst = "/Users/jgu3/GDS/01-Project/02-Luma/03-exclusions/outraster/puerto_rico_wind_hr_2019_boundary_layer_height.tif"
    # self.reformat_vectorfile(shape_file,dst=dst,field = test_field,buffer=100)
    
    # # 2. string example
    # shape_file = "/Users/jgu3/GDS/01-Project/02-Luma/01-Rawdata/offshore/PR_Tropical_Cyclone_Wind_Exposure.geojson"
    # dst = "/Users/jgu3/GDS/01-Project/02-Luma/03-exclusions/outraster/PR_Tropical_Cyclone_Wind_Exposure_wind_exposure.tif"
    # self.reformat_vectorfile(shape_file,dst,field = "leaseBlock")

    # multiple shapefiles with multiple fields
    # vectorfile_fields = {
    #     "/Users/jgu3/GDS/01-Project/02-Luma/01-Rawdata/offshore/puerto_rico_wind_hr_2019.gpkg": "boundary_layer_height",
    #     "/Users/jgu3/GDS/01-Project/02-Luma/01-Rawdata/offshore/PR_Tropical_Cyclone_Wind_Exposure.geojson": "leaseBlock",
    #     "/Users/jgu3/GDS/01-Project/02-Luma/01-Rawdata/offshore/PR_Tropical_Cyclone_Wind_Exposure.geojson": "protractionNumber"
    # }
    import pandas as pd
    from collections import defaultdict

    # read csv
    input_table = pd.read_csv("/Users/jgu3/GDS/01-Project/02-Luma/01-Rawdata/offshore/rev_input.csv")

    # iterate the table, create a vector_file - field dictionary
    vectorfile_fields  = defaultdict(list)
    for index, row in input_table.iterrows():
        vectorfile_fields[row["file_path"]].append(row["field"])
    # print(vectorfile_fields)
    

    # reformat
    TEMPLATE = "/Users/jgu3/GDS/01-Project/02-Luma/01-Rawdata/template/pr_template_32161.tif"
    reformat = Reformatter(TEMPLATE, vectorfile_fields=vectorfile_fields)
    out_folder = "/Users/jgu3/GDS/01-Project/02-Luma/03-exclusions/batch_output"
    reformat.reformat_batch_vectorfiles(out_folder)