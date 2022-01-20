"""
A set of functions for performing common spatial transformations using GDAL
bindings, geopandas, rasterio, and shapely.

Things to do:
    - Incorporate data types into these functions.
    - Continuously incorporate other GDAL functionality, too.
    - Some of these would be better placed in the spatial or utilities modules.
    - Create a file checking function and use it when writing new files. This
      could detect the file type, try to open it with the appropriate function,
      and raise an exception and delete it if it fails.
    - use **kwargs to include all available options. The check will still flag
      non-extant options.
    - add in creation options, as these are separate.
    - ask around about exceptions. For cases where no options are provided I
      want to simply return the function and print the options. If I don't
      raise an exception here, will that cause overly grave problems?
"""
import os
import shutil
import subprocess as sp
import sys
import zipfile

import geopandas as gpd
import numpy as np
import rasterio
import requests

from multiprocessing import Pool
from osgeo import gdal, ogr, osr
from shapely.geometry import Point
from tqdm import tqdm

gdal.UseExceptions()


# CONSTANTS
GDAL_TYPES = {
    "GDT_Byte": "Eight bit unsigned integer",
    "GDT_CFloat32": "Complex Float32",
    "GDT_CFloat64": "Complex Float64",
    "GDT_CInt16": "Complex Int16",
    "GDT_CInt32": "Complex Int32",
    "GDT_Float32": "Thirty two bit floating point",
    "GDT_Float64": "Sixty four bit floating point",
    "GDT_Int16": "Sixteen bit signed integer",
    "GDT_Int32": "Thirty two bit signed integer",
    "GDT_UInt16": "Sixteen bit unsigned integer",
    "GDT_UInt32": "Thirty two bit unsigned integer",
    "GDT_Unknown": "Unknown or unspecified type"
}

GDAL_TYPEMAP = {
    "byte": {
        "type": gdal.GDT_Byte,
        "min": np.iinfo("uint8").min,
        "max": np.iinfo("uint8").max
    },
    "cfloat32": {
        "type": gdal.GDT_CFloat32,
        "min": np.finfo("float32").min,
        "max": np.finfo("float32").max
    },
    "cfloat64": {
        "type": gdal.GDT_CFloat64,
        "min": np.finfo("float64").min,
        "max": np.finfo("float64").max
    },
    "cint16": {
        "type": gdal.GDT_CInt16,
        "min": np.iinfo("int16").min,
        "max": np.iinfo("int16").max
    },
    "cint32": {
        "type": gdal.GDT_CInt32,
        "min": np.iinfo("int32").min,
        "max": np.iinfo("int32").max
    },
    "float32": {
        "type": gdal.GDT_Float32,
        "min": np.finfo("float32").min,
        "max": np.finfo("float32").max
    },
    "float64": {
        "type": gdal.GDT_Float64,
        "min": np.finfo("float64").min,
        "max": np.finfo("float64").max
    },
    "int16": {
        "type": gdal.GDT_Int16,
        "min": np.iinfo("int16").min,
        "max": np.iinfo("int16").max
    },
    "int32": {
        "type": gdal.GDT_Int32,
        "min": np.iinfo("int32").min,
        "max": np.iinfo("int32").max
    },
    "uint16": {
        "type": gdal.GDT_UInt16,
        "min": np.iinfo("uint16").min,
        "max": np.iinfo("uint16").max
    },
    "uint32": {
        "type": gdal.GDT_UInt32,
        "min": np.iinfo("uint32").min,
        "max": np.iinfo("uint32").max
    },
    "unknown": {
        "type": gdal.GDT_Unknown,
        "min": np.nan,
        "max": np.nan
    }
}


GDAL_MAPTYPES = {
    gdal.GDT_Byte: {
        "type": "byte",
        "min": np.iinfo("uint8").min,
        "max": np.iinfo("uint8").max
    },
    gdal.GDT_CFloat32: {
        "type": "cfloat32",
        "min": np.finfo("float32").min,
        "max": np.finfo("float32").max
    },
    gdal.GDT_CFloat64: {
        "type": "cfloat64",
        "min": np.finfo("float64").min,
        "max": np.finfo("float64").max
    },
    gdal.GDT_CInt16: {
        "type": "cint16",
        "min": np.iinfo("int16").min,
        "max": np.iinfo("int16").max
    },
    gdal.GDT_CInt32: {
        "type": "cint32",
        "min": np.iinfo("int32").min,
        "max": np.iinfo("int32").max
    },
    gdal.GDT_Float32: {
        "type": "float32",
        "min": np.finfo("float32").min,
        "max": np.finfo("float32").max
    },
    gdal.GDT_Float64: {
        "type": "float64",
        "min": np.finfo("float64").min,
        "max": np.finfo("float64").max
    },
    gdal.GDT_Int16: {
        "type": "int16",
        "min": np.iinfo("int16").min,
        "max": np.iinfo("int16").max
    },
    gdal.GDT_Int32: {
        "type": "int32",
        "min": np.iinfo("int32").min,
        "max": np.iinfo("int32").max
    },
    gdal.GDT_UInt16: {
        "type": "uint16",
        "min": np.iinfo("uint16").min,
        "max": np.iinfo("uint16").max
    },
    gdal.GDT_UInt32: {
        "type": "uint32",
        "min": np.iinfo("uint32").min,
        "max": np.iinfo("uint32").max
    },
    gdal.GDT_Unknown: {
        "type": "unknown",
        "min": np.nan,
        "max": np.nan
    }
}
DRIVERS = {
    ".shp": "ESRI Shapefile",
    ".gpkg": "GPKG"
}


# FUNCTIONS
def gdal_options(module="translate", **kwargs):
    """Capture any availabe option for gdal functions. Print available options
    if one was mispelled. Alternately, run with no **kwargs for a list of
    available options and descriptions.

    Examples:
        gdal_options("warp")
        ops = gdal_options("warp", dstSRS="epsg:4326", xRes=.25, yRes=.25)
    """
    # Standardize case
    module = module.lower().replace("gdal", "").replace("_", "")

    # All available options
    options = [m for m in gdal.__dict__ if "Options" in m and "_" not in m and "GDAL" not in m]

    # Get the module option method associated with module
    modules = []
    for o in options:
        o = o.replace("GDAL", "").replace("Options", "").replace("_", "")
        o = o.lower()
        modules.append(o)

    # Create options dictionary
    option_dict = dict(zip(modules, options))

    # Get the requested option method
    try:
        option = option_dict[module]
        method = getattr(gdal, option)
    except KeyError:
        print("GDAL options for " + module + " are not available.")
        docs = "\n   ".join(modules)
        print("Available methods with options:\n   " + docs)
        return

    # Get the docs for that method
    docs = "\n".join(method.__doc__.split("\n")[1:])

    # Return the appropriate options object
    try:
        assert kwargs
        ops = method(**kwargs)
        return ops
    except AssertionError:
        print(docs)
        return
    except TypeError as terror:
        te = terror.args[0]
        missing = te.split()[-1]
        print("The " + missing + " option is not available or formatted "
              "incorrectly.")
        print(docs)
        return


def dlzip(url, path):
    """Download, unzip, and remove zip file from url."""
    if not os.path.exists(path):
        r = requests.get(url)
        with open(path, 'wb') as file:
            file.write(r.content)
        with zipfile.ZipFile(path, 'r') as zip_ref:
            save_dir = os.path.splitext(path.replace(".zip", ""))[0]
            zip_ref.extractall(save_dir)
        os.remove(path)


def gdal_progress(complete, message, unknown):
    """A progress callback that recreates the gdal printouts."""
    # We don't need the message or unknown objects
    del message, unknown

    # Complete is a number between 0 and 1
    percent = int(complete * 100)

    # Between numeric printouts we need three dots
    dots = [[str(i) + d for d in ["2", "5", "8"]] for i in range(10)]
    dots = [int(l) for sl in dots for l in sl]

    # If divisible by ten, print the number
    if percent % 10 == 0 and percent != 0:
        print("{}".format(percent), end="")

    # If one of three numbers between multiples of 10, print a dot
    elif percent in dots:
        print(".", end="")

    return 1


def rasterize(src, dst, attribute, t_srs=None, transform=None, height=None,
              width=None, template_path=None, navalue=None, all_touch=False,
              dtype=None, overwrite=False):
    """Rasterize a shapefile stored on disk and write outputs to a file.

    Parameters
    ----------
    src : str
        File path for the source file to rasterize.
    dst : str
        Destination path for the output raster.
    attribute : str
        Attribute name being rasterized.
    t_srs : int
        EPSG Code associated with target coordinate reference system.
    transform : list | tuple | array
        Geometric affine transformation:
            (x-min, x-resolution, x-rotation, y-max, y-rotation, y-resoltution)
    height : int
        Number of y-axis grid cells.
    width : int
        Number of x-axis grid cells.
    template_path : str
        The path to a raster with target geometries.
    na : int | float
        The value to assign to non-value grid cells. (defaults to -99999)
    all_touch : boolean
        Wether or not to associate vector values with all intersecting grid
        cells. (defaults to False)
    dtype : str | gdal object
        GDAL data type. Can be a string or a gdal type object (e.g.
        gdal.GDT_Float32, "GDT_Float32", "float32"). Available GDAL data types
        and descriptions can be found in the GDAL_TYPES dictionary.
    overwrite : boolean

    Returns
    -------
    None

    # Things to do:
        1) Catch exceptions
        2) Progress callback
        3) Use more than just EPSG (doesn't always work, also accept proj4)
    """
    # Overwrite existing file
    if os.path.exists(dst):
        if overwrite:
            if os.path.isfile(dst):
                os.remove(dst)
            else:
                shutil.rmtree(dst)
        else:
            print(dst + " exists, use overwrite=True to replace this file.")
            return

    # Open shapefile, retrieve the layer
    src_data = ogr.Open(src)
    layer = src_data.GetLayer()

    # Create a spatial reference object
    refs = osr.SpatialReference()

    # If a template is provided
    if template_path:
        template = gdal.Open(template_path)
        transform = template.GetGeoTransform()
        width = template.RasterXSize
        height = template.RasterYSize
        t_srs = template.GetProjection()
        refs.ImportFromWkt(t_srs)
    else:
        try:
            refs.ImportFromEPSG(t_srs)
        except TypeError:
            try:
                refs.ImportFromProj4(t_srs)
            except TypeError:
                try:
                    refs.ImportFromWkt(t_srs)
                except TypeError:
                    print("Could not interpret the coordinate reference "
                          "system using the EPSG, proj4, or WKT formats.")
                    raise

    # Use transform to derive coordinates and dimensions
    xmin, xres, xrot, ymax, yrot, yres = transform
    xs = [xmin + xres * i for i in range(width)]
    ys = [ymax + yres * i for i in range(height)]
    nx = len(xs)
    ny = len(ys)

    if isinstance(dtype, str):
        dtype = dtype.lower().replace("gdt_", "")
        try:
            idtype = GDAL_TYPEMAP[dtype]["type"]
        except KeyError:
            print("\n'" + dtype + "' is not an available data type. "
                  "Choose a value from this list:")
            print(str(list(GDAL_TYPEMAP.keys())))     

    # Set the nodata value to the maximum datatype value
    if not navalue:
        navalue = GDAL_TYPEMAP[dtype]["max"]

    # Create the target raster layer
    driver = gdal.GetDriverByName("GTiff")
    trgt = driver.Create(dst, nx, ny, 1, idtype, options=["COMPRESS=LZW"])
    trgt.SetGeoTransform((xmin, xres, xrot, ymax, yrot, yres))
    trgt.SetProjection(refs.ExportToWkt())

    # Set no value
    band = trgt.GetRasterBand(1)
    band.SetNoDataValue(navalue)

    # Set options
    if all_touch is True:
        ops = ["-at", "ATTRIBUTE=" + attribute]
    else:
        ops = ["ATTRIBUTE=" + attribute]

    # Finally rasterize
    gdal.RasterizeLayer(trgt, [1], layer, options=ops, callback=gdal_progress)

    # Close target an source rasters
    del trgt
    del src_data


def read_raster(rasterpath, band=1, navalue=-9999):
    """Read raster file return array, geotransform, and crs.

    Parameters
    ----------
    rasterpath : str
        Path to a raster file.
    band : int
        The band number desired.
    navalue : int | float
        The number used for non-values in the raster data set

    Returns
    -------
        tuple:
             raster values : numpy.ndarray
             affine transformation : tuple
                 (top left x coordinate, x resolution, row rotation,
                  top left y coordinate, column rotation, y resolution)),
            coordinate reference system : str
                 Well-Known Text format
    """
    # Open raster file and read in parts necessary for rewriting
    raster = gdal.Open(rasterpath)
    geometry = raster.GetGeoTransform()
    arrayref = raster.GetProjection()
    array = np.array(raster.GetRasterBand(band).ReadAsArray())
    raster = None

    # This helped for some old use-case, but might not be necessary
    array = array.astype(float)
    if np.nanmin(array) < navalue:
        navalue = np.nanmin(array)
    array[array == navalue] = np.nan

    return(array, geometry, arrayref)


def reproject_polygon(src, dst, t_srs):
    """Reproject a shapefile of polygons and write results to disk.

    Parameters
    ----------
    src : str
        Path to source shapefile.
    dst : str
        Path to target file.
    tproj (int | str):
        Target coordinate projection system as an epsg code or proj4 string.
        Sometimes EPSG codes aren't available to GDAL installations, but
        they're easier to use when they are so this will try both.

    Note
    ----
    This only handles ESRI Shapefiles at the moment, but can be written to
    handle any available driver.
    """
    # Create target directory
    save_path = os.path.dirname(dst)
    os.makedirs(save_path, exist_ok=True)

    # Create Shapefile driver
    name = DRIVERS[os.path.splitext(src)[-1]]
    driver = ogr.GetDriverByName(name)

    # Source reference information
    src_file = driver.Open(src)
    src_layer = src_file.GetLayer()
    src_srs = src_layer.GetSpatialRef()
    src_defn = src_layer.GetLayerDefn()

    # Target reference information
    trgt_srs = osr.SpatialReference()
    try:
        trgt_srs.ImportFromEPSG(t_srs)
    except TypeError:
        try:
            trgt_srs.ImportFromProj4(t_srs)
        except TypeError:
            trgt_srs.ImportFromWkt(t_srs)

    # The transformation equation
    transform = osr.CoordinateTransformation(src_srs, trgt_srs)

    # Target file and layer
    if os.path.exists(dst):
        driver.DeleteDataSource(dst)
    trgt_file = driver.CreateDataSource(dst)
    trgt_layer = trgt_file.CreateLayer('', trgt_srs, ogr.wkbMultiPolygon)

    # Add Fields
    for i in range(0, src_defn.GetFieldCount()):
        defn = src_defn.GetFieldDefn(i)
        trgt_layer.CreateField(defn)

    # Get the target layer definition
    trgt_defn = trgt_layer.GetLayerDefn()

    # You have to reproject each feature
    src_feature = src_layer.GetNextFeature()
    while src_feature:
        # Get geometry
        geom = src_feature.GetGeometryRef()

        # Reproject geometry
        geom.Transform(transform)

        # Create target feature
        trgt_feature = ogr.Feature(trgt_defn)
        trgt_feature.SetGeometry(geom)
        for i in range(0, trgt_defn.GetFieldCount()):
            trgt_feature.SetField(trgt_defn.GetFieldDefn(i).GetNameRef(),
                                  src_feature.GetField(i))

        # Add feature to target file
        trgt_layer.CreateFeature(trgt_feature)

        # Close current feature
        trgt_feature = None

        # Get the next feature
        src_feature = src_layer.GetNextFeature()

    # Close both shapefiles
    src_file = None
    trgt_file = None


def reproject_point(src, dst, tproj):
    """Reproject a shapefile of points and write results to disk. Recreates
    this GDAL command:

        ogr2ogr -s_srs <source_projection> -t_srs <target_projection> dst src

    Parameters
    ----------
    src : str
        Path to source shapefile.
    dst : str
        Path to target file.
    tproj (int | str):
        Target coordinate projection system as an epsg code or proj4 string.
        Sometimes EPSG codes aren't available to GDAL installations, but
        they're easier to use when they are so this will try both.

    Note
    ----
    This only handles ESRI Shapefiles at the moment, but can be written to
    handle any available driver.
    """
    # Create Shapefile driver
    name = DRIVERS[os.path.splitext(src)[-1]]
    driver = ogr.GetDriverByName(name)

    # Source reference information
    src_file = driver.Open(src)
    src_layer = src_file.GetLayer()
    src_srs = src_layer.GetSpatialRef()
    src_defn = src_layer.GetLayerDefn()

    # Target reference information
    trgt_srs = osr.SpatialReference()
    try:
        trgt_srs.ImportFromEPSG(tproj)
    except Exception:
        trgt_srs.ImportFromProj4(tproj)

    # The transformation equation
    transform = osr.CoordinateTransformation(src_srs, trgt_srs)

    # Target file and layer
    if os.path.exists(dst):
        driver.DeleteDataSource(dst)
    trgt_file = driver.CreateDataSource(dst)
    trgt_layer = trgt_file.CreateLayer('', trgt_srs, ogr.wkbPoint)

    # Add Fields
    for i in range(0, src_defn.GetFieldCount()):
        defn = src_defn.GetFieldDefn(i)
        trgt_layer.CreateField(defn)

    # Get the target layer definition
    trgt_defn = trgt_layer.GetLayerDefn()

    # You have to reproject each feature
    src_feature = src_layer.GetNextFeature()
    while src_feature:
        # Get geometry
        geom = src_feature.GetGeometryRef()

        # Reproject geometry
        geom.Transform(transform)

        # Create target feature
        trgt_feature = ogr.Feature(trgt_defn)
        trgt_feature.SetGeometry(geom)
        for i in range(0, trgt_defn.GetFieldCount()):
            trgt_feature.SetField(trgt_defn.GetFieldDefn(i).GetNameRef(),
                                  src_feature.GetField(i))

        # Add feature to target file
        trgt_layer.CreateFeature(trgt_feature)

        # Close current feature
        trgt_feature = None

        # Get the next feature
        src_feature = src_layer.GetNextFeature()

    # Close both shapefiles
    src_file = None
    trgt_file = None


def shape_dtype(src, attribute):
    """Get the data type name of a field in a shapefile."""
    driver_name = DRIVERS[os.path.splitext(src)[-1]]
    driver = ogr.GetDriverByName(driver_name)
    shp = driver.Open(src)

    # Get the layer and its definition
    layer = shp.GetLayer()
    layer_def = layer.GetLayerDefn()

    for i in range(layer_def.GetFieldCount()):
        field_name = layer_def.GetFieldDefn(i).GetName()
        if field_name == attribute:
            type_code = layer_def.GetFieldDefn(i).GetType()
            type_name = layer_def.GetFieldDefn(i).GetFieldTypeName(type_code)

            return type_name.lower()


def split_extent(raster_file, n=100):
    """Split a raster files extent into n extent pieces."""
    # Get raster geometry
    rstr = rasterio.open(raster_file)
    geom = rstr.get_transform()
    ny = rstr.height
    nx = rstr.width
    xs = [geom[0] + geom[1] * i for i in range(nx)]
    ys = [geom[3] + geom[-1] * i for i in range(ny)]

    # Get number of chunks along each axis
    nc = np.sqrt(n)

    # Split coordinates into 10 pieces along both axes...
    xchunks = np.array(np.array_split(xs, nc))
    ychunks = np.array(np.array_split(ys, nc))

    # Get min/max of each coordinate chunk...
    sides = lambda x: [min(x), max(x)]
    xmap = map(sides, xchunks)
    ymap = map(sides, ychunks)
    xext = np.array([v for v in xmap])
    yext = np.array([v for v in ymap])

    # Combine these in this order [xmin, ymin, xmax, ymax]....
    extents = []
    for xex in xext:
        for yex in yext:
            extents.append([xex[0], yex[0], xex[1], yex[1]])

    return extents


def tile_raster(raster_file, out_folder, ntiles, ncpu):
    """ Take a raster and write n tiles from it.

    Parameters
    ----------
    raster_file : str
        Path to a GeoTiff
    out_folder : str
        Path to a folder in which to store tiles. Will create if not present.
    ntiles : int
        Number of tiles to write.
    ncpu : int
        Number of cpus to use for processing.

    Returns
    -------
    None.
    """
    # Create the output folder
    if not out_folder:
        base_name = os.path.splitext(raster_file)[0]
        out_folder = "_".join([base_name, "tiles"])
    os.makedirs(out_folder, exist_ok=True)

    # Get all of the extents needed to make n tiles
    extents = split_extent(raster_file, n=ntiles)

    # Wrap arguments into one object
    raster_files = np.repeat(raster_file, len(extents))
    chunknumbers = [i for i in range(len(extents))]
    out_folders = np.repeat(out_folder, len(extents))
    args = list(zip(extents, raster_files, chunknumbers, out_folders))

    # Run each
    with Pool(ncpu) as pool:
        tfiles = []
        for tfile in tqdm(pool.imap(tile_single, args), total=len(extents),
                          position=0, file=sys.stdout):
            tfiles.append(tfile)

    return tfiles


def tile_single(arg):
    """Use gdal to cut a raster into a smaller pieces.

    Note:
        This is made for tile_raster and is not intuitive as a standalone.
        Add in a check to make sure each output file is good. Moving to
        a class method soon.
    """
    # Separate arguments
    extent = arg[0]
    rfile = arg[1]
    chunk = arg[2]
    outfolder = arg[3]

    # Get everything in order
    extent = [str(e) for e in extent]
    chunk = "{:02d}".format(chunk)
    outbase = os.path.basename(rfile).split(".")[0]
    outfile = os.path.join(outfolder, outbase + "_" + chunk + ".tif")

    # Let's not overwrite - use the warp function from below here instead
    if not os.path.exists(outfile):
        sp.call(["gdalwarp",
                 "-q",
                 "-te", extent[0], extent[1], extent[2], extent[3],
                 rfile,
                 outfile],
                stdout=sp.PIPE, stderr=sp.PIPE)

    return outfile


def to_geo(data_frame, loncol="lon", latcol="lat", epsg=4326):
    """Convert a Pandas DataFrame object to a GeoPandas GeoDataFrame object.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        A pandas data frame with latitude and longitude coordinates.
    loncol : str
        The name of the longitude column.
    latcol : str
        The name of the latitude column.
    epsg : int
        EPSG code associated with the Coordinate Reference System.

    Returns
    -------
    geopandas.geodataframe.GeoDataFrame
        A GeoPandas GeoDataFrame object.
    """

    crs = {"init": "epsg:{}".format(epsg)}
    to_point = lambda x: Point((x[loncol], x[latcol]))
    data_frame["geometry"] = data_frame.apply(to_point, axis=1)
    gdf = gpd.GeoDataFrame(data_frame, geometry="geometry", crs=crs)

    return gdf


def to_raster(array, savepath, crs=None, geometry=None, template=None,
              dtype=gdal.GDT_Float32, compress=None, navalue=-9999):
    """Takes in a numpy array and writes data to a GeoTiff.

    Parameters
    ----------
    array : numpy.ndarray
        Numpy array to write to raster file.
    savepath : str
        Path to the target raster file.
    crs : str
        Coordinate reference system in Well-Known Text format.
    geometry : tuple
        Affine transformation information in this order:
            (top left x coordinate, x resolution, row rotation,
            top left y coordinate, column rotation, y resolution)
    template : str
        Path to a raster file with desired target raster geometry, crs, and na
        value. This will overwrite other arguments provided for these
        parameters.
    dtype : str | gdal object
        GDAL data type. Can be a string or a gdal type object (e.g.
        gdal.GDT_Float32, "GDT_Float32", "float32"). Available GDAL data types
        and descriptions can be found in the GDAL_TYPES dictionary.
    compress : str
        A compression technique. Available options are "DEFLATE", "JPEG",
        "LZW"
    navalue : int | float
        The number used for non-values in the raster data set. Defaults to
        -9999.
    """
    # Retrieve needed raster elements
    xpixels = array.shape[1]
    ypixels = array.shape[0]

    # This helps sometimes
    savepath = savepath.encode('utf-8')

    # Specifying data types shouldn't be so difficult
    if isinstance(dtype, str):
        dtype = dtype.lower().replace("gdt_", "")
        try:
            dtype = GDAL_TYPEMAP[dtype]["type"]
        except KeyError:
            print("\n'" + dtype + "' is not an available data type. "
                  "Choose a value from this list:")
            print(str(list(GDAL_TYPEMAP.keys())))

    # Create file
    driver = gdal.GetDriverByName("GTiff")

    # Get options here - not built out yet
    if compress:
        creation_ops = ["compress=LZW"]
        image = driver.Create(savepath, xpixels, ypixels, 1, dtype,
                              options=creation_ops)
    else:
        image = driver.Create(savepath, xpixels, ypixels, 1, dtype)

    # Use a template file to extract affine transformation, crs, and na value
    if template:
        template_file = gdal.Open(template)
        geometry = template_file.GetGeoTransform()
        crs = template_file.GetProjection()

    # Write raster data and attributes to file
    image.SetGeoTransform(geometry)
    image.SetProjection(crs)
    image.GetRasterBand(1).WriteArray(array)
    image.GetRasterBand(1).SetNoDataValue(navalue)


def translate(src, dst, overwrite=False, compress=None, **kwargs):
    """
    Translate a raster dataset from one format to another.

    Parameters
    ----------
    src : str
        Path to source raster file or containing folder for ESRI Grids.
    dst : str
        Path to target raster file.
    overwrite : boolean
    compress : str
        A compression technique. Available options are "DEFLATE", "JPEG",
        "LZW"
    **kwargs
        Any available key word arguments for gdal_translate. Available options
        and descriptions can be found using gdal_options("translate").

    Returns
    -------
    None.

    Notes
    -----

    The progress bar needs work.

    """
    # Create progress callback - these behave differently by module
    def translate_progress(percent, message, unknown):
        """A progress callback that recreates the gdal printouts."""

        # We don't need the message or unknown objects
        del message, unknown

        # Between numeric printouts we need three dots
        dots = [[str(i) + d for d in ["2", "5", "8"]] for i in range(10)]
        dots = [int(l) for sl in dots for l in sl]

        # If divisible by ten, print the number
        if percent % 10 == 0 and percent != 0:
            print("{}".format(percent), end="")

        # If one of three numbers between multiples of 10, print a dot
        elif percent in dots:
            print(".", end="")

        return 1

    # Expand user paths
    src = os.path.expanduser(src)
    dst = os.path.expanduser(dst)

    # Overwrite existing file
    if os.path.exists(dst):
        if overwrite:
            if os.path.isfile(dst):
                os.remove(dst)
            else:
                shutil.rmtree(dst)
        else:
            print(dst + " exists, use overwrite=True to replace this file.")
            return

    # What is the best way to deal with ESRI grids?
    if not os.path.isfile(src):
        files = os.listdir(src)
        if not "hdr.adf" in files:
            raise FileNotFoundError("Cannot find a translatable file.")

    # Add in key word arguments
    kwargs["callback"] = gdal_progress

    # Compress
    if compress:
        kwargs["creationOptions"] = ["COMPRESS=" + compress]

    # Create an options object
    ops = gdal_options("translate", **kwargs)

    # We need to open the src data set
    src = gdal.Open(src)

    # Call
    print("Processing " + dst + " :")
    ds = gdal.Translate(destName=dst, srcDS=src, options=ops)
    del ds


def warp(src, dst, dtype="Float32", template=None, overwrite=False,
         compress=None, **kwargs):
    """
    Warp a raster to a new geometry.

    Parameters
    ----------
    src : str
        Path to source raster file.
    dst : str
        Path to target raster file.
    dtype : str | gdal object
        GDAL data type. Can be a string or a gdal type object (e.g.
        gdal.GDT_Float32, "GDT_Float32", "float32"). Available GDAL data types
        and descriptions can be found in the GDAL_TYPES dictionary.
    template : str
        Path to a raster file with desired target raster geometry, crs,
        resolution, and extent values. This will overwrite other arguments
        provided for these parameters. Template-derived arguments will
        overwrite **kwargs.
    overwrite : boolean
    compress : str
        A compression technique. Available options are "DEFLATE", "JPEG",
        "LZW"
    **kwargs
        Any available key word arguments for gdalwarp. Available options
        and descriptions can be found using gdal_options("warp").

    Returns
    -------
    None.

    Example:
        warp(src="/Users/twillia2/Box/WETO 1.2/data/rasters/agcounty_product.tif",
             dst="/Users/twillia2/Box/WETO 1.2/data/rasters/test.tif",
             template="/Users/twillia2/Box/WETO 1.2/data/rasters/albers/acre/cost_codes_ac.tif",
             dstSRS="epsg:102008")
    """
    # Create progress callback - these behave differently by module
    def warp_progress(percent, message, unknown):
        """A progress callback that recreates the gdal printouts."""

        # We don't need the message or unknown objects
        del message, unknown

        # Between numeric printouts we need three dots
        dots = [[str(i) + d for d in ["2", "5", "8"]] for i in range(10)]
        dots = [int(l) for sl in dots for l in sl]

        # If divisible by ten, print the number
        if percent % 10 == 0 and percent != 0:
            print("{}".format(percent), end="")

        # If one of three numbers between multiples of 10, print a dot
        elif percent in dots:
            print(".", end="")

        return 1

    # Overwrite existing file
    if os.path.exists(dst):
        if overwrite:
            if os.path.isfile(dst):
                os.remove(dst)
            else:
                shutil.rmtree(dst)
        else:
            print(dst + " exists, use overwrite=True to replace this file.")
            return

    # Specifying data types shouldn't be so difficult
    if isinstance(dtype, str):
        dtype = dtype.lower().replace("gdt_", "")
        try:
            dtype = GDAL_TYPEMAP[dtype]["type"]
        except KeyError:
            print("\n'" + dtype + "' is not an available data type. "
                  "Choose a value from this list:")
            print(str(list(GDAL_TYPEMAP.keys())))

    # Create a spatial reference object
    spatial_ref = osr.SpatialReference()

    # If a template is provided, use its geometry for target figures
    if template:
        temp = gdal.Open(template)
        spatial_ref.ImportFromWkt(temp.GetProjection())
        srs = spatial_ref.ExportToProj4()
        width = temp.RasterXSize  # consider using these warp options
        height = temp.RasterYSize
        transform = temp.GetGeoTransform()
        xmin, xres, xrot, ymax, yrot, yres = transform
        xs = [xmin + xres * i for i in range(width)]
        ys = [ymax + yres * i for i in range(height)]
        xmax = max(xs) + 0.5*xres
        ymax = ymax + 0.5*xres
        ymin = min(ys)
        extent = [xmin, ymin, xmax, ymax]
        kwargs["dstSRS"] = srs
        kwargs["outputBounds"] = extent
        kwargs["xRes"] = transform[1]
        kwargs["yRes"] = transform[-1]  # careful here
        kwargs["outputType"] = dtype
    elif not kwargs:
        print("No warp options provided.")
        gdal_options("warp")
        return

    # Get source srs
    source = gdal.Open(src)
    spatial_ref.ImportFromWkt(source.GetProjection())
    srs = spatial_ref.ExportToProj4()
    kwargs["srcSRS"] = srs

    # Use the progress callback
    kwargs["callback"] = gdal_progress

    # Compress
    if compress:
        kwargs["creationOptions"] = ["COMPRESS=" + compress]

    # Check Options: https://gdal.org/python/osgeo.gdal-module.html#WarpOptions
    ops = gdal_options("warp", **kwargs)

    # Call
    print("Processing " + dst + " :")
    ds = gdal.Warp(dst, src, options=ops)
    del ds

        
class Map_Values:
    """Map a set of keys from an input raster (or rasters) to values in an
    output raster (or rasters) using a dictionary of key-value pairs."""

    def __init__(self, val_dict, err_val=-9999):
        """Initialize Map_Values.

        Parameters
        ----------
        val_dict : dict
            A dictionary of key-value pairs
        errval : int | float
            A value to assign where there are no matching keys in val_dict.
        """
        self.val_dict = val_dict
        self.err_val = err_val

    def map_file(self, src, dst):
        """Take an input raster file, map values from a dictionary to an output
        raster file.

        Parameters
        ----------
        src : str
            Path to the input raster file.
        dst : str
            Path to the output raster file. Directory will be created if it
            does not exist.

        Returns
        -------
        None.
        """
        # Create the output path
        out_folder = os.path.dirname(dst)
        os.makedirs(out_folder, exist_ok=True)

        # Bundle the arguments for map_single (single function)
        arg = [src, dst, self.val_dict]

        # Run it
        self._map_single(arg)

    def map_files(self, src_files, out_folder, ncpu):
        """Take a list of tiled raster files, map values from a dictionary to
        a list of output raster files.

        Parameters
        ----------
        src_files : list-like
            A list of paths to raster files.
        outfolder : str
            A path to a target directory to store output files. Will be
            created if it does not exist.
        ncpu : int
            The number of cpus to use for multiprocessing.

        Returns
        -------
        outfiles : list
            A list of paths to output files.
        """
        # Create the output paths
        os.makedirs(out_folder, exist_ok=True)
        dst_files = []
        for file in src_files:
            dst_file = os.path.basename(file)
            dst_files.append(os.path.join(out_folder, dst_file))

        # Bundle the arguments for map_single (single function)
        dicts = [self.val_dict.copy() for i in range(len(dst_files))]
        args = list(zip(src_files, dst_files, dicts))

        # Run it
        with Pool(ncpu) as pool:
            for _ in tqdm(pool.imap(self._map_single, args), position=0,
                          total=len(dst_files), file=sys.stdout):
                pass

        # Return the output file paths
        return dst_files

    def _map_single(self, arg, overwrite=True):
        """Map dictionary values from one raster file to another.

        Parameters
        ----------
        arg : list-like
            A list containing an input raster file path, and output raster file
            path and a dictionary (bundled for multiprocessing).

        Returns
        -------
        None.
        """
        # Get arguments
        src = arg[0]
        dst = arg[1]
        val_dict = arg[2]

        # overwrite
        if os.path.exists(dst): 
            if overwrite:
                os.remove(dst)

        # Try to map values from the mapvals dictionary to a new raster
        if not os.path.exists(dst):
            ds = gdal.Open(src)
            crs = ds.GetProjection()
            geom = ds.GetGeoTransform()
            array = ds.ReadAsArray()
            try:
                new_array = np.vectorize(self._map_try)(val_dict, array)
                to_raster(new_array, dst, crs, geom, navalue=-9999)
            except Exception as error:
                print("\n")
                print(src + ": ")
                print(error)
                print("\n")
                raise

    def _map_try(self, val_dict, key):
        """Use a key to return a dictionary value, return a specified value for
        exceptions.

        Parameters
        ----------
        val_dict : dict
            A dictionary of values.
        key : str | int | float
            A key that corresponds to a value in val_dict.

        Returns
        -------
        x : int | float
            The value from val_dict corresponding with the key.
        """
        # Try to retrieve a value with the key
        try:
            x = val_dict[key]

        # Return a specified error value if the key is not present
        except KeyError:
            x = self.err_val

        return x
