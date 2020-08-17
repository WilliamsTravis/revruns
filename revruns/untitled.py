import os
import geopandas as gpd
import subprocess as sp
import revruns as rr
import pandas as pd
import rasterio as rio

DP = rr.Data_Path("/shared-projects/rev/projects/soco/rev/runs/reference/aggregation/")
sc_paths = {120: DP.join("120hh/150ps/150ps_sc.csv"),
            140: DP.join("140hh/140hh_150ps/140hh_150ps_sc.csv"),
            160: DP.join("160hh/160hh_150ps/160hh_150ps_sc.csv")}

def csv(path, dst, dataset, res, crs, mask, fillna):

    # This is inefficient
    df = pd.read_csv(path)
    gdf = df.rr.to_geo()
    gdf = gdf[["geometry", dataset]]    

    # We need to reproject to the specified projection system
    gdf = gdf.to_crs(crs)

    # And finally rasterize
    rasterize(gdf, res, dst, mask, fillna)


def rasterize(gdf, res, dst, mask, fillna):

    # Make sure we have the raget directory
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)

    # Not doing this in memory in case it's big
    tmp_src = dst.replace(".tif" ,".gpkg")
    layer_name = os.path.basename(dst).replace(".tif", "")
    gdf.to_file(tmp_src, driver="GPKG")

    # There will only be two columns
    attribute = gdf.columns[1]

    # Write to dst
    sp.call(["gdal_rasterize",
             tmp_src, dst,
             "-a_nodata", "0",
             "-l", layer_name,
             "-a", attribute,
             "-at",
             "-tr", str(res), str(res)])

    # Fill na values
    if fillna:
        sp.call(["gdal_fillnodata.py", dst])

    # If mask is provided
    if mask:
        with rio.open(dst) as raster:
            with rio.open(mask) as rmask:
                r = raster.read(1)
                m = rmask.read(1)
                final = r * m
                profile = raster.profile
        profile["nodata"] = 0
        with rio.Env():
            with rio.open(dst, "w", **profile) as file:
                file.write(final, 1)

    # Get rid of temporary shapefile
    os.remove(tmp_src)
  


for key, path in sc_paths.items():
#     for ds in ["total_lcoe", "mean_cf", "capacity", "lcot", "dist_mi"]:
    for ds in ["total_lcoe"]:
        dst = DP.join("{}hh_150ps_{}.tif".format(key, ds))
        res = 7000
        crs = "epsg:3466"
        csv(path, dst, ds, res, crs, mask=False, fillna=False)
