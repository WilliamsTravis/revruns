# -*- coding: utf-8 -*-
"""Test rreformatter

Created on Fri Jan 28 16:57:22 2022

@author: twillia2
"""
from revruns import rreformatter


TEMPLATE = "data/rasters/pr_template.tif"
INPUTS = {
    "cyclones_name": {
        "path": "data/shapefiles/PR_Tropical_Cyclone_Storm_Segments_32161.geojson",
        "field": "stormName",
        "buffer": 0
    },
    "significant_wave_height_annual": {
        "path": "data/shapefiles/pr_wave_sig_ht_32161.geojson",
        "field": "ann_ssh",
        "buffer": 0
    }
}


def test():
    """Test sample refromatting routine."""
    

# if __name__ == "__main__":
#     self = rreformatter.Reformatter(TEMPLATE, INPUTS)
