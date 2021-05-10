# -*- coding: utf-8 -*-
"""Probably another fruitless attempt to extract info from qgis project files.

Created on Mon May 10 11:34:06 2021

@author: twillia2
"""
import os

from qgis.core import QgsProject


QGZ = "~/nrel/transmission/transmission_barriers.qgz"
LAYER = "HIFLD Open Federal_Lands"
FILE = "~/nrel/transmission/transmission_barriers.qgz"
TEMPLATE = "~/nrel/lbnl_char/data/rasters/template.tif"


class QProject:
    """Methods for extracting information from a QGIS project file."""

    def __init__(self, path):
        """Initialize revruns QProject object."""
        self.path = os.path.expanduser(path)

    def __repr__(self):
        """Return QProject representation string."""
        msg = f"<QProject instance: path={self.path}>"
        return msg

    @property
    def project(self):
        """Return a pyqgis project instance."""
        project = QgsProject.instance()
        project.read(self.path)
        return project

    @property
    def layers(self):
        """Return all map layer objects."""
        # Get layer dict with original instance keys
        original_layers = self.project.layerStore().mapLayers()

        # Make comprehensible keys
        layers = {}
        for value in original_layers.values():
            name = value.name()
            layers[name] = value

        return layers      

    @property
    def displayed(self):
        """Open the qgis project file and subset data for shown features."""
        # Create holder for all layers and features
        shown = {}
        for name, layer in self.layers.items():
            shown[name] = {}
            shown[name]["features"] = {}

            # Select our target layer and get all rendering information
            renderer = layer.renderer()
            symbology = renderer.dump().split("\n")
    
            # Get the field (data frame column) displayed from the first item
            if ": idx " in symbology[0]:
                column = symbology[0].split(": idx ")[-1]
                symbology = symbology[1:]
            else:
                column = None
            shown[name]["field"] = column

            # Whether a field is displayed is knowable from the rgb string
            for field in symbology:
                key = field.split("::")[0]
                if key != "":
                    
                    try:
                        displayed = int(field.split(":")[-1])
                    except:
                        displayed = 0
                    shown[name]["features"][key] = int(displayed)

        return shown


if __name__ == "__main__":
    self = QProject(QGZ)
