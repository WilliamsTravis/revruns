# -*- coding: utf-8 -*-
"""Probably another fruitless attempt to extract info from qgis project files.

Created on Mon May 10 11:34:06 2021

@author: twillia2
"""
from qgis.core import QgsProject


QGZ = "/Users/twillia2/Desktop/weto/transmission/transmission_barriers.qgz"
LAYER = "HIFLD Open Federal_Lands"


class QProject:
    """Methods for extracting information from a QGIS project file."""

    def __init__(self, path):
        """Initialize revruns QProject object."""
        self.path = path

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

    def shown_fields(self, layer):
        """Open the qgis project file and subset data for shown features."""
        # These are all our map layers
        layers = self.project.layerStore().mapLayers()

        # Make comprehensible keys for these layers
        layers_dict = {}
        for value in layers.values():
            name = value.name()
            layers_dict[name] = value

        # Select our target layer and get all rendering information
        layer = layers_dict[layer]
        renderer = layer.renderer()
        symbology = renderer.dump().split("\n")

        # Whether or not a field is displayed is knowable from the rgb string
        shown = {}
        for field in symbology:
            if "color" in field:
                key = field.split("::")[0]
                displayed = field.split(":")[-1]
                shown[key] = int(displayed)

        return shown


if __name__ == "__main__":
    self = QProject(QGZ)
