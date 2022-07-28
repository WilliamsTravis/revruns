# -*- coding: utf-8 -*-
"""Package Data Helpers.

Created on Mon May 23 20:31:32 2022

@author: twillia2
"""
import os

from importlib import resources


class Paths:
    """Methods for handling paths to package data."""

    @classmethod
    @property
    def data(cls):
        """Return data root directory."""
        return cls.home.joinpath("data")

    @classmethod
    @property
    def home(cls):
        """Return application home directory."""
        return resources.files("revruns")

    @classmethod
    @property
    def paths(cls):
        """Return posix path objects for package data items."""
        contents = resources.files("revruns")
        paths = {}
        for path in cls.data.rglob("*"):
            if path.is_file():
                name = os.path.splitext(path.name)[0].lower()
                paths[name] = path
        return paths


class Paths:
    """Methods for handling paths to package data."""

    @classmethod
    @property
    def data(cls):
        """Return data root directory."""
        return cls.home.joinpath("data")

    @classmethod
    @property
    def home(cls):
        """Return application home directory."""
        return resources.files("revruns")

    @classmethod
    @property
    def paths(cls):
        """Return posix path objects for package data items."""
        contents = resources.files("revruns")
        paths = {}
        for folder in cls.data.iterdir():
            name = os.path.splitext(folder.name)[0].lower()
            paths[name] = folder
        return paths
