# -*- coding: utf-8 -*-
"""
Install revruns to help automate the configuration process.
"""

from setuptools import setup

setup(
    name = 'revruns',
    version = '0.0.1',
    packages = ['revruns'],
    description = "Helps to create config and point files for NREL's reV",
    author = "Travis Williams",
    author_email = "travis.williams@nrel.gov",
    install_requires = ['numpy', 'pandas']
    )