# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name='revruns',
    version='0.0.1',
    packages=['revruns'],
    description=("Functions and CLIs that to help to configure, run, "
		         "and check outputs for NREL's Renewable Energy Technical "
                 "Potential Model (reV)."),
    author="Travis Williams",
    author_email="travis.williams@nrel.gov",
    install_requires=['h5py', 'numpy', 'pandas'],
    entry_points={"console_scripts":
                      [
                       "rrbatch_collect = revruns.rrbatch_collect:main",
                       "rrbatch_hack = revruns.rrbatch_hack:main",
                       "rrbatch_logs = revruns.rrbatch_logs:main",
                       "rrcheck = revruns.rrcheck:main",
                       "rrerun = revruns.rrerun:main",
                       "rrexclusion = revruns.rrexclusion:main",
                       "rrmax = revruns.rrmax:main",
                       "rrmin = revruns.rrmin:main",
                       "rrpoints = revruns.rrpoints:main",
                       "rrsetup = revruns.rrsetup:main",
                       "rrshape = revruns.rrshape:main"
                       ]
                  }
    )
