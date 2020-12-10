# -*- coding: utf-8 -*-

from setuptools import setup

setup(
    name='revruns',
    version='0.0.2',
    packages=['revruns'],
    description=("Functions and CLIs that to help to configure, run, "
	             "and check outputs for NREL's Renewable Energy Technical "
                 "Potential Model (reV)."),
    author="Travis Williams",
    author_email="travis.williams@nrel.gov",
    package_data={"revruns": ["data/*txt", "data/*.csv", "data/*.xlsx"]},
    entry_points={"console_scripts":
                      [
                       "rrbatch_collect = revruns.rrbatch_collect:main",
                       "rrbatch_hack = revruns.rrbatch_hack:main",
                       "rrbatch_logs = revruns.rrbatch_logs:main",
                       "rrcheck = revruns.rrcheck:main",
                       "rrconnections = revruns.rrconnections:main",
                       "rrerun = revruns.rrerun:main",
                       "rrexclusion = revruns.rrexclusion:main",
                       "rrgraphs = revruns.rrgraphs:main",
                       "rrgeoref = revruns.rrgeoref:main",
                       "rrlogs = revruns.rrlogs:main",
                       "rrlist = revruns.rrlist:main",
                       "rrpipeline = revruns.rrpipeline:rrpipeline",
                       "rrpoints = revruns.rrpoints:main",
                       "rrprofiles = revruns.rrprofiles:main",
		       "rrshape = revruns.rrshape:main",
                       "rrsetup = revruns.rrsetup:main",
                       "rraster = revruns.rraster:main",
                       "rrtemplates = revruns.rrtemplates:main"
                       ]
                  }
    )
