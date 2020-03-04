#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a exclusion file specifically for the 2020 ATB scenarios.

Master sheet:

https://docs.google.com/spreadsheets/d/
1lqM6iAIXfhaKj9oT3Q7d6WHNCyNCq6hRiBmUbJkAPBg/edit#gid=1113222871

Created on Mon Mar  2 09:31:09 2020

@author: twillia2
"""


from gdalmethods import Data_Path
import subprocess as sp
from glob import glob

# Exclusions Data Path
DPE = Data_Path("/projects/rev/data/wind_deployment_potential")

# Characterization Data path
DPC = Data_Path(DPE.join("characterizations"))

# Path to master table
MASTER_PATH = ("https://docs.google.com/spreadsheets/d/1lqM6i"
               "AIXfhaKj9oT3Q7d6WHNCyNCq6hRiBmUbJkAPBg/")




def atb_exclusions():
    
    # Read in google sheets?
    master = Spread(spread=MASTER_PATH, sheet="Scenarios")



def create_exclusions(paths):
    
