#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Update power curve for sam configuration.

Created on Thu Feb 25 15:40:05 2021

@author: twillia2
"""
from revruns import rr


DP = rr.Data_Path("/shared-projects/rev/projects/awe")
PC_TABLE = DP.join("data", "tables", "AWES power curve for reV.xlsx")
SAM = DP.join("rev", "generation", "config_sam.json")
