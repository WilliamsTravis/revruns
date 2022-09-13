#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 19:57:26 2022

@author: twillia2
"""

def find_mixed(df):
    """Return a dictionary of unique dtypes for each seroes in a pandas df."""
    for i, row in df.iterrows():
        nd = row.to_dict()
        if i == 0:
            dtypes = {key: [type(v)] for key, v in nd.items()}
        else:
            for key, value in nd.items():
                dtype = type(value)
                if dtype not in dtypes[key]:
                    dtypes[key].append(dtype)
    return dtypes
