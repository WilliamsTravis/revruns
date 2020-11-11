"""Join area weighted field to reV supply-curve table.

This will attempt to recreate the resource point aggregation in the reV
supply-curve-aggregation module. This is an example case using captial costs.

Steps:
1) Associate variable with resource points
2) Use the gid_counts list to weight by area within each resource point
3) Aggregate by a user-defined function using the above weights

Secondary Steps:
4) If needed, aggregate by a secondary grouping variable, this time weighting
   by n_gids.


Created on Thu Nov 8:24:20 2020

@author Travis
"""
import json

import numpy as np
import pandas as pd
import revruns as rr

from tqdm import tqdm

tqdm.pandas()


DP = rr.Data_Path("/shared-projects/rev/projects/weto/bat_curtailment")
SC_FPATH = DP.join("data", "fixed_results", "no_curtailment_sd1_sc.csv")
RES_FPATH = DP.join("reruns", "econ", "_inputs", "capital_costs_100ps.csv")


def get_capex(params):
    """Map capital costs in $/kw to resource points."""
    pdf = pd.read_csv(params)
    pdf["capital_cost_kw"] = pdf["capital_cost"] / pdf["kw"]
    return pdf[["gid", "capital_cost_kw"]]


def res_to_sc(sc_fpath, res_fpath, field, left_on, right_on, weight_col):
    """Append resource-scale value to supply-curve points.

    Parameters
    ----------
    sc_fpath : str
        Path to a reV supply curve table CSV.
    res_fpath : str
        Path to reV resource point CSV with target variable mapping.
    field : str
        Column name of the field in the resource mapping file to map to the reV
        supply-curve table.

    Returns
    -------
    pandas.core.frame.DataFrame
        The reV supply-curve table with the extra field
    """
    def row_map(row, left_on, weight_col, value_dict):
        gids = json.loads(row[left_on])
        values = [value_dict[i] for i in gids]
        weights = json.loads(row[weight_col])
        wavg = np.average(values, weights=weights)
        return wavg

    sc = pd.read_csv(sc_fpath)
    cdf = get_capex(res_fpath)
    value_dict = dict(zip(cdf["gid"], cdf[field]))
    new_field = "mean_" + field

    sc[new_field] = sc.apply(row_map, axis=1, left_on=left_on,
                             weight_col=weight_col, value_dict=value_dict)
    return sc


if __name__ == "__main__":
    sc_fpath = SC_FPATH
    res_fpath = RES_FPATH
    field = "capital_cost_kw"
    left_on = "res_gids"
    right_on = "gid"
    weight_col = "gid_counts"
    sc = res_to_sc(sc_fpath, res_fpath, field, left_on, right_on, weight_col)
    sc.to_csv(sc_fpath, index=False)
