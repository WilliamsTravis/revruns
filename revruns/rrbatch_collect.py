#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Postprocessing for batch model outputs. These are heavily nested and need to
be consolidated some how.


Things to do:

    1) Catch cases where a pipeline is not complete. The status log is not
       consistent at catching this situation btw.

Created on Tue Jan 21 08:38:11 2020

@author: twillia2
"""
import click
import h5py
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from reV.utilities.utilities import parse_year


FOLDER_HELP = ("Path to a folder with a completed set of batched reV runs. "
               "Defaults to current directory. (str)")
DESC_HELP = ("A string representing a description of the data sets. "
             "Defaults to None. (str)")
SHORT_HELP = ("Truncate data set keys to include only variable parameters "
              "Defaults to False. (boolean)")
SAVE_HELP = ("The filename of the output HDF5 file. Defaults to the "
             "name of the containing folder. The year of the output file "
             "will be appended to this file name before the extension. (str)")


def batch_key(key, value):
    """Take a single reV parameter and value and convert it to a reV batch key.

    Parameters
    ----------
    key (str)
        A reV configuration key (e.g. 'array_type')

    value (float | int | str)
        A value corresponding to the key

    Returns
    -------
    str
        A single character string corresponding to value that the reV
        batch module assigns to this key value pair
    """
    # Reduce the key
    rkey = "".join([k[0] for k in key.split("_")])

    # Reduce the value
    rvalue = "{}".format(value).replace(".", "")

    # Join
    final_key = "".join([rkey, rvalue])

    # Slight complication if this value would be parsed as a year
    try:
        parse_year(final_key)
        final_key += "0"
    except RuntimeError:
        pass

    return final_key, rkey


def batch_keys(config_path="./config_batch.json"):
    """
    Use the batch configuration file to infer

    Parameters
    ----------
    config_path (str)
        File path for the batch configuration json file.

    Returns
    -------
    dict: A dictionary of reV generated batch keys and the parameters
        they represent.
    """
    # Create a dictionary from the batch config file
    with open(config_path, "r") as config_file:
        config_batch = json.load(config_file)

    # There might be multiple sets of parameter combinations
    arg_sets = config_batch["sets"]
    key_dict = {"key": [],
                "parameter": [],
                "arg": [],
                "value": []}

    # Set tag first then naming logic for each argument
    for arg_set in arg_sets:
        tag = arg_set["set_tag"]
        key_dict["key"].append(tag)
        key_dict["parameter"].append("set_tag")
        key_dict["arg"].append(tag)
        key_dict["value"].append("")

        # Each arg key is the first letter of each "_"-separated piece
        args = arg_set["args"]
        for arg_key in args.keys():
            for arg_val in args[arg_key]:
                final_key, rkey = batch_key(arg_key, arg_val)
                key_dict["key"].append(final_key)
                key_dict["parameter"].append(arg_key)
                key_dict["arg"].append(rkey)
                key_dict["value"].append(str(arg_val))

    # Combine everything into a data frame?
    key_df = pd.DataFrame(key_dict)

    return key_df


def get_outputs():
    """ Get a list of the output folders from a batched rev run.

    Parameters
    ----------
    folder (str)
        A folder with all of the configuration files and batched run
        folders.

    Returns
    -------
    list: List of all batched run output folder path strings
    """
    with open("config_gen.json", "r") as config_file:
        config_gen = json.load(config_file)
    module = config_gen["technology"]
    outputs = glob(module + "_*/" + module + "_*h5")

    return outputs


def to_sarray(df):
    """
    Encode data frame values, return a structured array and an array of
    dtypes. This is needed for storing pandas data frames in the h5 format.
    """

    def make_col_type(col_type, col):
        try:
            if 'numpy.object_' in str(col_type.type):
                maxlens = col.dropna().str.len()
                if maxlens.any():
                    maxlen = maxlens.max().astype(int)
                    col_type = ('S%s' % maxlen)
                else:
                    col_type = 'f2'
            return col.name, col_type
        except:
            print(col.name, col_type, col_type.type, type(col))
            raise

    v = df.values
    types = df.dtypes
    numpy_struct_types = [make_col_type(types[col], df.loc[:, col]) for
                          col in df.columns]
    dtypes = np.dtype(numpy_struct_types)
    array = np.zeros(v.shape[0], dtypes)
    for (i, k) in enumerate(array.dtype.names):
        try:
            if dtypes[i].str.startswith('|S'):
                array[k] = df[k].str.encode('utf-8').astype('S')
            else:
                array[k] = v[:, i]
        except:
            print(k, v[:, i])
            raise

    return array, dtypes


def reset_key(param_string, config_path="./config_batch.json"):
    """ Return index positions of parameters with multiple entries in a
    reV batch configuration. This avoids key formatting discrepencies.

    Parameters
    ----------
    config_path : str, optional
        Path to a batch configureation json file. The default is
        "./config_batch.json".

    Returns
    -------
    dict: Dictionary of index position for each batch set where there are
          more than one entry

    param_string = 'pv_at20_a16500_dar12_g04_ie960_l140757_mt00_sc10000_t00'
    """
    # Include decimals in new keys
    key_df = batch_keys(config_path)
    key_df["short_value"] = key_df["arg"] + key_df["value"]
    replace_dict = dict(zip(key_df["key"], key_df["short_value"]))

    # Only include variable parameters
    key_df["repeat"] = key_df.groupby("arg")["arg"].transform(lambda x: len(x))
    rep_df = key_df[["arg", "repeat"]].drop_duplicates()
    idx = np.where(rep_df["repeat"] > 1)[0]

    # The generated string might have extra zeros
    param_bits = np.array(param_string.split("_"))
    set_tag = param_bits[0] + "_"
    param_bits = param_bits[idx]
    new_key = set_tag + "_".join([replace_dict[p] for p in param_bits])

    return new_key


@click.command()
@click.option("--folder", "-f", default=".", help=FOLDER_HELP)
@click.option("--name", "-n", default=None, help=SAVE_HELP)
@click.option("--desc", "-d", default=None, help=DESC_HELP)
@click.option("--short_keys", "-s", is_flag=True, help=SHORT_HELP)
@click.option("--verbose", "-v", is_flag=True)
def main(folder, name, desc, short_keys, verbose):
    """Take all of the outputs of all batched runs in a reV project folder
    and consolidate them into a single HDF5 file.

    sample args:

    folder = '/lustre/eaglefs/projects/rev/new_projects/sergei_doubleday/5min/'
    name = 'test'
    short_keys = True
    desc = None
    verbose = False
    """
    # This will be easier if we change to the folder directory
    os.chdir(folder)
    cwd = os.getcwd()

    # Name this run after its containing folder if argument not provided
    if not name:
        name = Path(cwd).parts[-1]

    # If the user provides an extension get rid of it
    if "." in name:
        name = os.path.splitext(name)[0]

    # Get all of the output folders
    with open("./config_gen.json", "r") as config_file:
        config_gen = json.load(config_file)
    module = config_gen["technology"]
    outputs = glob(module + "_*/" + module + "_*h5")

    # Get all of the years
    years = config_gen["analysis_years"]

    # Get the key reference
    batch_config = os.path.join(folder, "config_batch.json")
    key_df = batch_keys(batch_config)

    # Reformat key_df for hdf5 storage
    key_sdf, key_dtypes = to_sarray(key_df)

    # One year at a time for sanity
    for year in years:
        year = str(year)
        target_file = name + "_" + year + ".h5"
        year_outputs = [o for o in outputs if year in o]
        if verbose:
            print("Collecting batched reV run outputs to " + target_file)

        if os.path.exists(target_file):
            os.remove(target_file)

        with h5py.File(target_file, "w") as new_file:

            # Add in all of the data from each output folder
            for file in year_outputs:

                params = Path(file).parts[0]
                with h5py.File(file, "r") as old_file:
                    keys = list(old_file.keys())
                    for key in keys:

                        # We don't need meta or time indexes more than once
                        if key not in ["meta", "time_index"]:
                            if short_keys:
                                nm = key + "_" + reset_key(params)
                            else:
                                nm = params
                            new_file.create_dataset(name=nm,
                                                    data=old_file[key])
                            for k in old_file[key].attrs.keys():
                                new_file[nm].attrs[k] = old_file[key].attrs[k]

                        # But we do need them once
                        elif key not in new_file.keys():
                            nm = key
                            new_file.create_dataset(name=nm,
                                                    data=old_file[key])
                            for k in old_file[key].attrs.keys():
                                new_file[nm].attrs[k] = old_file[key].attrs[k]

            # Add in the key reference data frame
            new_file.create_dataset(name="key_reference",
                                    data=key_sdf,
                                    dtype=key_dtypes)

            # Add in a description
            if desc:
                new_file.attrs["description"] = desc


if __name__ == "__main__":
    main()
