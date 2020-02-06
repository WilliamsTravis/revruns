"""
Activate a reV environment and run this in a folder with everything a normal
reV run would require (including the batch configuration) and it should take
care of the rest.
"""
import json
import os
import subprocess as sp
import time
from glob import glob
from tqdm import tqdm

def get_batch_outputs(folder):
    """ Get a list of the output folders from a batched rev run.

    Parameters
    ----------
    folder (str)
        A folder with all of the configuration files and batched run
        folders.

    Returns
    -------
    list: List of all batched run output folder path strings

    Sample Argument
    ---------------
    folder = "/lustre/eaglefs/projects/rev/new_projects/sergei_doubleday/30min"
    """
    with open("config_gen.json", "r") as config_file:
        config_gen = json.load(config_file)
    module = config_gen["project_control"]["technology"]
    out = config_gen["directories"]["output_directory"].replace("./", "")
    outputs = glob(module + "_*/" + out + "*")

    return outputs



# Create all of the batch folders
sp.call(["reV",
         "-c",
         "config_batch.json",
         "batch",
         "--dry-run"])

# List of the folders
outputs = glob("pv_*/batchout")
folders = glob("pv_*")

# To have different job names, we need to have different batch names
# We'll have to use the config entry I think, and just erase the original
new_outs = [outputs[i] + "_" + "{:02d}".format(i) for i in range(len(outputs))]

for i, folder in enumerate(folders):
    # get the configuration file name
    config_gen = os.path.join(folder, "config_gen.json")

    # Read this as a dictionary
    with open(config_gen, "r+") as file:
        config = json.load(file)

    # replace the output_directory
    new_output = "./batchout" + "_" + "{:02d}".format(i)
    config["directories"]["output_directory"] = new_output

    # rewrite configuration file
    with open(config_gen, "w") as new_file:
        new_file.write(json.dumps(config, indent=4))

# Submitting each job separately
print("Submitting {} jobs.".format(len(folders)))
for folder in tqdm(folders):
    sp.call(["reV",
             "-c",
             os.path.join(folder, "config_gen.json"),
             "generation"])
    time.sleep(5)
