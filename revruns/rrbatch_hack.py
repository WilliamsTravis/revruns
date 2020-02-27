import click
import json
import os
import subprocess as sp
import time
from glob import glob


FOLDER_HELP = ("Path to a folder with a set of configuration files for a "
               "batched reV run. Defaults to current directory. (str)")

def dryrun():
    """Run the --dry-run option of batch to create target folders."""

    # Create all of the batch folders
    sp.call(["reV",
             "-c",
             "config_batch.json",
             "batch",
             "--dry-run"])


def get_names(folder):
    """ Get parameters from the config files."""
    with open("config_gen.json", "r") as config_file:
        config_gen = json.load(config_file)
    module = config_gen["project_control"]["technology"]
    outfolder = config_gen["directories"]["output_directory"].replace("./", "")

    return module, outfolder


def rename(folder):
    """Append numbers to output directories."""

    # get empty directories - outputfolders don't exist yet
    dryrun()

    # Create a new set of number output paths
    module, outfolder = get_names(folder)
    batch_folders = glob(module + "_*/")

    for i, folder in enumerate(batch_folders):

        # get the configuration file name
        config_gen = os.path.join(folder, "config_gen.json")
    
        # Read this as a dictionary
        with open(config_gen, "r+") as file:
            config = json.load(file)
    
        # replace the output_directory
        new_output = "./" + outfolder + "_" + "{:02d}".format(i)
        config["directories"]["output_directory"] = new_output
    
        # rewrite configuration file
        with open(config_gen, "w") as new_file:
            new_file.write(json.dumps(config, indent=4))

@click.command()
@click.option("-f", "--folder", default=".", help=FOLDER_HELP)
def main(folder):
    """Run the batched reV generation model until the bugs are worked out.

    Activate a reV environment and run this in a folder with everything a
    normal reV run would require (including the batch configuration) and it
    should take care of the rest.
    """
    
    # Create target batch run folders
    dryrun()

    # Rename configuration output folders
    rename(folder)
    
    # Get the batch folders
    module, outfolder = get_names(folder)
    batch_folders = glob(module + "_*/")

    # Submit each job separately
    print("Submitting {} jobs.".format(len(batch_folders)))
    for folder in batch_folders:
        sp.call(["reV",
                 "-c",
                 os.path.join(folder, "config_gen.json"),
                 "generation"])
        time.sleep(5)

if __name__ == "__main__":
    main()
