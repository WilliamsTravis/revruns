"""
This extracts all of the coordinates with grid ids as index values from an
NREL resource hdf5 file.
"""
import click
import os
from revruns import get_coordinates

file_help = "The hdf5 file from which to extract coordinates."
save_help = "The target file name for the output coordinate file."

@click.command()
@click.option("--file", "-f", help=file_help)
@click.option("--savepath", "-p", default="points.csv", help=save_help)
def main(file, savepath):
    """Write an hdf5 file coordinate data to a csv file."""
    if not savepath.endswith(".csv"):
        savepath = os.path.splitext(savepath)[0] + ".csv"
    get_coordinates(file, savepath)

if __name__ == "__main__":
    main()
