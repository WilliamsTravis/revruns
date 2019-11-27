
from dask.diagnostics import ProgressBar as pb
import click
import dask.array as da
import h5py
import sys

# Help statements
file_help = "The hdf5 file name."
dataset_help = "The name of the data set within the hdf5 file."

# The command
@click.command()
@click.option("--file", "-f", help=file_help)
@click.option("--dataset", "-d", help=dataset_help)
def main(file, dataset):
    """Prints the maximum value of an hdf5 file data set."""
    # Open the data set
    ds = h5py.File(file, mode="r")[dataset]

    # Create the dask data array
    dds = da.from_array(ds)

    # Let's see how many partitions we're using
    npart = dds.npartitions

    # Broadcast what's happening
    print("Calculating max value for {}:{} using {} partitions".format(file, dataset, npart))

    # And compute the max value
    with pb():
        result = dds.max().compute()

    # Print result
    print(result)

if __name__ == "__main__":
    main()

