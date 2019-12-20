"""
This extracts all of the coordinates with grid ids as index values from an
NREL resource hdf5 file.
"""
import click
import os

file_help = "The hdf5 file from which to extract coordinates."
save_help = "The target file name for the output coordinate file."


def get_coordinates(file, savepath):
    """Get all of the coordintes and their grid ids from an hdf5 file"""
    # Get numpy array of coordinates
    with h5py.File(file, mode="r") as pointer:
        crds = pointer["coordinates"][:]

    # Create a data frame and save it
    lats = crds[:, 0]
    lons = crds[:, 1]
    data = pd.DataFrame({"lat": lats, "lon": lons})
    data.to_csv(savepath, index=False)


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
