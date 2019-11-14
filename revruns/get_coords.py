import h5py as hp
import pandas as pd
import sys

file = sys.argv[1]
savepath = sys.argv[2]

def get_coordinates(file, savepath):
    """Get all of the coordintes and their grid ids from an hdf5 file"""
    # Get numpy array of coordinates
    with hp.File(file, mode="r") as f:
        crds = f["coordinates"][:]

    # Create a data frame and save it
    lats = crds[:, 0]
    lons = crds[:, 1]
    df = pd.DataFrame({"lat": lats, "lon": lons})
    df.to_csv(savepath)

if __name__ == "__main__":
    print("Saving " + file + " to " + savepath)
    get_coordinates(file, savepath)
