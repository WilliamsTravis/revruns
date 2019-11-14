"""
This extracts all of the coordinates with grid ids as index values from an
NREL resource hdf5 file.
"""

from revruns import get_coordinates
import sys

file = sys.argv[1]
savepath = sys.argv[2]

if __name__ == "__main__":
    get_coordinates(file, savepath)
