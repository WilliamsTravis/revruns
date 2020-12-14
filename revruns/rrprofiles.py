# -*- coding: utf-8 -*-
"""Check samples of reV timeseries profiles.

Created on Wed Dec  9 10:34:06 2020

@author: twillia2
"""
import click
import datetime as dt
import os

import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from revruns.rr import isint


HELP = {
    "file": ("The HDF5 file containing the target dataset. (str)"),
    "file2": ("An optional second HDF5 file containing a dataset to be "
              "compared to the first. (str)"),
    "dataset": ("The HDF5 dataset containing the target timeseries. (str)"),
    "dataset2": ("An optional second HDF5 dataset containing a timeseries to "
                 "be compared to the first. (str)"),
    "time_1": ("The first time index position of the timeseries sample to "
               "display. Defaults to 0. (int)"),
    "time_2": ("The second time index position of the timeseries sample to "
               "display. Defaults to 500. (int)"),
    "agg_fun": ("The aggregation function to apply to the spatial axis of the "
                "dataset.  Defaults to 'mean'. (str)"),
    "save": ("Save the plot to as an image to file. If no save_path is "
             "provided it will generate a file name using the other arguments. "
             "Default is False. (boolean)"),
    "save_path": ("The file name to use if saving image. (str)")
}
COLORS = ["blue", "orange"]


def sample_args():
    """Return a set of sample arguments.
    
    Run locals().update(kwargs) for named objects.
    """
    kwargs = dict(
        file = ("/shared-projects/rev/projects/perform/wind/results/perform/"
                "day_ahead_2018_rf16_gen.h5"),
        file2 = ("/shared-projects/rev/projects/perform/wind/results/era5/"
                 "ercot_2017.h5"),
        dataset = "cf_profile",
        dataset2 = "gen_profile",
        time_1 = 0,
        time_2 = 500,
        agg_fun = "mean",
        save = True,
        save_path = None
    )
    args = list(kwargs.values())
    return args, kwargs


def find_scale(data):
    """Find the scale factor of an HDF5 dataset if it exists.

    Parameters
    ----------
    data : h5py._hl.dataset.Dataset
        An HDF5 dataset.

    Returns
    -------
    scale : int | float
        A scale factor to use to scale the values of a dataset.
    """
    if "scale_factor" in data.attrs.keys():
        scale = data.attrs["scale_factor"]
    else:
        scale = 1
    return scale


def make_path(*args):
    """Make a file name from the rrprofiles arguments."""
    elements = []
    for arg in args:
        if arg:
            if not isinstance(arg, bool):
                try:
                    arg = os.path.basename(arg).replace(".h5", "")
                    arg_pieces = []
                    for s in arg.split("_"):
                        numbers = "".join([i for i in s if isint(i)])
                        alpha = "".join([i for i in s if not isint(i)])
                        if alpha:
                            alpha = alpha[0]
                        piece = alpha + numbers
                        arg_pieces.append(piece)
                    reduced_arg = "".join(arg_pieces)
                    elements.append(reduced_arg)
                except TypeError:
                    arg = str(arg)
                    elements.append(arg)
    path = "_".join(elements) + ".png"
    return path


class Profiles:
    """Plot a set of sample timeseries profiles from reV output."""

    def __init__(self, file, dataset, time, agg_fun="mean", file2=None,
                 dataset2=None):
        self.file = file
        self.dataset = dataset
        self.time = time
        self.agg_fun = agg_fun
        self.file2 = file2
        self.dataset2 = dataset2
        self.units = {}

    def get_data(self, file, dataset):
        """Get the data associated with a dataset and file name."""
        # Find the aggregation function and set the file key
        fun = np.__dict__[self.agg_fun]
        name = os.path.splitext(os.path.basename(file))[0]

        # Retrieve data
        with h5py.File(file, "r") as ds:
            data = ds[dataset]
            scale = find_scale(data)
            units = self._get_units(file, dataset)
            if len(data.shape) == 2:
                series = data[self.time[0]: self.time[1], :] / scale
                series = fun(series, axis=1)
            else:
                raise ValueError(dataset + " is not a time series.")

        # Set units for later
        if dataset not in self.units:
            self.units[dataset] = {}
        self.units[dataset][name] = units
        return series

    def plot_single(self, ax, array, dataset, file):
        """Add a single timeseries to a plot."""
        # Set up the time stamp
        time_index = self.time_index
        step = int(np.ceil(len(time_index) / 10))
        time_labels = time_index[::step]
        time_ticks = np.arange(0, len(time_index), step)

        # Set up the x-axis
        x = np.arange(0, array.shape[0], 1)
        ax.xaxis.set_ticks(time_ticks)
        ax.xaxis.set_ticklabels(time_labels, ha="left", rotation=315)

        # Set up the y-axis
        name = os.path.splitext(os.path.basename(file))[0]
        ax.set_ylabel(dataset + " - " + self.units[dataset][name])

        # Finally Plot
        ax.plot(x, array)

    def plot(self, save, save_path=None):
        """Plot all timeseries requested."""
        # Retrieve all arrays
        data = self.data

        # Figure out the dimensions of the subplot
        plot_shape = len(data.keys())

        # Set legend handles
        handles = []
        key1 = list(data.keys())[0]
        for i, name in enumerate(data[key1].keys()):
            handles.append(mpatches.Patch(color=COLORS[i], label=name))

        # Create a figure
        fig, axes = plt.subplots(plot_shape, 1)
        fig.set_size_inches(10, 7)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        # Same datasets are plotted together
        for i, (dataset, entry) in enumerate(data.items()):
            ax = axes[i]
            for name, array in entry.items():
                self.plot_single(ax, array, dataset, name)
        fig.legend(handles=handles, framealpha=1)
        fig.tight_layout()
        if save or save_path:
            if not save_path:
                save_path = make_path(self.file, self.file2, self.dataset,
                                      self.dataset2, self.time[0], self.time[0])
            if os.path.exists(save_path):
                os.remove(save_path)
            plt.savefig(save_path)
        else:
            plt.show()

    @property
    def time_index(self):
        """Retrieve the datetime stamps for the chosen time range."""
        with h5py.File(self.file, "r") as ds:
            encoded_index = ds["time_index"][self.time[0]: self.time[1]]
        time_index = [time.decode()[5:19] for time in encoded_index]
        return time_index

    @property
    def data(self):
        """Retrieve a dictionary of each timeseries data requested."""
        # Create a container
        timeseries = {
            self.dataset: {},
        }
        if self.dataset2:
            timeseries[self.dataset2] = {}

        # First file/first dataset
        name = os.path.splitext(os.path.basename(self.file))[0]
        series = self.get_data(self.file, self.dataset)
        timeseries[self.dataset][name] = series

        # First file/second dataset
        if self.dataset2:
            series = self.get_data(self.file, self.dataset2)
            timeseries[self.dataset2][name] = series

        # Second file
        if self.file2:
            # First Dataset
            name = os.path.splitext(os.path.basename(self.file2))[0]
            series = self.get_data(self.file2, self.dataset)
            timeseries[self.dataset][name] = series

            # Second dataset
            if self.dataset2:
                series = self.get_data(self.file2, self.dataset2)
                timeseries[self.dataset2][name] = series

        return timeseries

    def _get_units(self, file, dataset):
        """Find the units of an HDF5 dataset if it exists.

        Parameters
        ----------
        data : h5py._hl.dataset.Dataset
            An HDF5 dataset.

        Returns
        -------
        scale : str
            A string representing the dataset units.
        """
        with h5py.File(file, "r") as ds:
            data = ds[dataset]
            if "units" in data.attrs.keys():
                units = data.attrs["units"]
            else:
                units = ""
        return units


@click.command()
@click.argument("file", required=True)
@click.argument("dataset", required=True)
@click.option("--time_1", "-t1", default=0, help=HELP["time_1"])
@click.option("--time_2", "-t2", default=500, help=HELP["time_2"])
@click.option("--agg_fun", "-a", default="mean", help=HELP["agg_fun"])
@click.option("--file2", "-f2", default=None, help=HELP["file2"])
@click.option("--dataset2", "-d2", default=None, help=HELP["dataset2"])
@click.option("--save", "-s", is_flag=True, help=HELP["save"])
@click.option("--save_path", "-sp", default=None, help=HELP["save_path"])
def main(file, dataset, time_1, time_2, agg_fun, file2, dataset2, save,
         save_path):
    """REVRUNS - Timeseries profile check.

    Generate a line plot of sampled timeseries for a reV generated
    timeseries.
    """
    time = [int(time_1), int(time_2)]
    plotter = Profiles(file, dataset, time, agg_fun, file2, dataset2)
    plotter.plot(save, save_path)


if __name__ == "__main__":
    main()
