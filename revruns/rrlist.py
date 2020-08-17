# -*- coding: utf-8 -*-
"""List all revruns clis and what they do.
"""

import importlib
import pkgutil

import click
import pandas as pd
import revruns

from colorama import Fore, Style
from tabulate import tabulate

df = pd.DataFrame ({'Text': ['abcdef', 'x'], 'Value': [12.34, 4.2]})



def build_docs():
    rrdocs = {}
    for p in pkgutil.iter_modules(revruns.__path__):
        if "rr" in p.name[:2]:
            rr = importlib.import_module("revruns.{}".format(p.name))
            try:
                rrdocs[p.name] = rr.__doc__.replace("\n", "")
            except:
                print("Docs for " + p.name + " not found.")
                pass
    return rrdocs


@click.command()
def main():
    """List all revruns clis and what they do."""

    rrdocs = build_docs()
    rrdf = pd.DataFrame(rrdocs, index=[0]).T
    rrdf["rr"] = rrdf.index
    rrdf.columns = ["desc" ,"rr"]
    rrdf = rrdf[["rr", "desc"]]
    print_cols = ["RRCLI", "Description"]
    rrdf["desc"] = rrdf["desc"].apply(
            lambda x: Fore.GREEN + x + Style.RESET_ALL)
    print(tabulate(rrdf, showindex=False, headers=print_cols,
                   tablefmt="simple"))

if __name__ == "__main__":
    main()
