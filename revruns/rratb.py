# -*- coding: utf-8 -*-
"""revruns ATB

Methods for accessing ATB values and updating versions.

Author: twillia2
Date: Fri Apr 15 09:40:28 MDT 2022
"""


class ATB:
    """Methods for retrieving data from the Annual Technology Baseline."""

    def __init__(self, table_fpath=None, year=2030, tech="wind"):
        """Initialize ATB object."""
        self.table_fpath = table_fpath
        self.url = "https://data.openei.org/submissions/4129"  # <------------- Will it consistently be stored at 4129?
        self.tech = tech
        self.year = year

    def __repr__(self):
        """Return ATB representation string."""
        msgs = [f"{k}={v}" for k, v in self.__dict__.items()]
        msg = ", ".join(msgs)
        return f"<ATB object: {msg}>"

    @property
    def sheets(self):
        """Return list of available sheet names."""
        if not self.table_fpath:
            # Get from website
            raise OSError("File not found...")
        else:
            names = get_sheet(self.table_fpath)
        return names

    @property
    def technologies(self):
        """Return lookup of technology - ATB names."""
        lookup = {
            "wind": "Land-Based Wind",
            "solar": "Solar - Utility PV"
        }
        return lookup

    @property
    def table(self):
        """Read in/download ATB data sheets."""
        sheet = self.technologies[self.tech]
        if not self.table_fpath:
            # Get from website
            raise OSError("File not found...")
        else:
            df = get_sheet(self.table_fpath, sheet_name=sheet)
            df = self._parse_table(df)
        return df

    @property
    def capex(self):
        """Return capex for given tech and year."""

    @property
    def opex(self):
        """Return opex for given tech and year."""
        sheet = self.technologies[self.tech]
        table = get_sheet(self.table_fpath, sheet)

    def _parse_table(self, df):
        # Get the ATB sheet into a useable form
        # Find starting y-position and filter
        iy, _ = np.where(df.values == "Base Year")
        assert len(iy) == 1, "_parse_table failed to parse."
        iy = iy[0] + 1
        df = df.iloc[iy:, :]

        # Find starting x-position and filter
        target_col = "Techno-Economic Cost and Performance Parameters"
        _, ix = np.where(df.values == target_col)
        assert len(ix) == 1, "_parse_table failed to parse."
        ix = ix[0] + 2
        df = df.iloc[:, ix:]

        # Adjust the columns
        years = [str(int(y)) for y in df.iloc[0, 2:]]
        cols = ["variable", "scenario", *years]
        df = df.iloc[1:]
        df.columns = cols

        # Fill in missing variable entries
        variables = []
        variable = None
        for value in df["variable"].values:
            df["variable"]
    
        return df