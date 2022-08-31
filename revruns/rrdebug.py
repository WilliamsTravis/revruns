# -*- coding: utf-8 -*-
"""Debug CLI command.

This could turn into a more significant convenience wrapper for CliRunner. I'd
like to be able to quickly enter a debug session using a cli command with
pointers to configs. This would need to be used carefully, since some errors
require enormous memory and run times.

In this case I'm trying to recreate a transmission error on a chunked bespoke
hdf5 file.

Author: twillia2
Date: Fri Aug 19 13:21:39 MDT 2022
"""
import json

from pathlib import Path

from click.testing import CliRunner
from click.core import Context

import reV


CONFIG = ("/shared-projects/rev/projects/naris/fy22/rev/canada_wind/config_supply-curve.json")
CMD = f"reV -c {CONFIG} supply-curve"


class RRDebug:
    """Methods for setting up and debugging sample reV runs from a CLI."""

    def __init__(self, cmd):
        """Initialize RRDebug object.
        
        Parameters
        ----------
        cmd : str
            A string representation of a reV cli command. (e.g. "reV -c 
            config_pipeline.json pipeline")
        """
        self.cmd = cmd
        self.setup()

    def __repr__(self):
        """Return reprentation string for RRDebug object."""
        name = self.__class__.__name__
        attrs = [f"{k}='{v}'" for k, v in self.__dict__.items()]
        return f"<{name} object: {', '.join(attrs)}>"

    @property
    def args(self):
        """Return arguments for cli command."""
        return self.cmd.split()[1:]

    @property
    def base_cli(self):
        """Return the appropriate reV click object for a given cli command."""
        module = self.cmd.split()[0]
        if module == "reV":
            from reV.cli import main
            return main
        else:
            print(f"I haven't found the {module} python object yet.")

    def setup(self):
        """Setup debug directory, configs, and commands."""
        # Build paths and make debug directory
        run_dir = self.config_path.parent
        self.dbg_dir = run_dir.joinpath("debug")
        self.dbg_dir.mkdir(exist_ok=True)

        # Adjust the config to run locally
        dbg_config = self.config
        dbg_config["execution_control"] = {"option": "local"}
        self.dbg_config_path = self.dbg_dir.joinpath(self.config_path.name)
        self.dbg_cmd = self.cmd.replace(
            str(self.config_path),
            str(self.dbg_config_path)
        )

        # Check for large inputs, perhaps check logs from full run

        # Adjust for large inputs with sample...possibly derived from logs

        # Write config
        with open(self.dbg_config_path, "w") as file:
            file.write(json.dumps(dbg_config, indent=4))

    @property
    def cli(self):
        """Return the appropriate reV click object for a given cli command."""
        # Find the command and config
        module = self.cmd.split()[-1]
        module = module.replace("-", "_")
        return reV.__dict__[module].__dict__[f"cli_{module}"].from_config

    @property
    def config(self):
        """Return the config dictionary for a command."""
        return json.load(open(self.config_path))

    @property
    def config_path(self):
        """Return the config path from a command."""
        return Path(self.cmd.split("-c ")[1:][0].split()[0])

    def run(self):
        """Run a CLI command in a Python session.

        Parameters
        ----------
        cmd : str
            A string representation of a reV cli command line interface call.
        """
        # Build a debug direcotry
        self.build_dir()

        # Build runner object
        runner = CliRunner()

        # Using base command
        cli = self.base_cli
        args = cmd.split()[1:]
        out = runner.invoke(cli, args=args)

        # This need to get past submission, raise error
        assert out.exit_code == 0, f"{cmd} failed"

    def write_script(self):
        """Write an implementation of `RRDebug.run` to a python file."""
        py_path = self.dbg_dir.joinpath("debug.py")
        lines = self._header + self._imports + self._body + self._call
        lines = [line + "\n" for line in lines]
        with open(py_path, "w") as file:
            file.writelines(lines)

    @property
    def _body(self):
        """Return body for script."""
        body = [
            "",
            "",
            "def main():",
            "    # Build runner and debug objects",
            "    runner = CliRunner()",
           f"    rrdebugger = RRDebug('{self.dbg_cmd}')",
            "", 
            "    # Using base command",
            "    cli = rrdebugger.base_cli",
            "    args = rrdebugger.args",
            "    out = runner.invoke(self.base_cli, args=args)",
            "",
            "    # This need to get past submission, raise error",
           f"    assert out.exit_code == 0, '{self.dbg_cmd} failed.'"
        ]
        return body

    @property
    def _call(self):
        """Return body for script."""
        call = [
            "",
            "",
            "if __name__ == '__main__':",
            "    main()",
        ]
        return call

    @property
    def _header(self):
        """Return header for script."""
        header = [
            "# -*- coding: utf-8 -*-",
            '"""RRDebug Script.',
            "",
            "Use this script with a debugger and break points to catch reV.",
            "errors from the following CLI command:",
            "",
            f"`{self.cmd}`",
            '"""',
        ]
        return header

    @property
    def _imports(self):
        """Return body for script."""
        imports = [
            "from click.testing import CliRunner",
            "from click.core import Context",
            "",
            "import reV"
        ]
        return imports

def main():
    """Run rrdebug."""
    # Build debug directory
    rrdebug(CMD)


if __name__ == "__main__":
    cmd = CMD
    self = RRDebug(cmd)
    self.write_script()
