"""General purpose logger for atomistic simulations."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from ase import units
from ase.parallel import world
from ase.utils import IOContext

if TYPE_CHECKING:
    from pathlib import Path
    from typing import IO, Any, Callable, Union

    from ase.md.md import MolecularDynamics
    from ase.optimize.optimize import Optimizer


class Logger(IOContext):
    """
    A general purpose logger for atomistic simulations, if created manually,
    the :meth:`add_fields` method must be called to configure the fields to log.

    Callable required for each field should return the value to log. These can
    be easily created with lambda functions that return the desired value.
    For example, to log the current energy of an ASE atoms object, use:

    ``` python
    logger.add_fields("Epot[eV]", lambda: atoms.get_potential_energy())
    ```

    The logger can also be configured using convenience methods, such as
    :meth:`add_md_fields` and :meth:`add_opt_fields`, which add commonly used
    fields for molecular dynamics and ASE optimizers, respectively. This will
    be done automatically by the :class:`MolecularDynamics` and
    :class:`Optimizer` classes.

    Parameters
    ----------
    logfile
        File path or open file object for logging.
        Use "-" for standard output.
    mode
        File opening mode if logfile is a filename. Default: "a".
    comm
        MPI communicator for parallel simulations. Default: world.

    Attributes
    ----------
    logfile
        The opened log file object.
    fields
        Dictionary of fields to log. Fields can be added with the
        :meth:`~Logger.add_fields` method.
    names
        List of field names.
    formats
        List of field formats.
    """

    def __init__(
        self,
        logfile: Union[IO, str, Path],
        mode: str = 'a',
        comm: Any = world,
    ) -> None:
        """Initialize the molecular dynamics logger."""
        self.fields = {}
        self.names, self.formats = [], []
        self.logfile = self.openfile(logfile, mode=mode, comm=comm)

    def __call__(self) -> None:
        """
        Log the current state of the simulation.

        Writes a new line to the log file containing the current values
        of all configured fields (time, energies, temperature, stress).
        """
        values = [field[0]() for field in self.fields.values()]

        self.logfile.write(
            ' '.join(f'{val:{fmt}}' for val, fmt in zip(values, self.formats))
            + '\n'
        )
        self.logfile.flush()

    def __del__(self) -> None:
        """Clean up by closing the log file."""
        self.close()

    def _create_header_format(self) -> str:
        """
        Create the header format string based on configured options.

        Returns
        -------
        str
            Formatted header string.
        """
        return ' '.join(
            f'{name:{fmt}}'
            for name, fmt in zip(self.names, self.string_formats)
        )

    def add_fields(
        self, names: str, callables: Callable, formats: str = '10.3f'
    ) -> None:
        """
        Add one or multiple field(s) to the logger, which track value(s) that
        change during the simulation.

        Parameters
        ----------
        names
            Name of the field(s) to add.
        callables
            Callable object(s) returning the value of the field(s).
        formats
            Format string(s) for field value(s).

        Examples
        --------
        ``` python
        logger.add_fields("Epot[eV]", lambda: atoms.get_potential_energy())
        logger.add_fields(
            ["Class", "Step"],
            [lambda: simulation.__class__.__name__, lambda: simulation.nsteps],
            [">12s", ">12d"],
        )
        ```
        """
        if type(names) is not list:
            names = [names]
        if type(callables) is not list:
            callables = [callables]
        if type(formats) is not list:
            formats = [formats]

        for name, f, fmt in zip(names, callables, formats):
            self.fields[name] = (f, fmt)

        self.names = list(self.fields.keys())
        self.formats = [field[1] for field in self.fields.values()]
        self.string_formats = [
            get_auto_header_format(fmt) for fmt in self.formats
        ]

    def add_md_fields(self, dyn: MolecularDynamics) -> None:
        """
        Convenience function to add commonly used fields for MD simulations.
        Add the following fields to the logger:

        - Time[ps]: The current simulation time in picoseconds.
        - Etot[eV]: The current total energy.
        - Epot[eV]: The current potential energy.
        - Ekin[eV]: The current kinetic energy.
        - T[K]: The current temperature.

        Parameters
        ----------
        dyn
            The `MolecularDynamics` object.
        """
        names = ['Time[ps]', 'Etot[eV]', 'Epot[eV]', 'Ekin[eV]', 'T[K]']
        callables = [
            lambda: dyn.nsteps / (1000 * units.fs),
            lambda: dyn.atoms.get_total_energy(),
            lambda: dyn.atoms.get_potential_energy(),
            lambda: dyn.atoms.get_kinetic_energy(),
            lambda: dyn.atoms.get_temperature(),
        ]
        formats = ['<12.4f'] + ['>12.4f'] * 3 + ['>10.2f']

        self.add_fields(names, callables, formats)

    def add_opt_fields(self, opt: Optimizer) -> None:
        """
        Convenience function to add commonly used fields for ASE optimizers.
        Add the following fields to the logger:

        - Optimizer: The name of the optimizer class.
        - Step: The current optimization step.
        - Time: The current time in HH:MM:SS format.
        - Epot[eV]: The current potential energy.
        - Fmax[eV/A]: The maximum force component.

        Parameters
        ----------
        optimizer
            The ASE optimizer object.
        """
        names = ['Optimizer', 'Step', 'Time', 'Epot[eV]', 'Fmax[eV/A]']
        callables = [
            lambda: opt.__class__.__name__,
            lambda: opt.nsteps,
            lambda: '{:02d}:{:02d}:{:02d}'.format(*time.localtime()[3:6]),
            lambda: opt.optimizable.get_potential_energy(),
            lambda: np.linalg.norm(opt.optimizable.get_forces(), axis=1).max(),
        ]

        formats = ['<20s'] + ['>6d'] + ['>12s'] + ['>12.4f'] * 2
        self.add_fields(names, callables, formats)

    def write_header(self) -> None:
        """Write the header line to the log file."""
        self.logfile.write(f'{self._create_header_format()}\n')


def get_auto_header_format(data_format: str) -> str:
    """
    Convert a data format string to a header format string.

    Parameters
    ----------
    data_format
        The data format string to convert.

    Returns
    -------
    str
        The converted header format string to maintain alignment.
        Returns '>10s' if the function fails to parse the data format.

    Examples
    --------
    get_header_format('10.3f') -> '>10s'
    get_header_format('4s') -> '>4s'
    """
    data_format = data_format.lstrip(':')

    align = '>' if data_format[0] not in '<>^' else data_format[0]

    width = ''
    for char in data_format.lstrip('<>^'):
        if char.isdigit():
            width += char
        else:
            break

    return f'{align}{width}s' if width else '>10s'
