"""Logging for molecular dynamics."""

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
    r"""
    A logger for molecular dynamics simulations
    that tracks energy and temperature.

    Attributes
    ----------
    dyn
        Reference to the dynamics object.
    atoms
        Reference to the atomic system.
    logfile
        The opened log file object.
    stress
        Whether to include stress calculations in the log.
    peratom
        Whether to normalize energies by number of atoms.
    has_extra_fields
        Whether the dynamics object has extra fields.
    header_format
        Format string for the log header.
    data_format
        Format string for the log data entries.

    Notes
    -----
    The precision of energy values in the output is automatically adjusted
    based on the total number of atoms in the system:
    - ≤ 100 atoms: 4 decimal places
    - ≤ 1000 atoms: 3 decimal places
    - ≤ 10000 atoms: 2 decimal places
    - \> 10000 atoms: 1 decimal place
    """

    def __init__(
        self,
        logfile: Union[IO, str, Path],
        mode: str = 'a',
        comm: Any = world,
    ) -> None:
        """
        Initialize the molecular dynamics logger.

        Parameters
        ----------
        dyn
            The dynamics object containing simulation state.
            Only a weak reference is maintained.
        atoms
            The atomic system being simulated.
        logfile
            File path or open file object for logging.
            Use "-" for standard output.
        stress
            Whether to include stress calculations in the log. Default: False.
        peratom
            Whether to normalize energies by number of atoms. Default: False.
        mode
            File opening mode if logfile is a filename. Default: "a".
        header
            Whether to write the header line in the log file. Default: True.
        comm
            MPI communicator for parallel simulations. Default: world.
        """

        self.logfile = self.openfile(logfile, mode=mode, comm=comm)
        self.fields = {}

        self.names = []
        self.formats = []

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

    def add_fields(
        self, names: str, callables: Callable, formats: str = '10.3f'
    ) -> None:
        """
        Add a custom field to the logger.

        Parameters
        ----------
        names
            Name of the field to add.
        callables
            Callable object that returns the value of the field.
        formats
            Format string for the field value. Default: "10.3f".
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
            '>' + fmt.split('.')[0] + 's' for fmt in self.formats
        ]

    def add_md_fields(self, dyn: MolecularDynamics) -> None:
        names = ['Time[ps]', 'Etot[eV]', 'Epot[eV]', 'Ekin[eV]', 'T[K]']
        callables = [
            lambda: dyn.nsteps / (1000 * units.fs),
            lambda: dyn.atoms.get_total_energy(),
            lambda: dyn.atoms.get_potential_energy(),
            lambda: dyn.atoms.get_kinetic_energy(),
            lambda: dyn.atoms.get_temperature(),
        ]
        formats = ['12.4f'] * 4 + ['10.2f']

        self.add_fields(names, callables, formats)

    def add_opt_fields(self, opt: Optimizer) -> None:
        names = ['Optimizer', 'Step', 'Time', 'Epot[eV]', 'Fmax[eV/A]']
        callables = [
            lambda: opt.__class__.__name__,
            lambda: opt.nsteps,
            lambda: time.localtime(),
            lambda: opt.optimizable.get_potential_energy(),
            lambda: np.linalg.norm(opt.optimizable.get_forces(), axis=1).max(),
        ]
        formats = ['10.3f'] * 5
        self.add_fields(names, callables, formats)

    def write_header(self) -> None:
        """Write the header line to the log file."""
        self.logfile.write(f'{self._create_header_format()}\n')
