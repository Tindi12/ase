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

    from ase import Atoms
    from ase.md.md import MolecularDynamics
    from ase.optimize.optimize import Optimizer


class Logger(IOContext):
    """
    A general purpose logger for atomistic simulations, if created manually,
    the :meth:`add_field` method must be called to configure the fields to log.

    Callable required for each field should return the value to log. These can
    be easily created with lambda functions that return the desired value.
    For example, to log the current energy of an ASE atoms object, use:

    ``` python
    logger.add_field("Epot[eV]", lambda: atoms.get_potential_energy())
    ```

    The logger can also be configured using convenience methods, such as
    :meth:`add_md_fields` and :meth:`add_opt_fields`, which add commonly used
    fields for molecular dynamics and ASE optimizers, respectively. This will
    be done automatically by the :class:`MolecularDynamics` and
    :class:`Optimizer` classes if `logfile` is passed to their constructors.

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
        :meth:`~Logger.add_field` method.
    """

    def __init__(
        self,
        logfile: Union[IO, str, Path],
        mode: str = 'a',
        comm: Any = world,
    ) -> None:
        """Initialize the molecular dynamics logger."""
        self.fields = {}
        self.logfile = self.openfile(logfile, mode=mode, comm=comm)
        
        self._cache = {}
        self._field_is_list = {}

    def __call__(self) -> None:
        """
        Log the current state of the simulation.

        Writes a new line to the log file containing the current values
        of all configured fields (time, energies, temperature, stress).
        """
        parts = []

        for key in self.fields:
            value = key()

            if key not in self._cache:
                fmt = self.fields[key][1]
                if isinstance(value, (list, tuple, np.ndarray)):
                    self._cache[key] = ' '.join(f'{{:{f}}}' for f in fmt)
                    self._field_is_list[key] = True
                else:
                    self._cache[key] = f'{{:{fmt}}}'
                    self._field_is_list[key] = False

            if self._field_is_list[key]:
                parts.append(self._cache[key].format(*value))
            else:
                parts.append(self._cache[key].format(value))

        self.logfile.write(' '.join(parts) + '\n')
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
        names = [field[0] for field in self.fields.values()]
        formats = [field[1] for field in self.fields.values()]

        to_write = []

        for name, fmt in zip(names, formats):
            if isinstance(fmt, (list, tuple, np.ndarray)) and isinstance(
                name, (list, tuple, np.ndarray)
            ):
                to_write.extend(
                    [
                        f'{n:{get_auto_header_format(f)}}'
                        for n, f in zip(name, fmt)
                    ]
                )
            else:
                to_write.append(f'{name:{get_auto_header_format(fmt)}}')

        return ' '.join(to_write)

    def add_field(
        self,
        name: str,
        function: Callable,
        fmt: str = '10.3f',
    ) -> None:
        """
        Add one field to the logger, which track a value that
        change during the simulation. The callable can return a list of values
        to log multiple values in a single field. In this case, the format
        and name should be lists of the same length.

        Parameters
        ----------
        names
            Name of the field to add.
        callables
            Callable object returning the value of the field.
        formats
            Format string for field value.

        Examples
        --------
        ``` python
        logger.add_field("Epot[eV]", lambda: atoms.get_potential_energy())
        logger.add_field(
            ["Class", "Step"],
            [lambda: simulation.__class__.__name__, lambda: simulation.nsteps],
            [">12s", ">12d"],
        )
        ```
        """
        self.fields[function] = (name, fmt)

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
            The :class:~ase.md.md.MolecularDynamics` object.
        """
        names = ['Time[ps]', 'Etot[eV]', 'Epot[eV]', 'Ekin[eV]', 'T[K]']
        callables = [
            lambda: dyn.get_time() / (1000 * units.fs),
            dyn.atoms.get_total_energy,
            dyn.atoms.get_potential_energy,
            dyn.atoms.get_kinetic_energy,
            dyn.atoms.get_temperature,
        ]
        formats = ['<12.4f'] + ['>12.4f'] * 3 + ['>10.2f']

        for name, func, fmt in zip(names, callables, formats):
            self.add_field(name, func, fmt)

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
            opt.optimizable.get_potential_energy,
            lambda: np.linalg.norm(opt.optimizable.get_forces(), axis=1).max(),
        ]

        formats = ['<20s'] + ['>6d'] + ['>12s'] + ['>12.4f'] * 2

        for name, func, fmt in zip(names, callables, formats):
            self.add_field(name, func, fmt)

    def add_stress_fields(
        self,
        atoms: Atoms,
        include_ideal_gas: bool = True,
        mask: list[bool] = None,
    ) -> None:
        """
        Add the stress fields to the logger.

        Parameters
        ----------
        atoms
            The ASE atoms object.
        include_ideal_gas
            Whether to include the ideal gas contribution to the stress.
        """
        if mask is None:
            mask = [True] * 6

        def log_stress():
            stress = atoms.get_stress(include_ideal_gas=include_ideal_gas)
            stress = tuple(stress / units.GPa)
            return np.array([s for n, s in enumerate(stress) if mask[n]])

        components = ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']

        names = [
            f'{component}Stress[GPa]'
            for n, component in enumerate(components)
            if mask[n]
        ]

        format_ = ['>14.3f'] * sum(mask)

        self.add_field(names, log_stress, format_)

    def remove_fields(self, name: str) -> None:
        """
        Remove one or multiple field(s) from the logger. Work by finding
        partial matches of the field name(s) in the current fields. List fields
        count as a single field, i.e., if a match is found in a list field, the
        whole list field is removed

        Parameters
        ----------
        name
            Name of the field to remove.
        """
        for func, (field_name, _) in list(self.fields.items()):
            if name in field_name:
                self.fields.pop(func, None)

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
