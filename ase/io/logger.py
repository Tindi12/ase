"""General purpose logger for atomistic simulations."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from ase import units
from ase.parallel import world
from ase.utils import IOContext

if TYPE_CHECKING:
    from pathlib import Path
    from typing import IO, Any, Callable, Union

    from ase import Atoms


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
        self.fields: dict[str | tuple[str], dict] = {}
        self.logfile = self.openfile(logfile, mode=mode, comm=comm)

    def __call__(self) -> None:
        """
        Log the current state of the simulation.

        Writes a new line to the log file containing the current values
        of all configured fields (time, energies, temperature, stress).
        """
        parts = []

        for key in self.fields:
            value = self.fields[key]['function']()

            if self.fields[key]['is_list']:
                parts.append(self.fields[key]['fmt'].format(*value))
            else:
                parts.append(self.fields[key]['fmt'].format(value))

        self.logfile.write(' '.join(parts) + '\n')
        self.logfile.flush()

    def __del__(self) -> None:
        """Clean up by closing the log file."""
        self.close()

    def create_header(self) -> str:
        """
        Create the header format string based on configured options.

        Returns
        -------
        str
            Formatted header string.
        """
        to_write = []

        for name in self.fields:
            header_fmt = self.fields[name]['header_fmt']
            if self.fields[name]['is_list']:
                to_write.append(header_fmt.format(*name))
            else:
                to_write.append(header_fmt.format(name))

        return ' '.join(to_write)

    def add_field(
        self,
        name: str | list[str] | tuple[str],
        function: Callable,
        fmt: str = '{:10.3f}',
        header_fmt: str | None = None,
        is_list: bool = False,
    ) -> None:
        """
        Add one field to the logger, which track a value that
        change during the simulation. The callable can return a list of values
        to log multiple values in a single field. In this case, the format
        and name should be lists of the same length.

        Parameters
        ----------
        name
            Name of the field to add.
        function
            Callable object returning the value of the field.
        fmt
            Format string for field value.
        is_list
            Whether the field's function returns a list of values.

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

        Notes
        -----
        If the field is a list, the format string should be a list of the same
        length as the name. The format string should be a single format string
        which length cannot change during the simulation.
        """
        if isinstance(name, list):
            name = tuple(name)

        if header_fmt is None:
            header_fmt = get_auto_header_format(fmt)

        self.fields[name] = {
            'function': function,
            'fmt': fmt,
            'header_fmt': header_fmt,
            'is_list': is_list,
        }

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
        mask
            A list of booleans to mask the stress components to log.
            The default is to log all components.
        """
        if mask is None:
            mask = [True] * 6

        def log_stress():
            stress = atoms.get_stress(include_ideal_gas=include_ideal_gas)
            stress = tuple(stress / units.GPa)
            return np.array([s for n, s in enumerate(stress) if mask[n]])

        components = ['xx', 'yy', 'zz', 'yz', 'xz', 'xy']

        names = [
            f'Stress[{component}][GPa]'
            for n, component in enumerate(components)
            if mask[n]
        ]

        formats = '{:>18.3f}' * sum(mask)

        self.add_field(names, log_stress, formats, is_list=True)

    def remove_fields(self, pattern: str) -> None:
        """
        Remove fields whose names contain the given pattern.

        Parameters
        ----------
        pattern : str
            Pattern to match in field names. For compound fields
            (tuple of names), matches if any component contains the pattern.
        """
        for field_name in list(self.fields.keys()):
            if isinstance(field_name, tuple):
                if any(pattern in name for name in field_name):
                    self.fields.pop(field_name)
            else:
                if pattern in field_name:
                    self.fields.pop(field_name)

    def write_header(self) -> None:
        """Write the header line to the log file."""
        self.logfile.write(f'{self.create_header()}\n')


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
    return re.sub(
        r':([<>^])?(\d+)?[^}]*',
        lambda m: f':{m.group(1) or ">"}{m.group(2) or "10"}s',
        data_format,
    )
