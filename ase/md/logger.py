"""Logging for molecular dynamics."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import ase.units as units
from ase.parallel import world
from ase.utils import IOContext

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, IO, Union

    from ase.atoms import Atoms
    from ase.optimize.optimize import Dynamics


class MDLogger(IOContext):
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
        dyn: Dynamics,
        atoms: Atoms,
        logfile: Union[IO, str, Path],
        stress: bool = False,
        peratom: bool = False,
        mode: str = 'a',
        header: bool = True,
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
        self.dyn = weakref.proxy(dyn) if hasattr(dyn, 'get_time') else None

        self.atoms = atoms
        self.logfile = self.openfile(logfile, mode=mode, comm=comm)
        self.stress = stress
        self.peratom = peratom

        self.has_extra_fields = (
            True if hasattr(self.dyn, 'extra_fields') else False
        )

        self.header_format = self._create_header_format()
        self.data_format = self._create_data_format()

        if header:
            self.logfile.write(f'{self.header_format}\n')

    def _create_header_format(self) -> str:
        """
        Create the header format string based on configured options.

        Returns
        -------
        str
            Formatted header string.
        """
        header_parts = []

        # Time column if dynamics is available
        if self.dyn is not None:
            header_parts.append('%-9s ' % 'Time[ps]')

        # Energy columns
        energy_suffix = '/N' if self.peratom else ''
        header_parts.extend(
            [
                f"{f'Etot{energy_suffix}[eV]':>12}",
                f"{f'Epot{energy_suffix}[eV]':>12}",
                f"{f'Ekin{energy_suffix}[eV]':>12}",
                f"{'T[K]':>8}",
            ]
        )

        if self.has_extra_fields:
            header_parts.extend(
                [f"{f'{field}':>10}" for field in self.dyn.extra_fields]
            )

        # Stress columns if enabled
        if self.stress:
            header_parts.append(
                '      ----------------------'
                ' stress[GPa] -----------------------'
            )

        return ' '.join(header_parts)

    def _create_data_format(self) -> str:
        """
        Create the data format string based on configured options.

        Returns
        -------
        str
            Format string for data entries.
        """
        format_parts = []

        # Time format if dynamics is available
        if self.dyn is not None:
            format_parts.append('%-10.4f')

        natoms = self.atoms.get_global_number_of_atoms()

        # Energy format
        digits = 4
        if not self.peratom:
            if natoms > 100:
                digits = 3
            if natoms > 1000:
                digits = 2
            if natoms > 10000:
                digits = 1

        # Energy and temperature columns
        format_parts.extend(
            [
                f'%12.{digits}f',  # Etot
                f'%12.{digits}f',  # Epot
                f'%12.{digits}f',  # Ekin
                '%8.1f',  # Temperature
            ]
        )

        if self.has_extra_fields:
            format_parts.extend(['%10.3f'] * len(self.dyn.extra_fields))

        # Stress format if enabled
        if self.stress:
            format_parts.extend(['%10.3f'] * 6)

        format_parts.append('\n')
        return ' '.join(format_parts)

    def __call__(self) -> None:
        """
        Log the current state of the simulation.

        Writes a new line to the log file containing the current values
        of all configured fields (time, energies, temperature, stress).
        """
        epot = self.atoms.get_potential_energy()
        ekin = self.atoms.get_kinetic_energy()
        temp = self.atoms.get_temperature()

        natoms = self.atoms.get_global_number_of_atoms()

        if self.peratom:
            epot /= natoms
            ekin /= natoms

        log_data = []

        # Add time if dynamics is available
        if self.dyn is not None:
            time_ps = self.dyn.get_time() / (1000 * units.fs)
            log_data.append(time_ps)

        # Add energy and temperature data
        to_write = [epot + ekin, epot, ekin, temp]

        if self.has_extra_fields:
            to_write.extend(self.dyn.extra_fields.values())

        log_data.extend(to_write)

        # Add stress data if enabled
        if self.stress:
            stress_values = (
                self.atoms.get_stress(include_ideal_gas=True) / units.GPa
            )
            log_data.extend(stress_values)

        self.logfile.write(self.data_format % tuple(log_data))
        self.logfile.flush()

    def __del__(self) -> None:
        """Clean up by closing the log file."""
        self.close()
