"""Logging for molecular dynamics."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import ase.units as units
from ase.parallel import world
from ase.utils import IOContext

if TYPE_CHECKING:
    from pathlib import Path

    from ase.atoms import Atoms
    from ase.optimize.optimize import Dynamics


class MDLogger(IOContext):
    r"""
    A logger for molecular dynamics simulations
    that tracks energy and temperature.

    Attributes
    ----------
    dynamics
        Reference to the dynamics object.
    atoms
        Reference to the atomic system.
    logfile
        The opened log file object.
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
        dynamics: Dynamics,
        atoms: Atoms,
        logfile: str | Path,
        log_stress: bool = False,
        energy_per_atom: bool = False,
        file_mode: str = 'a',
        include_header: bool = True,
    ) -> None:
        """
        Initialize the molecular dynamics logger.

        Parameters
        ----------
        dynamics
            The dynamics object containing simulation state.
            Only a weak reference is maintained.
        atoms
            The atomic system being simulated.
        logfile
            File path or open file object for logging.
            Use "-" for standard output.
        log_stress
            Whether to include stress calculations in the log. Default: False.
        energy_per_atom
            Whether to normalize energies by number of atoms. Default: False.
        file_mode
            File opening mode if logfile is a filename. Default: "a".
        include_header
            Whether to write the header line in the log file. Default: True.
        """
        self.dynamics = (
            weakref.proxy(dynamics) if hasattr(dynamics, 'get_time') else None
        )

        self.atoms = atoms
        self.total_atoms = atoms.get_global_number_of_atoms()
        self.logfile = self.openfile(logfile, comm=world, mode=file_mode)
        self.log_stress = log_stress
        self.energy_per_atom = energy_per_atom

        self.has_extra_fields = (
            True if hasattr(self.dynamics, 'extra_fields') else False
        )

        self.header_format = self._create_header_format()
        self.data_format = self._create_data_format()

        if include_header:
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
        if self.dynamics is not None:
            header_parts.append('%-9s ' % 'Time[ps]')

        # Energy columns
        energy_suffix = '/N' if self.energy_per_atom else ''
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
                [f"{f'{field}':>10}" for field in self.dynamics.extra_fields]
            )

        # Stress columns if enabled
        if self.log_stress:
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
        if self.dynamics is not None:
            format_parts.append('%-10.4f')

        # Energy format
        digits = 4
        if not self.energy_per_atom:
            if self.total_atoms > 100:
                digits = 3
            if self.total_atoms > 1000:
                digits = 2
            if self.total_atoms > 10000:
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
            format_parts.extend(['%10.3f'] * len(self.dynamics.extra_fields))

        # Stress format if enabled
        if self.log_stress:
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

        if self.energy_per_atom:
            epot /= self.total_atoms
            ekin /= self.total_atoms

        log_data = []

        # Add time if dynamics is available
        if self.dynamics is not None:
            time_ps = self.dynamics.get_time() / (1000 * units.fs)
            log_data.append(time_ps)

        # Add energy and temperature data
        to_write = [epot + ekin, epot, ekin, temp]

        if self.has_extra_fields:
            to_write.extend(self.dynamics.extra_fields.values())

        log_data.extend(to_write)

        # Add stress data if enabled
        if self.log_stress:
            stress_values = (
                self.atoms.get_stress(include_ideal_gas=True) / units.GPa
            )
            log_data.extend(stress_values)

        self.logfile.write(self.data_format % tuple(log_data))
        self.logfile.flush()

    def __del__(self) -> None:
        """Clean up by closing the log file."""
        self.close()
