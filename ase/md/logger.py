"""Logging for molecular dynamics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ase import units
from ase.io.logger import Logger
from ase.parallel import world

if TYPE_CHECKING:
    from pathlib import Path
    from typing import IO, Any, Union

    from ase.md.md import MolecularDynamics


class MDLogger(Logger):
    """
    Convenience class to log MD simulation by adding commonly used fields
    Add the following fields to the logger:

    - Time[ps]: The current simulation time in picoseconds.
    - Etot[eV]: The current total energy.
    - Epot[eV]: The current potential energy.
    - Ekin[eV]: The current kinetic energy.
    - T[K]: The current temperature.

    Parameters
    ----------
    dyn
        The `:class:~ase.md.md.MolecularDynamics` object to track.
    logfile
        File path or open file object for logging.
        Use "-" for standard output.
    mode
        File opening mode if logfile is a filename. Default: "a".
    comm
        MPI communicator for parallel simulations. Default: world.
    """

    def __init__(
        self,
        dyn: MolecularDynamics,
        logfile: Union[str, Path, IO[str]],
        mode: str = 'a',
        comm: Any = world,
    ):
        super().__init__(logfile, mode, comm)

        names = ['Time[ps]', 'Etot[eV]', 'Epot[eV]', 'Ekin[eV]', 'T[K]']
        callables = [
            lambda: dyn.get_time() / (1000 * units.fs),
            dyn.atoms.get_total_energy,
            dyn.atoms.get_potential_energy,
            dyn.atoms.get_kinetic_energy,
            dyn.atoms.get_temperature,
        ]
        formats = ['{:<12.4f}'] + ['{:>12.4f}'] * 3 + ['{:>10.2f}']

        for name, func, fmt in zip(names, callables, formats):
            self.add_field(name, func, fmt)
