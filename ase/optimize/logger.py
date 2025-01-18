"""Logging for molecular dynamics."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from ase.io.logger import Logger
from ase.parallel import world

if TYPE_CHECKING:
    from pathlib import Path
    from typing import IO, Any, Union

    from ase.optimize.optimize import Optimizer


class OptLogger(Logger):
    """
    Convenience class to to log ASE optimizers by adding commonly used fields.
    Add the following fields to the logger:

    - Optimizer: The name of the optimizer class.
    - Step: The current optimization step.
    - Time: The current time in HH:MM:SS format.
    - Epot[eV]: The current potential energy.
    - Fmax[eV/A]: The maximum force component.

    Parameters
    ----------
    opt
        The `:class:~ase.optimize.optimize.Optimizer` object to track.
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
        opt: Optimizer,
        logfile: Union[str, Path, IO[str]],
        mode: str = 'a',
        comm: Any = world,
    ):
        super().__init__(logfile, mode, comm)

        names = ['Optimizer', 'Step', 'Time', 'Epot[eV]', 'Fmax[eV/A]']
        callables = [
            lambda: opt.__class__.__name__,
            lambda: opt.nsteps,
            lambda: '{:02d}:{:02d}:{:02d}'.format(*time.localtime()[3:6]),
            opt.optimizable.get_potential_energy,
            lambda: np.linalg.norm(opt.optimizable.get_forces(), axis=1).max(),
        ]

        formats = ['{:<26s}'] + ['{:>6d}'] + ['{:>12s}'] + ['{:>12.4f}'] * 2

        for name, func, fmt in zip(names, callables, formats):
            self.add_field(name, func, fmt)
