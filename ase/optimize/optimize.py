"""Structure optimization."""

import warnings
from os.path import isfile
from typing import IO, Optional, Union

from ase import Atoms
from ase.calculators.calculator import PropertyNotImplementedError
from ase.filters import UnitCellFilter
from ase.logger import Logger
from ase.runner import Runner
from ase.utils import lazyproperty
from ase.utils.abc import Optimizable


class RestartError(RuntimeError):
    pass


class OptimizableAtoms(Optimizable):
    def __init__(self, atoms):
        self.atoms = atoms

    def get_positions(self):
        return self.atoms.get_positions()

    def set_positions(self, positions):
        self.atoms.set_positions(positions)

    def get_forces(self):
        return self.atoms.get_forces()

    @lazyproperty
    def _use_force_consistent_energy(self):
        # This boolean is in principle invalidated if the
        # calculator changes.  This can lead to weird things
        # in multi-step optimizations.
        try:
            self.atoms.get_potential_energy(force_consistent=True)
        except PropertyNotImplementedError:
            # warnings.warn(
            #     'Could not get force consistent energy (\'free_energy\').  '
            #     'Please make sure calculator provides \'free_energy\', even '
            #     'if equal to the ordinary energy.  '
            #     'This will raise an error in future versions of ASE.',
            #     FutureWarning)
            return False
        else:
            return True

    def get_potential_energy(self):
        force_consistent = self._use_force_consistent_energy
        return self.atoms.get_potential_energy(
            force_consistent=force_consistent
        )

    def iterimages(self):
        # XXX document purpose of iterimages
        return self.atoms.iterimages()

    def __len__(self):
        # TODO: return 3 * len(self.atoms), because we want the length
        # of this to be the number of DOFs
        return len(self.atoms)


class Optimizer(Runner):
    """Base-class for all structure optimization classes."""

    # default maxstep for all optimizers
    defaults = {'maxstep': 0.2}
    _deprecated = object()

    def __init__(
        self,
        atoms: Atoms,
        restart: Optional[str] = None,
        logfile: Optional[Union[IO, str]] = None,
        trajectory: Optional[str] = None,
        append_trajectory: bool = False,
        **kwargs,
    ):
        """

        Parameters
        ----------
        atoms: :class:`~ase.Atoms`
            The Atoms object to relax.

        restart: str
            Filename for restart file. Default value is *None*.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        trajectory: Trajectory object or str
            Attach trajectory object. If *trajectory* is a string a
            Trajectory will be constructed. Use *None* for no
            trajectory.

        append_trajectory: bool
            Appended to the trajectory file instead of overwriting it.

        kwargs : dict, optional
            Extra arguments passed to :class:`~ase.optimize.optimize.Dynamics`.

        """
        super().__init__(
            atoms=atoms,
            trajectory=trajectory,
            append_trajectory=append_trajectory,
            **kwargs,
        )

        self.restart = restart

        self.fmax = None

        if logfile:
            self.logger = Logger(logfile)
            self.logger.add_opt_fields(self)
            self.attach(self.closelater(self.logger), 1)
        else:
            self.logger = None

        if restart is None or not isfile(restart):
            self.initialize()
        else:
            self.read()
            self.comm.barrier()

    def read(self):
        raise NotImplementedError

    def todict(self):
        description = {
            'type': 'optimization',
            'optimizer': self.__class__.__name__,
        }
        # add custom attributes from subclasses
        for attr in ('maxstep', 'alpha', 'max_steps', 'restart', 'fmax'):
            if hasattr(self, attr):
                description.update({attr: getattr(self, attr)})
        return description

    def initialize(self):
        pass

    def irun(self, fmax=0.05, steps=Runner.DEFAULT_MAX_STEPS):
        """Run optimizer as generator.

        Parameters
        ----------
        fmax : float
            Convergence criterion of the forces on atoms.
        steps : int, default=DEFAULT_MAX_STEPS
            Number of optimizer steps to be run.

        Yields
        ------
        converged : bool
            True if the forces on atoms are converged.
        """
        self.fmax = fmax

        if self.logger:
            self.logger.write_header()

        return Runner.irun(self, steps=steps)

    def run(self, fmax=0.05, steps=Runner.DEFAULT_MAX_STEPS):
        """Run optimizer.

        Parameters
        ----------
        fmax : float
            Convergence criterion of the forces on atoms.
        steps : int, default=DEFAULT_MAX_STEPS
            Number of optimizer steps to be run.

        Returns
        -------
        converged : bool
            True if the forces on atoms are converged.
        """
        self.fmax = fmax

        if self.logger:
            self.logger.write_header()

        return Runner.run(self, steps=steps)

    def converged(self, forces=None):
        """Did the optimization converge?"""
        if forces is None:
            forces = self.optimizable.get_forces()
        return self.optimizable.converged(forces, self.fmax)

    def dump(self, data):
        from ase.io.jsonio import write_json

        if self.comm.rank == 0 and self.restart is not None:
            with open(self.restart, 'w') as fd:
                write_json(fd, data)

    def load(self):
        from ase.io.jsonio import read_json

        with open(self.restart) as fd:
            try:
                from ase.optimize import BFGS

                if not isinstance(self, BFGS) and isinstance(
                    self.atoms, UnitCellFilter
                ):
                    warnings.warn(
                        'WARNING: restart function is untested and may result '
                        'in unintended behavior. Namely orig_cell is not '
                        'loaded in the UnitCellFilter. Please test on your own'
                        ' to ensure consistent results.'
                    )
                return read_json(fd, always_array=False)
            except Exception as ex:
                msg = (
                    'Could not decode restart file as JSON.  '
                    'You may need to delete the restart file '
                    f'{self.restart}'
                )
                raise RestartError(msg) from ex
