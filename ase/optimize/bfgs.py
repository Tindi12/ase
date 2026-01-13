# fmt: off

import warnings
from pathlib import Path
from typing import IO

import numpy as np

from ase import Atoms
from ase._4.optimize.bfgs import BFGSMethod
from ase.optimize.optimize import Optimizer, UnitCellFilter


class BFGS(Optimizer):
    # default parameters
    defaults = {**Optimizer.defaults, 'alpha': 70.0}

    def __init__(
        self,
        atoms: Atoms,
        restart: str | Path | None = None,
        logfile: IO | str | Path | None = '-',
        trajectory: str | Path | None = None,
        append_trajectory: bool = False,
        maxstep: float | None = None,
        alpha: float | None = None,
        **kwargs,
    ):
        """BFGS optimizer.

        Parameters
        ----------
        atoms: :class:`~ase.Atoms`
            The Atoms object to relax.

        restart: str | Path | None
            JSON file used to store hessian matrix. If set, file with
            such a name will be searched and hessian matrix stored will
            be used, if the file exists.

        trajectory: str or Path
            Trajectory file used to store optimisation path.

        logfile: file object, Path, or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        maxstep: float
            Used to set the maximum distance an atom can move per
            iteration (default value is 0.2 Å).

        alpha: float
            Initial guess for the Hessian (curvature of energy surface). A
            conservative value of 70.0 is the default, but number of needed
            steps to converge might be less if a lower value is used. However,
            a lower value also means risk of instability.

        kwargs : dict, optional
            Extra arguments passed to
            :class:`~ase.optimize.optimize.Optimizer`.

        """
        if maxstep is None:
            self.maxstep = self.defaults['maxstep']
        else:
            self.maxstep = maxstep

        if self.maxstep > 1.0:
            warnings.warn('You are using a *very* large value for '
                          'the maximum step size: %.1f Å' % self.maxstep)

        self.alpha = alpha
        if self.alpha is None:
            self.alpha = self.defaults['alpha']

        self.state = None

        super().__init__(
            atoms=atoms, restart=restart,
            logfile=logfile, trajectory=trajectory,
            append_trajectory=append_trajectory,
            **kwargs)

    def initialize(self):
        # initial hessian
        self.H0 = np.eye(self.optimizable.ndofs()) * self.alpha
        self.state = None

        self.pos0 = None
        self.forces0 = None

    @property
    def H(self):
        return self.state.hessian

    def read(self):
        data = self.load()
        H, self.pos0, self.forces0, self.maxstep = data[:4]
        if len(data) == 5:
            self.atoms.orig_cell = data[4]
        else:
            assert len(data) == 4
        self.state = BFGSMethod(H)

    def step(self, gradient=None):
        gradient = self._get_gradient(gradient)
        optimizable = self.optimizable

        pos = optimizable.get_x()
        dpos, steplengths = self.prepare_step(pos, gradient)
        dpos = self.determine_step(dpos, steplengths)
        optimizable.set_x(pos + dpos)
        if isinstance(self.atoms, UnitCellFilter):
            self.dump((self.state.hessian, self.pos0, self.forces0,
                       self.maxstep, self.atoms.orig_cell))
        else:
            self.dump((self.state.hessian, self.pos0, self.forces0,
                       self.maxstep))

    def prepare_step(self, pos, gradient):
        pos = pos.ravel()
        gradient = gradient.ravel()
        self.update(pos, -gradient, self.pos0, self.forces0)

        # XXX Here we are calling gradient_norm() on some positions.
        # Should there be a general norm concept
        self.pos0 = pos
        self.forces0 = -gradient.copy()
        dpos = self.state.compute_step(gradient)
        steplengths = self.optimizable.gradient_norm(dpos)
        return dpos, steplengths

    def determine_step(self, dpos, steplengths):
        """Determine step to take according to maxstep

        Normalize all steps as the largest step. This way
        we still move along the direction.
        """
        maxsteplength = np.max(steplengths)
        if maxsteplength >= self.maxstep:
            scale = self.maxstep / maxsteplength
            # FUTURE: Log this properly
            # msg = '\n** scale step by {:.3f} to be shorter than {}'.format(
            #     scale, self.maxstep
            # )
            # print(msg, flush=True)

            dpos *= scale
        return dpos

    def update(self, pos, forces, pos0, forces0):
        if self.state is None:
            self.state = BFGSMethod(self.H0)
            return

        # We'll want to work with gradients in the future,
        # but (awkwardly) this method gets 'forces' for backwards compatibility
        self.state.update(pos, -forces, pos0, -forces0)

    def replay_trajectory(self, traj):
        """Initialize hessian from old trajectory."""
        if isinstance(traj, str):
            from ase.io.trajectory import Trajectory
            traj = Trajectory(traj, 'r')
        self.state = None
        atoms = traj[0]
        pos0 = atoms.get_positions().ravel()
        forces0 = atoms.get_forces().ravel()
        for atoms in traj:
            pos = atoms.get_positions().ravel()
            forces = atoms.get_forces().ravel()
            self.update(pos, forces, pos0, forces0)
            pos0 = pos
            forces0 = forces

        self.pos0 = pos0
        self.forces0 = forces0


class oldBFGS(BFGS):
    def determine_step(self, dpos, steplengths):
        """Old BFGS behaviour for scaling step lengths

        This keeps the behaviour of truncating individual steps. Some might
        depend of this as some absurd kind of stimulated annealing to find the
        global minimum.
        """
        dpos /= np.maximum(steplengths / self.maxstep, 1.0).reshape(-1, 1)
        return dpos
