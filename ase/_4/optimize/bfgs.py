import json
from dataclasses import asdict, dataclass

import numpy as np

from ase._4.optimize.cellutil import CellUtility
from ase.io.jsonio import (
    default as jsonio_default,
)
from ase.io.jsonio import (
    object_hook as jsonio_object_hook,
)
from ase.io.trajectory import Trajectory
from ase.units import GPa


class BFGSMethod:
    methodname = 'BFGS'

    def __init__(self, hessian):
        self.hessian = hessian

    @property
    def H(self):
        return self.hessian

    def compute_step(self, gradient):
        omega, vectors = np.linalg.eigh(self.hessian)
        return -vectors @ (gradient @ vectors / np.fabs(omega))

    def update(self, pos, gradient, pos0, gradient0):
        dpos = pos - pos0

        if np.abs(dpos).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return

        dgradient = gradient - gradient0
        a = dpos @ dgradient
        dg = self.hessian @ dpos
        b = dpos @ dg
        self.hessian -= (
            -np.outer(dgradient, dgradient) / a + np.outer(dg, dg) / b
        )

    def datafy(self):
        return self.hessian.ravel().tolist()

    @classmethod
    def undatafy(cls, hessian):
        n = int(np.round(len(hessian) ** 0.5))
        hessian = np.array(hessian).reshape(n, n)
        return cls(hessian)


class Target:
    def __init__(self, atoms, fmax):
        self.optimizable = atoms.__ase_optimizable__()
        self.fmax = fmax

    def get_value(self):
        return self.optimizable.get_value()

    def get_gradient(self):
        forces = self.optimizable.atoms.get_forces()
        gradient = -forces.ravel()
        fnorm = get_maxforce(forces)
        converged = fnorm < self.fmax
        return ForceGradient(
            gradient=gradient,
            forces=forces,
            fnorm=fnorm,
            converged=converged,
        )

    def get_x(self):
        return self.optimizable.get_x()

    def set_x(self, x):
        self.optimizable.set_x(x)

    def gradient_norm(self, gradient):
        return self.optimizable.gradient_norm(gradient)

    def converged(self, gradient) -> bool:
        return self.gradient_norm(gradient) < self.fmax

    def initial_hessian(self, alpha=70.0) -> np.ndarray:
        return initial_position_hessian(self.optimizable.ndofs(), alpha)


@dataclass
class FrechetGradient:
    gradient: np.ndarray
    forces: np.ndarray
    stress: np.ndarray
    conv_crit_stress: np.ndarray
    fnorm: float
    snorm: float
    converged: bool
    volume: float

    def loginfo(self):
        return {'fmax': self.fnorm, 'smax': self.snorm, 'vol': self.volume}

    def datafy(self):
        # XXX must be able to handle the type somehow.
        # The Target type would know what Gradient type to restore.
        return asdict(self)

    @classmethod
    def undatafy(cls, dct):
        return cls(**dct)


def default_mask(pbc):
    mask = np.ones(6, bool)
    mask[:3] = pbc
    for i in range(3):
        if not mask[i]:
            mask[3 + (i + 1) % 3] = 0
            mask[3 + (i - 1) % 3] = 0
    return mask


class FrechetTarget:
    def __init__(self, atoms, mask=None, *, fmax, smax, orig_cell=None):
        self.atoms = atoms
        if mask is None:
            mask = default_mask(atoms.pbc)
        self.optimizable = atoms.__ase_optimizable__()
        if orig_cell is None:
            orig_cell = atoms.cell.copy()
        self._utility = CellUtility(orig_cell, mask)

        # XXX Should Target have the max values?  Maybe, because
        # it knows what they mean.
        self.fmax = fmax
        self.smax = smax

    def datafy(self):
        return {
            'fmax': self.fmax,
            'smax': self.smax,
            # 'atoms': self.atoms,
            # do we need atoms?  Requires ASE encoder.
            # If we do not save Atoms, we need to get at least
            # the species etc. back.  That's tricky, I suppose we should
            # save the atoms them.
            'atoms': self.atoms,
            # Also atoms include constraints, which nobody else will save
            # for us.
            'mask': self._utility.mask6.tolist(),
            'orig_cell': self._utility.orig_cell.ravel().tolist(),
            # XXX We may need to save multiple things from the Utility.
        }

    @classmethod
    def undatafy(cls, dct, calc):
        # XXX Here we depend directly on calculator since it's the only thing
        # we don't know how to restore.
        atoms = dct['atoms'].copy()
        atoms.calc = calc
        mask = np.array(dct['mask'])
        orig_cell = np.array(dct['orig_cell']).reshape(3, 3)
        return cls(
            atoms,
            mask,
            fmax=dct['fmax'],
            smax=dct['smax'],
            orig_cell=orig_cell,
        )

    @classmethod
    def undatafy_gradient(cls, dct):
        return FrechetGradient.undatafy(dct)

    def get_value(self):
        return (
            self.optimizable.get_value()
            + self._utility.get_energy_correction(self.atoms.cell.volume)
        )

    def get_gradient(self):
        atoms_forces = self.atoms.get_forces()
        stress = self.atoms.get_stress()
        frechet_forces, conv_crit_stress = self._utility.get_forces_frechet(
            atoms_forces=atoms_forces,
            stress=stress,
            cell=self.atoms.get_cell(),
            exp_cell_factor=self._exp_cell_factor,
        )

        # (Convergence criterion and maybe metric should be more pluggable)
        fnorm = get_maxforce(atoms_forces)
        snorm = get_maxstress(conv_crit_stress)
        converged = fnorm < self.fmax and snorm < self.smax

        return FrechetGradient(
            gradient=-frechet_forces.ravel(),
            forces=atoms_forces,
            stress=stress,
            conv_crit_stress=conv_crit_stress,
            fnorm=fnorm,
            snorm=snorm,
            converged=converged,
            volume=self.atoms.cell.volume,
        )

    @property
    def _cell_factor(self):
        # XXX Default behaviour taken from unitcellfilter:
        return float(len(self.atoms))

    @property
    def _exp_cell_factor(self):
        return 1.0  # always 1.0 with 'cellaware'

    def get_x(self):
        return self._utility.get_positions_frechet(
            self.atoms.get_positions(),
            self.atoms.get_cell(),
            cell_factor=self._cell_factor,
            exp_cell_factor=self._exp_cell_factor,
        ).ravel()

    def set_x(self, x):
        self._utility.set_positions_frechet(
            x.reshape(-1, 3),
            self.atoms,
            self._cell_factor,
            self._exp_cell_factor,
        )

    def initial_hessian(
        self,
        bulk_modulus: float = 145 * GPa,
        poisson_ratio: float = 0.3,
        alpha: float = 70.0,
    ) -> np.ndarray:
        return initial_frechet_hessian(
            len(self.atoms) * 3,
            # XXX volume should be set intelligently in lowdim cases.
            # What happens currently in 1d/2d?
            # We need a test of that.
            self.atoms.cell.volume,
            self._utility.mask3x3,
            bulk_modulus,
            poisson_ratio,
            alpha,
        )

    def iterimages(self):
        yield self.atoms


def get_maxforce(forces) -> float:
    return np.linalg.norm(forces, axis=1).max()


def get_maxstress(stress) -> float:
    return np.abs(stress).max()


@dataclass
class ForceGradient:
    gradient: np.ndarray
    forces: np.ndarray
    fnorm: float
    converged: bool

    def loginfo(self):
        return {'fmax': self.fnorm}


def initial_position_hessian(ndofs, alpha=70.0):
    return np.diag(np.full(ndofs, 70.0))


def initial_frechet_hessian(
    position_dofs: int,
    volume: float,
    mask3x3: np.ndarray,
    bulk_modulus: float = 145 * GPa,
    poisson_ratio: float = 0.3,
    alpha: float = 70.0,
):
    from ase.optimize.cellawarebfgs import calculate_isotropic_elasticity_tensor

    C_ijkl = calculate_isotropic_elasticity_tensor(
        bulk_modulus, poisson_ratio, suppress_rotation=alpha
    )

    ndofs = position_dofs + 9
    hessian = initial_position_hessian(ndofs, alpha)
    hessian[:-9, :-9] = initial_position_hessian(position_dofs)

    mask_ind = np.where(mask3x3.ravel() != 0)[0]
    indices = np.ix_(mask_ind, mask_ind)
    # Instead of zeroing, can we make the Hessian smaller when we are not
    # optimizing all cell DOFs?
    # Also, instead of not assigning masked cell DOFs, can't we just assign
    # them unconditionally and rely on the algorithm to do what it likes?
    cell_hessian = hessian[-9:, -9:]
    cell_hessian[indices] = C_ijkl.reshape((9, 9))[indices] * volume
    hessian[position_dofs:, position_dofs:] = cell_hessian
    return hessian


def new_bfgs(target, hessian):
    method = BFGSMethod(hessian=hessian)

    step = Step.start(target)
    assert step.gradient_obj.gradient.shape == (len(step.x),)

    yield step
    yield from irun(target, method, step)


@dataclass
class Step:
    i: int
    x: np.ndarray
    gradient_obj: object
    value: float

    @classmethod
    def start(cls, target):
        return cls(0, target.get_x(), target.get_gradient(), target.get_value())

    def datafy(self):
        return {
            'i': self.i,
            'x': self.x.tolist(),
            'gradient_obj': self.gradient_obj.datafy(),
            'value': self.value,
        }

    @classmethod
    def undatafy(cls, dct, gradient_obj):
        return cls(
            i=dct['i'],
            x=np.array(dct['x']),
            gradient_obj=gradient_obj,
            value=dct['value'],
        )


def irun(target, method, step):
    while not step.gradient_obj.converged:
        # (Both method and target change in this update)
        step = next_step(target, method, step)
        yield step


def next_step(target, method, step) -> Step:
    dx = method.compute_step(step.gradient_obj.gradient)
    # We do not have maxstep right now.  This will not run the same
    # as legacy optimizations until we apply a maxstep.

    # Target may apply constraints or other magic, so we may not
    # get the same x back as the one we set.
    target.set_x(step.x + dx)

    newstep = Step(
        i=step.i + 1,
        x=target.get_x(),
        gradient_obj=target.get_gradient(),
        value=target.get_value(),
    )

    method.update(
        newstep.x,
        newstep.gradient_obj.gradient,
        step.x,
        step.gradient_obj.gradient,
    )
    return newstep


def write_to_log(method, log, step):
    loginfo = step.gradient_obj.loginfo()
    name = method.methodname
    txt = ' '.join(f'{key}={value:e}' for key, value in loginfo.items())
    msg = f'{name} i={step.i:4d} e={step.value:f} {txt}\n'
    log.write(msg)


def write_to_traj(target, trajpath, comm):
    with Trajectory(trajpath, comm=comm, mode='a') as traj:
        # XXX we are not setting metadata (like old optimizers)
        traj.write(target)


def read_images(trajpath):
    with Trajectory(trajpath) as traj:
        return [*traj]


def write_restartfile(restartpath, method, target, step):
    # Unsafe if we just overwrite, we should backup/delete to prevent
    # accidental partial save

    # Still need some things, like maximum iterations.
    # How about trajectory writing, logfile settings, etc.?
    # General observers obviously cannot be saved.
    savedata = {
        'method': [method.methodname, method.datafy()],
        'target': target.datafy(),
        'step': step.datafy(),
    }
    json_text = json.dumps(savedata, default=jsonio_default)
    restartpath.write_text(json_text)


def read_restartfile(restartpath, calc):
    json_text = restartpath.read_text()
    dct = json.loads(json_text, object_hook=jsonio_object_hook)
    methodname, data = dct['method']
    if methodname == 'BFGS':
        method = BFGSMethod.undatafy(data)
    else:
        raise ValueError(f'No such method: {methodname}')

    # XXX Identity of target must be coded in restartfile as well.
    target = FrechetTarget.undatafy(dct['target'], calc)
    gradient_obj = target.undatafy_gradient(dct['step']['gradient_obj'])
    step = Step.undatafy(dct['step'], gradient_obj)

    return target, method, step
