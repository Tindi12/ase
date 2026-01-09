from dataclasses import asdict, dataclass

import numpy as np

from ase._4.optimize.cellutil import CellUtility
from ase.units import GPa


def get_maxstress(stress) -> float:
    return np.abs(stress).max()


def initial_frechet_hessian(
    position_dofs: int,
    volume: float,
    mask3x3: np.ndarray,
    bulk_modulus: float = 145 * GPa,
    poisson_ratio: float = 0.3,
    alpha: float = 70.0,
):
    from ase._4.optimize.bfgs import initial_position_hessian
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
        from ase._4.optimize.bfgs import get_maxforce

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
