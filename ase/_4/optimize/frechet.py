from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterator

import numpy as np

from ase import Atoms
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
) -> np.ndarray:
    from ase._4.optimize.run import initial_position_hessian
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

    def loginfo(self) -> dict[str, float]:
        return {'fmax': self.fnorm, 'smax': self.snorm, 'vol': self.volume}

    def datafy(self) -> dict[str, Any]:
        # XXX must be able to handle the type somehow.
        # The Target type would know what Gradient type to restore.
        return asdict(self)

    @classmethod
    def undatafy(cls, dct: dict[str, Any]) -> FrechetGradient:
        return cls(**dct)


def default_mask(pbc) -> np.ndarray:
    mask = np.ones(6, bool)
    mask[:3] = pbc
    for i in range(3):
        if not mask[i]:
            mask[3 + (i + 1) % 3] = 0
            mask[3 + (i - 1) % 3] = 0
    return mask


class FrechetTarget:
    iotype = 'frechet'

    def __init__(
        self,
        atoms: Atoms,
        mask: Any = None,
        *,
        fmax: float,
        smax: float,
        orig_cell: np.ndarray | None = None,
    ):
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

    def datafy(self) -> dict[str, Any]:
        return {
            'fmax': self.fmax,
            'smax': self.smax,
            'atoms': self.atoms,
            'mask': self._utility.mask3x3.tolist(),
            'orig_cell': self._utility.orig_cell.ravel().tolist(),
            # XXX We may need to save multiple things from the Utility.
            # But currently not, because we create the utility, and
            # don't set those things.
        }

    @classmethod
    def undatafy(cls, dct: dict, calc) -> FrechetTarget:
        # XXX Here we depend directly on calculator since it's the only thing
        # we don't know how to restore.
        atoms = dct['atoms'].copy()
        atoms.calc = calc
        mask = np.array(dct['mask']).reshape(3, 3)
        assert mask.dtype == bool
        orig_cell = np.array(dct['orig_cell']).reshape(3, 3)
        return cls(
            atoms,
            mask,
            fmax=dct['fmax'],
            smax=dct['smax'],
            orig_cell=orig_cell,
        )

    @classmethod
    def undatafy_gradient(cls, dct: dict) -> FrechetGradient:
        return FrechetGradient.undatafy(dct)

    def get_value(self) -> float:
        return (
            self.optimizable.get_value()
            + self._utility.get_energy_correction(self.atoms.cell.volume)
        )

    def get_gradient(self) -> FrechetGradient:
        from ase._4.optimize.run import get_maxforce

        atoms_forces = self.atoms.get_forces()
        if self._utility.mask3x3.any():
            stress = self.atoms.get_stress()
        else:
            stress = np.zeros(6)

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
    def _cell_factor(self) -> float:
        # XXX Default behaviour taken from unitcellfilter:
        return float(len(self.atoms))

    @property
    def _exp_cell_factor(self) -> float:
        return 1.0  # always 1.0 with 'cellaware'

    def get_x(self) -> np.ndarray:
        return self._utility.get_positions_frechet(
            self.atoms.get_positions(),
            self.atoms.get_cell(),
            cell_factor=self._cell_factor,
            exp_cell_factor=self._exp_cell_factor,
        ).ravel()

    def set_x(self, x: np.ndarray) -> None:
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

    def iterimages(self) -> Iterator[Atoms]:
        yield self.atoms
