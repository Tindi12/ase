import numpy as np
import pytest

from ase.units import GPa


class BFGSState:
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


def setup_surface():
    from ase.build import fcc111
    from ase.calculators.emt import EMT

    rng = np.random.RandomState(42)
    atoms = fcc111('Au', size=(1, 2, 2), vacuum=5.0)
    atoms.rattle(stdev=0.01, rng=rng)
    cell = atoms.get_cell()
    cell[:2, :2] += 0.05 * rng.random((2, 2))
    atoms.set_cell(cell, scale_atoms=True)
    atoms.calc = EMT()
    return atoms


class Target:
    def __init__(self, atoms):
        self.optimizable = atoms.__ase_optimizable__()

    def get_value(self):
        return self.optimizable.get_value()

    def get_gradient(self):
        return self.optimizable.get_gradient()

    def get_x(self):
        return self.optimizable.get_x()

    def set_x(self, x):
        self.optimizable.set_x(x)

    def gradient_norm(self, gradient):
        return self.optimizable.gradient_norm(gradient)


class CellUtility:
    def __init__(self, orig_cell):
        from scipy.linalg import expm, expm_frechet, logm

        self.orig_cell = orig_cell
        self.expm = expm
        self.expm_frechet = expm_frechet
        self.logm = logm

    def deform_grad(self, cell):
        return np.linalg.solve(self.orig_cell, cell).T


class FrechetTarget:
    def __init__(self, atoms, mask):
        self.atoms = atoms
        self.optimizable = atoms.__ase_optimizable__()
        self.utility = CellUtility(atoms.cell.copy())

    def get_value(self):
        return self.optimizable.get_value()

    def get_gradient(self):
        natomdofs = len(self.atoms) * 3
        ncelldofs = 9
        ndofs = len(self.atoms) * 3 + 9
        gradient = np.empty(natomdofs + ncelldofs)
        gradient[:natomdofs] = self.atoms.get_forces().ravel()
        # Instead of multiplying mask, we should simply not expose those DOFs.
        # Also if there are only 6 stresses should we really be optimizing 3x3?
        stress = self.atoms.get_stress(voigt=False) * self.utility.mask_3x3
        gradient[natomdofs:] = stress.ravel()
        return gradient

    def get_x(self):
        # from scipy.linalg import expm, expm_frechet, logm

        exp_cell_factor = 1.0  # always 1.0 with 'cellaware'
        ...
        # pos = UnitCellFilter.get_positions(self)
        # natoms = len(self.atoms)
        # pos[natoms:] = self.utility.logm(pos[natoms:]) * exp_cell_factor
        # return pos.ravel()

    def set_x(self, x): ...

    def gradient_norm(self, gradient): ...


def initial_position_hessian(ndofs, alpha=70.0):
    return np.diag(np.full(ndofs, 70.0))


def initial_frechet_hessian(
    position_dofs: int,
    volume: float,
    mask_3x3: np.ndarray,
    bulk_modulus: float = 145 * GPa,
    poisson_ratio: float = 0.3,
    alpha: float = 70.0,
):
    from ase.optimize.cellawarebfgs import calculate_isotropic_elasticity_tensor

    C_ijkl = calculate_isotropic_elasticity_tensor(
        bulk_modulus, poisson_ratio, suppress_rotation=alpha
    )

    ndofs = position_dofs + 9
    hessian = np.zeros((ndofs, ndofs))
    hessian[:-9, :-9] = initial_position_hessian(position_dofs)

    mask_ind = np.where(mask_3x3.ravel() != 0)[0]
    indices = np.ix_(mask_ind, mask_ind)
    # Instead of zeroing, can we make the Hessian smaller when we are not
    # optimizing all cell DOFs?
    # Also, instead of not assigning masked cell DOFs, can't we just assign
    # them unconditionally and rely on the algorithm to do what it likes?
    cell_hessian = hessian[-9:, -9:]
    cell_hessian[indices] = C_ijkl.reshape((9, 9))[indices] * volume
    hessian[position_dofs:, position_dofs:] = cell_hessian
    return hessian


def new_bfgs(target, hessian, fmax=0.01):
    # atoms = setup_surface()

    # target = Target(atoms)

    x = target.get_x()
    ndofs = len(x)

    state = BFGSState(hessian=hessian)

    gradient = target.get_gradient()
    value = target.get_value()
    gradient_norm = target.gradient_norm(gradient)

    i = 0
    while True:
        print(f'BFGS i={i:4d} e={value:f} fmax={gradient_norm:f}')
        if gradient_norm < fmax:
            return

        i += 1
        dx = state.compute_step(gradient)

        target.set_x(x + dx)
        # Target may apply constraints or other magic
        newx = target.get_x()

        newgradient = target.get_gradient()
        value = target.get_value()
        gradient_norm = target.gradient_norm(newgradient)

        state.update(newx, newgradient, x, gradient)

        x = newx
        gradient = newgradient


@pytest.mark.skip
def test_surface():
    from ase.filters import FrechetCellFilter
    from ase.optimize.bfgs import BFGS as OldBFGS

    atoms = setup_surface()
    f = atoms.get_forces()

    bfgs = OldBFGS(atoms)  # FrechetCellFilter(atoms))
    for _ in bfgs.irun(fmax=0.01):
        pass
    # bfgs.run(fmax=0.01)


# @pytest.mark.skip
def test_surface_cellawarebfgs():
    from ase.filters import FrechetCellFilter
    from ase.optimize.cellawarebfgs import CellAwareBFGS

    atoms = setup_surface()
    bfgs = CellAwareBFGS(
        FrechetCellFilter(atoms, exp_cell_factor=1.0, mask=[1, 1, 0, 0, 0, 1])
    )
    bfgs.run(fmax=0.01, smax=0.001)


def test_new_bfgs():
    atoms = setup_surface()
    new_bfgs(Target(atoms), initial_position_hessian(3 * len(atoms)))


def test_new_bfgs_frechet():
    from ase.stress import voigt_6_to_full_3x3_stress

    atoms = setup_surface()
    pos_ndofs = 3 * len(atoms)

    mask = [1, 1, 0, 0, 0, 1]
    mask_3x3 = voigt_6_to_full_3x3_stress(mask)

    new_bfgs(
        FrechetTarget(atoms, mask),
        hessian=initial_frechet_hessian(
            pos_ndofs, volume=atoms.cell.volume, mask_3x3=mask_3x3
        ),
    )
