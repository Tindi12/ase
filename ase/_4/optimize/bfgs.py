import numpy as np


class BFGSState:
    def __init__(self, hessian):
        self.hessian = hessian

    @property
    def H(self):
        return self.hessian

    def compute_step(self, gradient):
        omega, vectors = np.linalg.eigh(self.hessian)
        return -vectors @ (gradient @ vectors / np.fabs(omega))

    def update(self, pos, forces, pos0, forces0):
        dpos = pos - pos0

        if np.abs(dpos).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return

        dforces = forces - forces0
        a = dpos @ dforces
        dg = self.hessian @ dpos
        b = dpos @ dg
        self.hessian -= np.outer(dforces, dforces) / a + np.outer(dg, dg) / b


def setup_surface():
    from ase.build import fcc111
    from ase.calculators.emt import EMT

    rng = np.random.RandomState(42)
    atoms = fcc111('Au', size=(2, 2, 2), vacuum=5.0)
    atoms.rattle(stdev=0.05, rng=rng)
    cell = atoms.get_cell()
    cell[:2, :2] += 0.1 * rng.random((2, 2))
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


def new_bfgs(tolerance=0.01):
    atoms = setup_surface()

    target = Target(atoms)

    x = target.get_x()
    ndofs = len(x)

    state = BFGSState(hessian=np.diag(np.full(ndofs, 70.0)))

    gradient = target.get_gradient()
    value = target.get_value()
    gradient_norm = target.gradient_norm(gradient)

    i = 0
    while True:
        print(f'BFGS i={i:4d} e={value:f} fmax={gradient_norm:f}')
        if gradient_norm < tolerance:
            return

        i += 1
        dx = state.compute_step(gradient)

        target.set_x(x + dx)
        # Target may apply constraints or other magic
        newx = target.get_x()

        newgradient = target.get_gradient()
        value = target.get_value()
        gradient_norm = target.gradient_norm(newgradient)

        state.update(newx, -newgradient, x, -gradient)

        x = newx
        gradient = newgradient


def test_surface():
    from ase.filters import FrechetCellFilter
    from ase.optimize.bfgs import BFGS as OldBFGS

    atoms = setup_surface()
    f = atoms.get_forces()

    bfgs = OldBFGS(atoms)  # FrechetCellFilter(atoms))
    for _ in bfgs.irun(fmax=0.01):
        pass
    # bfgs.run(fmax=0.01)

import pytest
@pytest.mark.skip
def test_surface_cellawarebfgs():
    from ase.filters import FrechetCellFilter
    from ase.optimize.cellawarebfgs import CellAwareBFGS

    atoms = setup_surface()
    bfgs = CellAwareBFGS(FrechetCellFilter(atoms, exp_cell_factor=1.0))
    bfgs.run(fmax=0.01)


def test_new_bfgs():
    new_bfgs()
