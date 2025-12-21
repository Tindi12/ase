from itertools import product

import numpy as np

from ase.stress import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress
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
            gradient=-forces.ravel(),
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

    def converged(self, gradient):
        return self.gradient_norm(gradient) < self.fmax


class CellUtility:
    def __init__(
        self,
        orig_cell,
        mask,
        scalar_pressure=0.0,
        constant_volume=False,
        hydrostatic_strain=False,
    ):
        from scipy.linalg import expm, expm_frechet, logm

        self.orig_cell = orig_cell
        self.expm = expm
        self.expm_frechet = expm_frechet
        self.logm = logm

        if mask is None:
            mask = np.ones(6, bool)
        mask = np.asarray(mask)
        if mask.shape == (6,):
            mask = voigt_6_to_full_3x3_stress(mask)
        elif mask.shape == (3, 3):
            mask = mask
        else:
            raise ValueError('shape of mask should be (3,3) or (6,)')

        self.mask6 = full_3x3_to_voigt_6_stress(mask)
        self.mask3x3 = mask

        # Somewhat uncertain how well these are tested in combinations
        self.scalar_pressure = scalar_pressure
        self.hydrostatic_strain = hydrostatic_strain
        self.constant_volume = constant_volume

    def deform_grad(self, cell):
        return np.linalg.solve(self.orig_cell, cell).T

    def get_energy(self, atoms, force_consistent):
        atoms_energy = atoms.get_potential_energy(
            force_consistent=force_consistent
        )
        return atoms_energy + self.get_energy_correction(atoms.cell.volume)

    def get_energy_correction(self, volume: float) -> float:
        return self.scalar_pressure * volume

    def get_positions_unitcellfilter(self, positions, cell, cell_factor):
        cur_deform_grad = self.deform_grad(cell)
        natoms = len(positions)
        pos = np.zeros((natoms + 3, 3))
        # UnitCellFilter's positions are the self.atoms.positions but without
        # the applied deformation gradient
        pos[:natoms] = np.linalg.solve(cur_deform_grad, positions.T).T
        # UnitCellFilter's cell DOFs are the deformation gradient times a
        # scaling factor
        pos[natoms:] = cell_factor * cur_deform_grad
        return pos

    def set_positions_unitcellfilter(
        self, new, atoms, cell_factor, **setpos_kwargs
    ):
        # We do a few non-trivial call with Atoms so this is not decoupled
        # from atoms (yet?).
        natoms = len(atoms)
        new_atom_positions = new[:natoms]
        new_deform_grad = new[natoms:] / cell_factor
        deform = (new_deform_grad - np.eye(3)).T * self.mask3x3
        # Set the new cell from the original cell and the new
        # deformation gradient.  Both current and final structures should
        # preserve symmetry, so if set_cell() calls FixSymmetry.adjust_cell(),
        # it should be OK
        newcell = self.orig_cell @ (np.eye(3) + deform)

        atoms.set_cell(newcell, scale_atoms=True)
        # Set the positions from the ones passed in (which are without the
        # deformation gradient applied) and the new deformation gradient.
        # This should also preserve symmetry, so if set_positions() calls
        # FixSymmetry.adjust_positions(), it should be OK
        atoms.set_positions(
            new_atom_positions @ (np.eye(3) + deform), **setpos_kwargs
        )

    def get_positions_frechet(
        self, positions, cell, cell_factor, exp_cell_factor
    ):
        # XXX This is unitcellfilter's
        # default behaviour
        cell_factor = float(len(positions))
        pos = self.get_positions_unitcellfilter(positions, cell, cell_factor)
        natoms = len(positions)
        pos[natoms:] = self.logm(pos[natoms:]) * exp_cell_factor
        return pos

    def set_positions_frechet(
        self, new, atoms, cell_factor, exp_cell_factor, **setpos_kwargs
    ):
        natoms = len(atoms)
        new2 = new.copy()
        new2[natoms:] = self.expm(new[natoms:] / exp_cell_factor)
        self.set_positions_unitcellfilter(
            new2, atoms, cell_factor=cell_factor, **setpos_kwargs
        )

    def get_forces_unitcellfilter(
        self, atoms_forces, stress, cell, cell_factor
    ):
        volume = cell.volume
        virial = -volume * (
            voigt_6_to_full_3x3_stress(stress)
            + np.diag([self.scalar_pressure] * 3)
        )
        cur_deform_grad = self.deform_grad(cell)
        atoms_forces = atoms_forces @ cur_deform_grad
        virial = np.linalg.solve(cur_deform_grad, virial.T).T

        if self.hydrostatic_strain:
            vtr = virial.trace()
            virial = np.diag([vtr / 3.0, vtr / 3.0, vtr / 3.0])

        # Zero out components corresponding to fixed lattice elements
        if (self.mask3x3 != 1.0).any():
            virial *= self.mask3x3

        if self.constant_volume:
            vtr = virial.trace()
            np.fill_diagonal(virial, np.diag(virial) - vtr / 3.0)

        natoms = len(atoms_forces)
        forces = np.zeros((natoms + 3, 3))
        forces[:natoms] = atoms_forces
        forces[natoms:] = virial / cell_factor

        modified_stress = -full_3x3_to_voigt_6_stress(virial) / volume
        return forces, modified_stress

    def get_forces_frechet(self, atoms_forces, stress, cell, exp_cell_factor):
        volume = cell.volume

        virial = -volume * (
            voigt_6_to_full_3x3_stress(stress)
            + np.diag([self.scalar_pressure] * 3)
        )

        cur_deform_grad = self.deform_grad(cell)
        cur_deform_grad_log = self.logm(cur_deform_grad)

        if self.hydrostatic_strain:
            vtr = virial.trace()
            virial = np.diag([vtr / 3.0, vtr / 3.0, vtr / 3.0])

        # Zero out components corresponding to fixed lattice elements
        if (self.mask3x3 != 1.0).any():
            virial *= self.mask3x3

        # Cell gradient for UnitCellFilter
        ucf_cell_grad = virial @ np.linalg.inv(cur_deform_grad.T)

        # Cell gradient for FrechetCellFilter
        deform_grad_log_force = np.zeros((3, 3))
        for mu, nu in product(range(3), repeat=2):
            dir = np.zeros((3, 3))
            dir[mu, nu] = 1.0
            # Directional derivative of deformation to (mu, nu) strain direction
            expm_der = self.expm_frechet(
                cur_deform_grad_log, dir, compute_expm=False
            )
            deform_grad_log_force[mu, nu] = np.sum(expm_der * ucf_cell_grad)

        # Cauchy stress used for convergence testing
        convergence_crit_stress = -(virial / volume)
        if self.constant_volume:
            # apply constraint to force
            dglf_trace = deform_grad_log_force.trace()
            np.fill_diagonal(
                deform_grad_log_force,
                np.diag(deform_grad_log_force) - dglf_trace / 3.0,
            )
            # apply constraint to Cauchy stress used for convergence testing
            ccs_trace = convergence_crit_stress.trace()
            np.fill_diagonal(
                convergence_crit_stress,
                np.diag(convergence_crit_stress) - ccs_trace / 3.0,
            )

        atoms_forces = atoms_forces @ cur_deform_grad

        # pack gradients into vector
        natoms = len(atoms_forces)
        forces = np.zeros((natoms + 3, 3))
        forces[:natoms] = atoms_forces
        forces[natoms:] = deform_grad_log_force / exp_cell_factor
        return forces, convergence_crit_stress


from dataclasses import dataclass


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


def default_mask(pbc):
    mask = np.ones(6, bool)
    mask[:3] = pbc
    for i in range(3):
        if not mask[i]:
            mask[3 + (i + 1) % 3] = 0
            mask[3 + (i - 1) % 3] = 0
    return mask


class FrechetTarget:
    def __init__(self, atoms, mask=None, *, fmax, smax):
        self.atoms = atoms
        if mask is None:
            mask = default_mask(atoms.pbc)
        self.optimizable = atoms.__ase_optimizable__()
        self._utility = CellUtility(atoms.cell.copy(), mask)

        # XXX Should Target have the max values?  Maybe, because
        # it knows what they mean.
        self.fmax = fmax
        self.smax = smax

    def get_value(self):
        return (
            self.optimizable.get_value()
            + self._utility.get_energy_correction(self.atoms.cell.volume)
        )

    def get_gradient(self):
        atoms_forces = self.atoms.get_forces()
        stress = self.atoms.get_stress()
        forces, conv_crit_stress = self._utility.get_forces_frechet(
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
            gradient=-forces.ravel(),
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
    # atoms = setup_surface()

    state = BFGSState(hessian=hessian)
    for step in _new_bfgs(target, state):
        pass

    # target = Target(atoms)


def _new_bfgs(target, state):
    x = target.get_x()

    gradient_obj = target.get_gradient()
    gradient = gradient_obj.gradient

    assert gradient.shape == (len(x),)
    value = target.get_value()
    # grad_info = target.gradient_info(gradient)
    # gradient_norm = target.gradient_norm(gradient)

    i = 0
    while True:
        loginfo = gradient_obj.loginfo()
        txt = ' '.join(f'{key}={value:e}' for key, value in loginfo.items())
        print(f'BFGS i={i:4d} e={value:f} {txt}')
        if gradient_obj.converged:
            return

        i += 1
        dx = state.compute_step(gradient)

        target.set_x(x + dx)
        # Target may apply constraints or other magic
        newx = target.get_x()

        newgradient_obj = target.get_gradient()
        newgradient = newgradient_obj.gradient
        value = target.get_value()
        # gradient_norm = target.gradient_norm(newgradient)

        state.update(newx, newgradient, x, gradient)

        x = newx
        gradient_obj = newgradient_obj
        gradient = newgradient
        yield


# @pytest.mark.skip
def test_surface():
    from ase.optimize.bfgs import BFGS as OldBFGS

    atoms = setup_surface()

    bfgs = OldBFGS(atoms)
    for _ in bfgs.irun(fmax=0.01):
        pass
    # bfgs.run(fmax=0.01)


def test_new_bfgs():
    atoms = setup_surface()
    new_bfgs(Target(atoms, fmax=0.01), initial_position_hessian(3 * len(atoms)))


def test_old_frechet():
    print('OLD FRECHET')
    from ase.filters import FrechetCellFilter
    from ase.optimize.cellawarebfgs import CellAwareBFGS

    atoms = setup_surface()
    bfgs = CellAwareBFGS(
        FrechetCellFilter(atoms, exp_cell_factor=1.0, mask=[1, 1, 0, 0, 0, 1])
    )
    bfgs.run(fmax=0.01, smax=0.001)


def test_new_bfgs_frechet():
    from ase.stress import voigt_6_to_full_3x3_stress

    atoms = setup_surface()
    pos_ndofs = 3 * len(atoms)

    mask6 = [1, 1, 0, 0, 0, 1]
    mask3x3 = voigt_6_to_full_3x3_stress(mask6)

    new_bfgs(
        FrechetTarget(atoms, mask6, fmax=0.01, smax=0.001),
        hessian=initial_frechet_hessian(
            pos_ndofs, volume=atoms.cell.volume, mask3x3=mask3x3
        ),
    )
