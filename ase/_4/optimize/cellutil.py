from itertools import product

import numpy as np

from ase.stress import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress


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
