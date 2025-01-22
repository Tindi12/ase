"""Bussi NVT dynamics class."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ase import units
from ase.md.verlet import VelocityVerlet

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ase import Atoms


class Bussi(VelocityVerlet):
    """Bussi stochastic velocity rescaling (NVT) molecular dynamics.

    Based on the paper from Bussi et al., J. Chem. Phys. 126, 014101 (2007)
    (also available from https://arxiv.org/abs/0803.4060).
    """

    def __init__(
        self,
        atoms,
        timestep,
        temperature_K,
        taut,
        rng=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        atoms : Atoms
            The atoms object.
        timestep : float
            The time step in ASE time units.
        temperature_K : float
            The desired temperature, in Kelvin.
        taut : float
            Time constant for Bussi temperature coupling in ASE time units.
        rng : RNG object, optional
            Random number generator, by default numpy.random.
        **kwargs : dict, optional
            Additional arguments are passed to
            :class:~ase.md.md.MolecularDynamics base class.
        """
        super().__init__(atoms, timestep, **kwargs)

        self.temp = temperature_K * units.kB
        self.taut = taut
        if rng is None:
            self.rng = np.random
        else:
            self.rng = rng

        self.ndof = self.atoms.get_number_of_degrees_of_freedom()

        self.target_kinetic_energy = 0.5 * self.temp * self.ndof

        if np.isclose(self.atoms.get_kinetic_energy(), 0.0, rtol=0, atol=1e-12):
            raise ValueError(
                'Initial kinetic energy is zero. '
                'Please set the initial velocities before running Bussi NVT.'
            )

        self._exp_term = math.exp(-self.dt / self.taut)
        self._masses = self.atoms.get_masses()[:, np.newaxis]

        self.transferred_energy = 0.0

    def scale_velocities(self):
        """Do the NVT Bussi stochastic velocity scaling."""
        kinetic_energy = self.atoms.get_kinetic_energy()
        alpha = self.calculate_alpha(kinetic_energy)

        momenta = self.atoms.get_momenta()
        self.atoms.set_momenta(alpha * momenta)

        self.transferred_energy += (alpha**2 - 1.0) * kinetic_energy

    def calculate_alpha(self, kinetic_energy):
        """Calculate the scaling factor alpha using equation (A7)
        from the Bussi paper."""

        energy_scaling_term = (
            (1 - self._exp_term)
            * self.target_kinetic_energy
            / kinetic_energy
            / self.ndof
        )

        # R1 in Eq. (A7)
        noisearray = self.rng.standard_normal(size=(1,))
        # ASE mpi interfaces can only broadcast arrays, not scalars
        self.comm.broadcast(noisearray, 0)
        normal_noise = noisearray[0]

        # \sum_{i=2}^{Nf} R_i^2 in Eq. (A7)
        # 2 * standard_gamma(n / 2) is equal to chisquare(n)
        sum_of_noises = 2.0 * self.rng.standard_gamma(0.5 * (self.ndof - 1))

        return math.sqrt(
            self._exp_term
            + energy_scaling_term * (sum_of_noises + normal_noise**2)
            + 2 * normal_noise * math.sqrt(self._exp_term * energy_scaling_term)
        )

    def step(self, forces=None):
        """Move one timestep forward using Bussi NVT molecular dynamics."""
        self.scale_velocities()
        return super().step(forces)


class BussiParinello(VelocityVerlet):
    """Bussi-Parinello (NVT) Langevin-based dynamics.

    Based on the paper from Bussi et al. (https://arxiv.org/abs/0803.4083)

    Parameters
    ----------
    atoms : Atoms
        The atoms object.
    timestep : float
        The time step in ASE time units.
    temperature_K : float
        The desired temperature, in Kelvin.
    friction : float
        Friction coefficient for the Langevin thermostat.
    rng : numpy.random, optional
        Random number generator.
    **md_kwargs : dict, optional
        Additional arguments passed to :class:`~ase.md.md.MolecularDynamics`
        base class.
    """

    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        temperature_K: float,
        friction: float,
        rng: Any = np.random,
        **md_kwargs,
    ):
        super().__init__(
            atoms,
            timestep,
            **md_kwargs,
        )

        self.temperature = temperature_K * units.kB
        self.friction = friction
        self.rng = rng

    def thermostat_half_step(self) -> None:
        """Move one half timestep forward using Bussi-Parinello
        NVT molecular dynamics."""
        self.atoms.set_momenta(
            self.coefficient_1 * self.atoms.get_momenta()
            + self.coefficient_2 * self.rng.normal(size=(len(self.atoms), 3))
        )

    def last_thermostat_half_step(self) -> None:
        """Move one half timestep forward using Bussi-Parinello
        NVT molecular dynamics."""
        self.atoms.set_momenta(
            self.coefficient_1 * self.atoms.get_momenta()
            + self.coefficient_2 * self.rng.normal(size=(len(self.atoms), 3))
        )

    @property
    def friction(self) -> float:
        """Friction coefficient for the Langevin thermostat."""
        return self._friction

    @friction.setter
    def friction(self, value: float) -> None:
        """Set the friction coefficient for the Langevin thermostat."""
        self._friction = value
        self.coefficient_1: float = math.exp(-self._friction * self.dt / 2)
        self.coefficient_2: NDArray = np.sqrt(
            (1 - self.coefficient_1**2)
            * self.atoms.get_masses()[:, None]
            * self._temperature
        )

    @property
    def temperature(self) -> float:
        """The desired temperature, in Kelvin."""
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        """Set the desired temperature, in Kelvin."""
        self._temperature = value
        self.target_kinetic_energy = (
            0.5
            * self._temperature
            * self.atoms.get_number_of_degrees_of_freedom()
        )

    def step(self, forces: NDArray | None = None) -> Any:
        """Move one timestep forward using Bussi-Parinello
        NVT molecular dynamics.

        Parameters
        ----------
        forces : NDArray, optional
            Forces acting on the atoms. If None, forces will be calculated.

        Returns
        -------
        forces : NDArray
            Forces acting on the atoms after the step.
        """
        self.thermostat_half_step()
        forces = super().step(forces)
        self.last_thermostat_half_step()

        return forces
