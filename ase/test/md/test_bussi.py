import numpy as np
import pytest

from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md.bussi import Bussi, BussiLangevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


def test_bussi():
    atoms = bulk('Pt')
    atoms.calc = EMT()

    with pytest.raises(ValueError):
        with Bussi(atoms, 0.1 * units.fs, 300, 100 * units.fs) as dyn:
            dyn.run(1)

    MaxwellBoltzmannDistribution(
        atoms, temperature_K=300, rng=np.random.default_rng(seed=42)
    )

    with Bussi(
        atoms,
        0.1 * units.fs,
        300,
        100 * units.fs,
    ) as dyn:
        dyn.run(10)

    assert dyn.taut == 100 * units.fs
    assert dyn.temp == 300 * units.kB
    assert dyn.dt == 0.1 * units.fs
    assert dyn.ndof == 3
    assert dyn.target_kinetic_energy == 0.5 * dyn.temp * dyn.ndof
    assert dyn.transferred_energy != 0.0


def test_bussi_transfered_energy_conservation():
    atoms = bulk('Cu') * (4, 4, 4)

    atoms.calc = EMT()

    MaxwellBoltzmannDistribution(
        atoms, temperature_K=300, rng=np.random.default_rng(seed=42)
    )

    conserved_quantity = []

    with Bussi(
        atoms,
        1.0e-5 * units.fs,
        300,
        100 * units.fs,
        rng=np.random.default_rng(seed=42),
    ) as dyn:
        for _ in dyn.irun(100):
            conserved_quantity.append(
                dyn.atoms.get_total_energy() - dyn.transferred_energy
            )

    assert np.unique(np.round(conserved_quantity, 10)).size == 1


def test_bussi_paranoia_check():
    """Test that check if the average temperature
    of the system is close to the target temperature.

    From Giovanni Bussi:

    As a further paranoia check, you can try to run a simulation with:

        - tiny timestep (say 1e-100) so that atoms do not move
        - tiny tau (say 1e-100) so that kinetic energy equilibrates anyway

    The distribution of the kinetic energy should converge
    quickly to the correct one."""

    atoms = bulk('Cu') * (3, 3, 3)

    atoms.calc = EMT()

    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=300,
        rng=np.random.default_rng(seed=10),
        force_temp=True,
    )

    temperatures = []

    with Bussi(
        atoms,
        1.0e-100 * units.fs,
        300,
        1.0e-100 * units.fs,
        rng=np.random.default_rng(seed=10),
    ) as dyn:
        for _ in dyn.irun(1000):
            temperatures.append(dyn.atoms.get_temperature())

    assert np.mean(temperatures) == pytest.approx(300, abs=5.0)


def test_bussi_paranoia_check2():
    """We test even DOF"""

    atoms = bulk('Cu') * (4, 4, 4)

    atoms.calc = EMT()

    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=300,
        rng=np.random.default_rng(seed=91),
        force_temp=True,
    )

    temperatures = []

    with Bussi(
        atoms,
        1.0e-100 * units.fs,
        300,
        1.0e-100 * units.fs,
        rng=np.random.default_rng(seed=91),
    ) as dyn:
        for _ in dyn.irun(1000):
            temperatures.append(dyn.atoms.get_temperature())

    assert np.mean(temperatures) == pytest.approx(300, abs=5.0)


def test_bussi_parinello():
    atoms = bulk('Cu') * (4, 4, 4)

    atoms.calc = EMT()

    MaxwellBoltzmannDistribution(
        atoms, temperature_K=500, rng=np.random.default_rng(seed=42)
    )

    temperatures = []
    velocities = []

    with BussiLangevin(
        atoms,
        0.1 * units.fs,
        500,
        5000,
        logfile=None,
        rng=np.random.default_rng(),
    ) as dyn:
        assert dyn.friction == 50
        assert dyn.temperature == 500 * units.kB
        assert dyn.coefficient_1 == pytest.approx(
            np.exp(-50 * 0.1 * units.fs / 2)
        )
        assert dyn.coefficient_2 == pytest.approx(
            np.sqrt(
                (1 - np.exp(-50 * 0.1 * units.fs / 2) ** 2)
                * atoms.get_masses()[:, None]
                * 500
                * units.kB
            )
        )
        assert (
            dyn.target_kinetic_energy == 0.5 * 500 * units.kB * len(atoms) * 3
        )
        for _ in dyn.irun(1000):
            temperatures.append(dyn.atoms.get_temperature())
            velocities.append(dyn.atoms.get_velocities())

        dyn.friction = 100

        assert dyn.friction == 100
        assert dyn.coefficient_1 == pytest.approx(
            np.exp(-100 * 0.1 * units.fs / 2)
        )
        assert dyn.coefficient_2 == pytest.approx(
            np.sqrt(
                (1 - np.exp(-100 * 0.1 * units.fs / 2) ** 2)
                * atoms.get_masses()[:, None]
                * 500
                * units.kB
            )
        )

        dyn.temperature = 1000

        assert dyn.temperature == 1000
        assert dyn.target_kinetic_energy == 0.5 * 1000 * len(atoms) * 3

    assert np.mean(temperatures) == pytest.approx(500, abs=50.0)

    velocities = np.array(velocities).flatten()

    from scipy.stats import norm

    mean, std = norm.fit(velocities)

    assert mean == pytest.approx(0, abs=1.0e-2)

    assert std == pytest.approx(
        np.sqrt(500 * units.kB / (atoms.get_masses()[0])), abs=1.0e-2
    )

    from scipy.stats import anderson

    results = anderson(velocities)

    # 5% significance level, if this is true, the data is normal
    assert results.statistic < results.critical_values[2]
