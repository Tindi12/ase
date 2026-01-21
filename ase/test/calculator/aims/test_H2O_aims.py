# fmt: off
import pytest

from ase import Atoms
from ase.optimize import QuasiNewton

@pytest.mark.calculator('aims')
def test_H2O_aims(factory):
    water = Atoms('HOH', [(1, 0, 0), (0, 0, 0), (0, 1, 0)])

    calc = factory.calc(
        xc='LDA',
        output=['dipole'],
        sc_accuracy_etot=1e-2,
        sc_accuracy_eev=1e-1,
        sc_accuracy_rho=1e-2,
        sc_accuracy_forces=1e-1,
    )

    water.calc = calc
    dynamics = QuasiNewton(water)
    dynamics.run(fmax=0.2)
