import numpy as np
import pytest

from ase._4.calculators.emt import EMT

pytestmark = pytest.mark.asev4


def test_emt(atoms):
    calculator = EMT()

    properties = ['energy', 'energies', 'free_energy', 'forces', 'stress']
    results = calculator.evaluate(atoms, properties=properties)

    ref_results = {
        'energy': -0.006688768,
        'free_energy': -0.006688768,
        'energies': np.array([-0.006688768]),
        'forces': np.array([[1.438114e-15, -4.589022e-15, 5.283145e-15]]),
        'stress': np.array(
            [
                7.000141e-03,
                7.000141e-03,
                7.000141e-03,
                -1.618551e-16,
                1.319962e-16,
                -7.050540e-17,
            ]
        ),
        # Gets added with 'energies' in outputs.py, should be changed?
        'natoms': 1,
    }

    for key, val in results.properties.items():
        assert np.allclose(val, ref_results[key])
