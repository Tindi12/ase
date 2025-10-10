import pytest

pytestmark = pytest.mark.asev4


def test_minimal(v4atoms, calculator):
    properties = ['energy', 'forces', 'stress']
    results = calculator.evaluate(v4atoms, properties=properties)
    v4atoms.store(results, label='test_')

    # could be stored as test_atoms.atoms_data
    assert 'test_energy' in v4atoms.info
    assert 'test_forces' in v4atoms.arrays
    assert 'test_stress' in v4atoms.info
