import numpy as np
import pytest

from ase import Atoms as V3Atoms
from ase._4.atoms import Atoms as V4Atoms
from ase._4.calculators.calculator import BaseCalculator
from ase._4.calculators.emt import EMT
from ase.build import bulk
from ase.outputs import Properties


def pytest_collection_modifyitems(config, items):
    """By default skips tests with "asev4" mark."""
    run_v4 = config.option.markexpr and 'asev4' in config.option.markexpr
    skip_marker = pytest.mark.skip(reason="ASEv4 test, run with '-m asev4'")

    for item in items:
        if '_4' in str(item.fspath):
            if not run_v4:
                item.add_marker(skip_marker)


@pytest.fixture
def atoms() -> V3Atoms:
    return bulk('Cu', 'fcc', a=3.6)


# a bit hacky for now
@pytest.fixture
def v4atoms(atoms) -> V4Atoms:
    v4atoms = V4Atoms(
        symbols='Cu',
        positions=atoms.positions,
        cell=atoms.cell,
        pbc=atoms.pbc,
    )
    return v4atoms


@pytest.fixture
def calculator() -> BaseCalculator:
    calculator = EMT()
    return calculator


@pytest.fixture
def metadata() -> dict:
    """ "Input to create CalculationResults from"""
    data = {'calculator_name': 'test_calculator', 'calculator_version': 'v3.2'}
    return data


@pytest.fixture()
def properties_dict() -> dict:
    """ "Input to create CalculationResults from"""
    data = {
        'energy': 3.14,
        'forces': np.arange(15).reshape((5, 3)),
        'stress': np.arange(6),
    }
    return data


@pytest.fixture(params=['dict', 'Properties'])
def properties(request, properties_dict) -> dict | Properties:
    """Two ways of initialising properties in CalculationResults."""
    if request.param == 'dict':
        properties = properties_dict
    elif request.param == 'Properties':
        properties = Properties(properties_dict)

    return properties
