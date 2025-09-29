import numpy as np
import pytest

from ase.outputs import Properties


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
