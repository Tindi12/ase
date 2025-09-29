"""`conftest.py` for ASEv4."""

import numpy as np
import pytest

from ase.outputs import Properties


@pytest.fixture
def metadata() -> dict:
    """ "Input to create CalculationResults from"""
    return {'calculator_name': 'test_calculator', 'calculator_version': 'v3.2'}


@pytest.fixture(name='properties_dict')
def fixture_properties_dict() -> dict:
    """ "Input to create CalculationResults from"""
    data = {
        'energy': 3.14,
        'forces': np.arange(15).reshape((5, 3)),
        'stress': np.arange(6),
    }
    return data


@pytest.fixture(params=['dict', 'Properties'])
def properties(
    request: pytest.FixtureRequest,
    properties_dict: dict,
) -> dict | Properties:
    """Two ways of initialising properties in CalculationResults."""
    if request.param == 'dict':
        return properties_dict
    if request.param == 'Properties':
        return Properties(properties_dict)
    raise RuntimeError
