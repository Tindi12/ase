"""Tests for ASE4 ``Geometry``."""

import pytest

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.dev.geometry import Geometry


def test_store() -> None:
    """Test if the ``store`` method works."""
    geom = Geometry(bulk('Cu'))
    calc = EMT()
    properties = ['energy']
    geom.store(calc.calculate_properties(geom, properties), label='EMT')
    assert geom.results  # check if non-empty
    assert geom.results['EMT']  # check if non-empty


def test_calc() -> None:
    """Test if the ``calc`` attribute is disabled."""
    geom = Geometry(bulk('Cu'))
    with pytest.raises(AttributeError):
        _ = geom.calc
    with pytest.raises(AttributeError):
        geom.calc = EMT()
