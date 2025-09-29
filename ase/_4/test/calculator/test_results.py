"""Tests for ASEv4 `CalculationResults`."""

import numpy as np
import pytest

from ase._4.calculators.results import CalculationResults


def test_data_at_initialisation(metadata, properties) -> None:
    """Test data assigned at initialisation."""
    CalculationResults(metadata=metadata, properties=properties)


def test_data_after_initialisation(metadata, properties) -> None:
    """Test data after initialisation."""
    res = CalculationResults()
    res.metadata = metadata
    res.properties = properties


def test_add_property() -> None:
    """Test `add_property`."""
    res = CalculationResults()
    res.add_property('energy', 3.14)

    # test immutability/assignment
    with pytest.raises(ValueError):
        res.add_property('energy', 4.32)


def test_wrong_dtype() -> None:
    """Test if `CalculationResults` raises `TypeError` for invalid dtypes."""
    res = CalculationResults()
    with pytest.raises(TypeError):
        res.metadata = 'test'  # type: ignore[assignment]
    with pytest.raises(TypeError):
        res.properties = 'test'  # type: ignore[assignment]


def test_no_overwriting(metadata, properties) -> None:
    """Test if `CalculationResults` does not allow overwriting."""
    res = CalculationResults(metadata=metadata, properties=properties)
    with pytest.raises(AttributeError):
        res.metadata = metadata
    with pytest.raises(AttributeError):
        res.properties = properties


def test_add_individual_metadata() -> None:
    """Test `add_metadata`."""
    res = CalculationResults()
    res.add_metadata('calculator_name', 'test')
    with pytest.raises(AttributeError):
        res.add_metadata('calculator_name', 'test2')


def test_add_individual_properties(properties) -> None:
    """Test `add_property`."""
    res = CalculationResults()
    # properties hasn't been set, should work
    res.properties = properties
    # now energy is set, should raise
    with pytest.raises(ValueError):
        res.add_property('energy', 1.62)


def test_add_unrecognised_property() -> None:
    """Test if `add_property` raises `KeyError` for unrecognized properties."""
    res = CalculationResults()
    with pytest.raises(KeyError):
        res.add_property('unrecognised_energy', 1.62)


def test_add_wrong_property_shape() -> None:
    """Test if `add_property` raises errors for wrong data types and shapes."""
    res = CalculationResults()
    # try to set the wrong shape
    with pytest.raises(ValueError):
        res.add_property('forces', np.arange(5))
    # try to set the wrong datatype
    with pytest.raises(TypeError):
        res.add_property('energy', np.arange(5))
