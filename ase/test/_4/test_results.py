import numpy as np
import pytest

from ase._4.calculators.results import CalculationResults

pytestmark = pytest.mark.asev4


def test_data_at_initialisation(metadata, properties):
    # Data assigned at initialisation
    res = CalculationResults(metadata=metadata, properties=properties)
    validate(res)


def test_data_after_initialisation(metadata, properties):
    # after initialisation
    res = CalculationResults()
    res.metadata = metadata
    res.properties = properties
    validate(res)


def test_add_single_property(metadata, properties):
    # initialise and add single property
    res = CalculationResults()
    res.add_property('energy', 3.14)
    validate(res)

    # test immutability/assignment
    with pytest.raises(ValueError):
        res.add_property('energy', 4.32)


def test_wrong_dtype():
    # shouldn't allow non-dict or non-Properties
    res = CalculationResults()
    with pytest.raises(TypeError):
        res.metadata = 'test'    # type: ignore
    with pytest.raises(TypeError):
        res.properties = 'test'  # type: ignore


def test_no_overwriting(metadata, properties):
    # shouldn't allow overwriting
    res = CalculationResults(metadata=metadata, properties=properties)
    with pytest.raises(AttributeError):
        res.metadata = metadata
    with pytest.raises(AttributeError):
        res.properties = properties


def test_add_individual_metadata():
    # try setting individual metadata and properties
    res = CalculationResults()
    res.add_metadata('calculator_name', 'test')
    with pytest.raises(AttributeError):
        res.add_metadata('calculator_name', 'test2')


def test_add_individual_properties(properties):
    res = CalculationResults()
    # properties hasn't been set, should work
    res.properties = properties
    # now energy is set, should raise
    with pytest.raises(ValueError):
        res.add_property('energy', 1.62)


def test_add_unrecognised_property():
    res = CalculationResults()
    with pytest.raises(KeyError):
        res.add_property('unrecognised_energy', 1.62)


def test_add_wrong_property_shape():
    res = CalculationResults()
    # try to set the wrong shape
    with pytest.raises(ValueError):
        res.add_property('forces', np.arange(5))
    # try to set the wrong datatype
    with pytest.raises(TypeError):
        res.add_property('energy', np.arange(5))


def validate(res):
    pass
