import warnings

import numpy as np
import pytest

from ase.atoms import Atoms
from ase.build import bulk
from ase.constraints import (
    FixAtoms,
    FixCartesian,
    FixCartesianParametricRelations,
    FixScaledParametricRelations,
)
from ase.io.aims import read_aims as read

format = "aims"

file = "geometry.in"


@pytest.fixture()
def Si():
    return bulk("Si")


@pytest.fixture()
def H2O():
    return Atoms("H2O", [(0.9584, 0.0, 0.0),
                 (-0.2400, 0.9279, 0.0), (0.0, 0.0, 0.0)])


def test_cartesian_Si(Si):
    """write cartesian coords and check if structure was preserved"""
    Si.write(file, format=format)
    new_atoms = read(file)
    assert np.allclose(Si.positions, new_atoms.positions)


def test_scaled_Si(Si):
    """write fractional coords and check if structure was preserved"""
    Si.write(file, format=format, scaled=True, wrap=False)
    new_atoms = read(file)
    assert np.allclose(Si.positions, new_atoms.positions)


def test_param_const_Si(Si):
    """Check to ensure parametric constraints are passed to crystal systems"""
    param_lat = ["a"]
    expr_lat = [
        "0",
        "a / 2.0",
        "a / 2.0",
        "a / 2.0",
        "0",
        "a / 2.0",
        "a / 2.0",
        "a / 2.0",
        "0",
    ]
    constr_lat = FixCartesianParametricRelations.from_expressions(
        indices=[0, 1, 2],
        params=param_lat,
        expressions=expr_lat,
        use_cell=True,
    )

    param_atom = []
    expr_atom = [
        "0.0",
        "0.0",
        "0.0",
        "0.25",
        "0.25",
        "0.25",
    ]
    constr_atom = FixScaledParametricRelations.from_expressions(
        indices=[0, 1],
        params=param_atom,
        expressions=expr_atom,
    )
    Si.set_constraint([constr_atom, constr_lat])
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        # Attempt to write a molecular system with geo_constrain=True
        Si.write(file, geo_constrain=True)
        assert len(w) == 1
        assert (
            str(w[-1].message)
            == "Setting scaled to True because a symmetry_block is detected."
        )

    new_atoms = read(file)
    assert np.allclose(Si.positions, new_atoms.positions)
    assert len(Si.constraints) == len(new_atoms.constraints)
    assert str(Si.constraints[0]) == str(new_atoms.constraints[1])
    assert str(Si.constraints[1]) == str(new_atoms.constraints[0])


def test_wrap_Si(Si):
    """write fractional coords and check if structure was preserved"""
    Si.positions[0, 0] -= 0.015625
    Si.write(file, format=format, scaled=True, wrap=True)
    new_atoms = read(file)

    assert not np.allclose(Si.positions, new_atoms.positions)
    Si.wrap()
    assert np.allclose(Si.positions, new_atoms.positions)


def test_constraints_Si(Si):
    """Test that non-parmetric constraints are written and read in properly"""
    Si.set_constraint([FixAtoms(indices=[0]), FixCartesian(1, [1, 0, 1])])
    Si.write(file, format=format, scaled=True, wrap=False)
    new_atoms = read(file)
    assert np.allclose(Si.positions, new_atoms.positions)
    assert len(Si.constraints) == len(new_atoms.constraints)
    assert str(Si.constraints[0]) == str(new_atoms.constraints[0])
    assert str(Si.constraints[1]) == str(new_atoms.constraints[1])


def test_cartesian_H2O(H2O):
    """write cartesian coords and check if structure was preserved for
    molecular systems"""
    H2O.write(file, format=format)
    new_atoms = read(file)
    assert np.allclose(H2O.positions, new_atoms.positions)


def test_scaled_H2O(H2O):
    """Attempt to write fractional coordinates and see if scaled is set to
    False and can be written properly"""
    with pytest.raises(
        ValueError,
        match="Requesting scaled for a calculation where scaled=True, "
            "but the system is not periodic",
    ):
        H2O.write(file, format=format, scaled=True, wrap=False)


def test_param_const_H2O(H2O):
    """Check to ensure if geo_constrain is True it does not affect the
    final geometry.in file for molecular systems"""
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        # Attempt to write a molecular system with geo_constrain=True
        H2O.write(file, geo_constrain=True)
        assert len(w) == 1
        assert (
            str(w[-1].message)
            == "Parameteric constraints can only be used in periodic systems."
        )

    new_atoms = read(file)
    assert np.allclose(H2O.positions, new_atoms.positions)


def test_velocities_H2O(H2O):
    """Confirm that the velocities are passed to the geometry.in file and
    can be read back in"""
    velocities = [(1.0, 0.0, 0.0), (-1.0, 1.0, 0.0), (0.0, 0.0, 0.0)]
    H2O.set_velocities(velocities)
    H2O.write(file, format=format, scaled=False, write_velocities=True)
    new_atoms = read(file)
    assert np.allclose(H2O.positions, new_atoms.positions)
    assert np.allclose(H2O.get_velocities(), new_atoms.get_velocities())


def test_info_str(H2O):
    """Confirm that the passed info_str is passed to the geometry.in file"""
    H2O.write(file, format=format, info_str="TEST INFO STR")
    with open(file) as fd:
        geometry_lines = fd.readlines()
        print("".join(geometry_lines))
        assert "# Additional information:" in geometry_lines[3]
        assert "# TEST INFO STR" in geometry_lines[4]
