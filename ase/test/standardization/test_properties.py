"""Tests for `__call__`."""

from ase.build import bulk
from ase.calculators.emt import EMT


def test_calc_properties() -> None:
    """Test `__call__`."""
    # XXX Test other calculators.
    # Or maybe this should just be a generic test, not standardization?
    # We'll see.
    atoms = bulk('Au', cubic=True)
    calc = EMT()
    props = calc(atoms, ['energy', 'stress', 'forces'])

    natoms = len(atoms)
    assert props['stress'].shape == (6,)
    assert props['forces'].shape == (natoms, 3)
    assert isinstance(props['energy'], float)
