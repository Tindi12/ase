import pytest

from ase import Atoms as V3Atoms
from ase._4.atoms import Atoms as V4Atoms
from ase._4.calculators.calculator import BaseCalculator
from ase._4.calculators.emt import EMT
from ase.build import bulk


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
