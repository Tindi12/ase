import pytest

import ase
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
def atoms() -> ase.Atoms:
    return bulk('Cu', 'fcc', a=3.6)
