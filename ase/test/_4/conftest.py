import pytest


def pytest_collection_modifyitems(config, items):
    """By default skips tests with "ase_v4" mark."""
    run_v4 = config.option.markexpr and 'ase_v4' in config.option.markexpr
    skip_marker = pytest.mark.skip(reason="ASEv4 test, run with '-m ase_v4'")

    for item in items:
        if '_4' in str(item.fspath):
            if not run_v4:
                item.add_marker(skip_marker)
