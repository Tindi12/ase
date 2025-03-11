# flake8: noqa
from pathlib import Path

import numpy as np
import pytest
pyfhiaims = pytest.importorskip("pyfhiaims")

from ase.io import ParseError, read
from ase.io.aims import read_aims_results, read_aims_output
from ase.stress import full_3x3_to_voigt_6_stress

parent = Path(__file__).parents[2]


def test_parse_out(testdir):
    traj = read(parent / "testdata/aims/c_relax.out", ":", format="aims-output")
    image_0 = read_aims_output(parent / "testdata/aims/c_relax.out", 0)
    image_1 = read_aims_output(parent / "testdata/aims/c_relax.out")

    assert len(traj) == 2
    assert traj[0] == image_0
    assert traj[-1] == image_1

    p0 = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
    assert all(np.allclose(at.get_scaled_positions(), p0) for at in traj)
    assert np.allclose(image_0.get_scaled_positions(), p0)
    assert np.allclose(image_1.get_scaled_positions(), p0)

    assert all(np.allclose(at.get_forces(), np.zeros((2, 3))) for at in traj)
    assert np.allclose(image_0.get_forces(), np.zeros((2, 3)))
    assert np.allclose(image_1.get_forces(), np.zeros((2, 3)))

    s0 = full_3x3_to_voigt_6_stress(
        [
            [0.02056358, 0.0, 0.0],
            [0.0, 0.02056358, 0.0],
            [0.0, 0.0, 0.02056358],
        ]
    )
    s_end = full_3x3_to_voigt_6_stress(
        [
            [0.00029342, 0.0, 0.0],
            [0.0, 0.00029342, 0.0],
            [0.0, 0.0, 0.00029342],
        ]
    )
    assert np.allclose(traj[0].get_stress(), s0)
    assert np.allclose(image_0.get_stress(), s0)

    assert np.allclose(traj[-1].get_stress(), s_end)
    assert np.allclose(image_1.get_stress(), s_end)

    cell_0 = [
        [0.000, 1.786072033467, 1.786072033467],
        [1.786072033467, 0.000, 1.786072033467],
        [1.786072033467, 1.786072033467, 0.000],
    ]
    cell_end = [
        [0.000, 1.78172069, 1.78172069],
        [1.78172069, 0.000, 1.78172069],
        [1.78172069, 1.78172069, 0.000],
    ]

    assert np.allclose(traj[0].get_cell(), cell_0)
    assert np.allclose(image_0.get_cell(), cell_0)

    assert np.allclose(traj[-1].get_cell(), cell_end)
    assert np.allclose(image_1.get_cell(), cell_end)

def test_parse_results(testdir):
    traj = read_aims_results(parent / "testdata/aims/c_relax.out", slice(None))
    image_0 = read_aims_results(parent / "testdata/aims/c_relax.out", 0)
    image_1 = read_aims_results(parent / "testdata/aims/c_relax.out")

    results_0 = {
        "forces": np.zeros((2, 3)),
        "stress": np.array(
            [
                [0.02056358, 0.0, 0.0],
                [0.0, 0.02056358, 0.0],
                [0.0, 0.0, 0.02056358],
            ]
        ),
        "total_energy": -0.207521252194984E+04,
        "free_energy": -0.207521252194984E+04,
        "vbm": -8.00379929,
        "cbm": -3.54596603,
        "gap": 4.45783326,
        "direct_gap": 5.87908857,
    }
    results_1 = {
        "forces": np.zeros((2, 3)),
        "stress": np.array(
            [
                [0.00029342, 0.0, 0.0],
                [0.0, 0.00029342, 0.0],
                [0.0, 0.0, 0.00029342],
            ]
        ),
        "total_energy": -0.207521411853867E+04,
        "free_energy": -0.207521411853867E+04,
        "vbm": -8.02644353,
        "cbm": -3.55243630,
        "gap": 4.47400723,
        "direct_gap": 5.89977761,
    }
    assert len(traj) == 2

    for key, val in results_0.items():
        assert np.allclose(val, traj[0][key])
        assert np.allclose(val, image_0[key])

    for key, val in results_1.items():
        assert np.allclose(val, traj[-1][key])
        assert np.allclose(val, image_1[key])
