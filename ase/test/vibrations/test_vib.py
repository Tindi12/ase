# fmt: off
"""Test the ase.vibrations.Vibrations object using a harmonic calculator."""
import os
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

try:
    from numpy.exceptions import ComplexWarning  # NumPy 2.0.0
except ImportError:
    from numpy import ComplexWarning  # type: ignore[attr-defined,no-redef]

import ase.io
from ase import Atoms, units
from ase.build import add_adsorbate, fcc111
from ase.calculators.calculator import compare_atoms
from ase.calculators.qmmm import ForceConstantCalculator
from ase.constraints import FixAtoms, FixCartesian
from ase.thermochemistry import IdealGasThermo
from ase.vibrations import Vibrations, VibrationsData


@pytest.fixture()
def random_dimer():
    rng = np.random.RandomState(42)

    d = 1 + 0.5 * rng.random()
    z_values = rng.randint(1, high=50, size=2)

    hessian = rng.random((6, 6))
    hessian += hessian.T  # Ensure the random Hessian is symmetric

    atoms = Atoms(z_values, [[0, 0, 0], [0, 0, d]])
    ref_atoms = atoms.copy()
    atoms.calc = ForceConstantCalculator(D=hessian,
                                         ref=ref_atoms,
                                         f0=np.zeros((2, 3)))
    return atoms


def test_harmonic_vibrations(testdir):
    """Check the numerics with a trivial case: one atom in harmonic well"""
    rng = np.random.RandomState(42)

    k = rng.random()

    ref_atoms = Atoms('H', positions=np.zeros([1, 3]))
    atoms = ref_atoms.copy()
    mass = atoms.get_masses()[0]

    atoms.calc = ForceConstantCalculator(D=np.eye(3) * k,
                                         ref=ref_atoms,
                                         f0=np.zeros((1, 3)))
    vib = Vibrations(atoms, name='harmonic')
    vib.run()
    vib.read()

    expected_energy = (units._hbar  # In J/s
                       * np.sqrt(k  # In eV/A^2
                                 * units._e  # eV -> J
                                 * units.m**2  # A^-2 -> m^-2
                                 / mass  # in amu
                                 / units._amu  # amu^-1 -> kg^-1
                                 )
                       ) / units._e  # J/s -> eV/s

    assert np.allclose(vib.get_energies(), expected_energy)


def test_frederiksen(testdir, random_dimer):
    # Apply appropriate symmetry to non-"self" terms so that Frederiksen result
    # is not modified by translational symmetrisation step:
    #
    # - We have a 6x6 matrix for the dimer force-constants
    # - On-diagonal 3x3 blocks should by modified by correction
    # - These blocks need to have translational symmetry after correction
    # - So we impose that symmetry on off-diagonal blocks used in correction
    random_dimer.calc.D[:3, 3:] += random_dimer.calc.D[:3, 3:].T
    random_dimer.calc.D[3:, :3] += random_dimer.calc.D[3:, :3].T

    vib = Vibrations(random_dimer, delta=1e-2, nfree=2)
    vib.run()
    vib_data_std = vib.get_vibrations(read_cache=False, method='standard')
    vib_data_frd = vib.get_vibrations(read_cache=False, method='frederiksen')

    # Check Frederiksen option made a difference
    assert not np.allclose(vib_data_std.get_hessian(),
                           vib_data_frd.get_hessian())

    # Check sum rule was violated by original Hessian
    assert not np.allclose(vib_data_std.get_hessian()[0, :, 0, :],
                           -vib_data_std.get_hessian()[0, :, 1, :])

    # And is fixed by Frederiksen scheme
    assert_array_almost_equal(vib_data_frd.get_hessian()[0, :, 0, :],
                              -vib_data_frd.get_hessian()[0, :, 1, :])


def test_consistency_with_vibrationsdata(testdir, random_dimer):
    vib = Vibrations(random_dimer, delta=1e-6, nfree=4)
    vib.run()
    vib_data = vib.get_vibrations()

    assert_array_almost_equal(vib.get_energies(),
                              vib_data.get_energies())

    for mode_index in range(3 * len(vib.atoms)):
        assert_array_almost_equal(vib.get_mode(mode_index),
                                  vib_data.get_modes()[mode_index])

    # Hessian should be close to the ForceConstantCalculator input
    assert_array_almost_equal(random_dimer.calc.D,
                              vib_data.get_hessian_2d(),
                              decimal=6)


def test_json_manipulation(testdir, random_dimer):
    vib = Vibrations(random_dimer, name='interrupt')
    vib.run()

    disp_file = Path('interrupt/cache.1x-.json')
    comb_file = Path('interrupt/combined.json')
    assert disp_file.is_file()
    assert not comb_file.is_file()

    # Should do nothing harmful as files are already split
    # (It used to raise an error but this is no longer implemented.)
    vib.split()

    # Build a combined file
    assert vib.combine() == 13

    # Individual displacements should be gone, combination should exist
    assert not disp_file.is_file()
    assert comb_file.is_file()

    # Not allowed to run after data has been combined
    with pytest.raises(RuntimeError):
        vib.run()
    # But reading is allowed
    vib.read()

    # Splitting should fail if any split file already exists
    with open(disp_file, 'w') as fd:
        fd.write("hello")

    with pytest.raises(AssertionError):
        vib.split()

    os.remove(disp_file)

    # Now split() for real: replace .all.json file with displacements
    vib.split()
    assert disp_file.is_file()
    assert not comb_file.is_file()


def test_vibrations_methods(testdir, random_dimer):
    vib = Vibrations(random_dimer)
    vib.run()
    vib_energies = vib.get_energies()

    for image in vib.iterimages():
        assert len(image) == 2

    thermo = IdealGasThermo(vib_energies=vib_energies, geometry='linear',
                            atoms=vib.atoms, symmetrynumber=2, spin=0)
    thermo.get_gibbs_energy(temperature=298.15, pressure=2 * 101325.,
                            verbose=False)

    logfilename = 'vib.log'

    with open(logfilename, 'w') as fd:
        vib.summary(log=fd)

    with open(logfilename) as fd:
        log_txt = fd.read()
        assert log_txt == '\n'.join(
            VibrationsData._tabulate_from_energies(vib_energies)) + '\n'

    last_mode = vib.get_mode(-1)
    scale = 0.5
    assert_array_almost_equal(vib.show_as_force(-1, scale=scale,
                                                show=False).get_forces(),
                              last_mode * 3 * len(vib.atoms) * scale)

    vib.write_mode(n=3, nimages=5)
    for i in range(3):
        assert not Path(f'vib.{i}.traj').is_file()
    mode_traj = ase.io.read('vib.3.traj', index=':')
    assert len(mode_traj) == 5

    assert_array_almost_equal(mode_traj[0].get_all_distances(),
                              random_dimer.get_all_distances())
    with pytest.raises(AssertionError):
        assert_array_almost_equal(mode_traj[4].get_all_distances(),
                                  random_dimer.get_all_distances())

    assert vib.clean(empty_files=True) == 0
    assert vib.clean() == 13
    assert len(list(vib.iterimages())) == 13

    d = dict(vib.iterdisplace(inplace=False))

    for name, image in vib.iterdisplace(inplace=True):
        assert d[name] == random_dimer


def test_vibrations_restart_dir(testdir, random_dimer):
    vib = Vibrations(random_dimer)
    vib.run()
    freqs = vib.get_frequencies()
    assert freqs is not None

    # write/read the data from another working directory
    atoms = random_dimer.copy()  # This copy() removes the Calculator

    with ase.utils.workdir('run_from_here', mkdir=True):
        vib = Vibrations(atoms, name=str(Path.cwd().parent / 'vib'))
        assert_array_almost_equal(freqs, vib.get_frequencies())
        assert vib.clean() == 13


class TestVibrationsDataStaticMethods:
    @pytest.mark.parametrize('mask,expected_indices',
                             [([True, True, False, True], [0, 1, 3]),
                              ([False, False], []),
                              ([], []),
                              (np.array([True, True]), [0, 1]),
                              (np.array([False, True, True]), [1, 2]),
                              (np.array([], dtype=bool), [])])
    def test_indices_from_mask(self, mask, expected_indices):
        assert VibrationsData.indices_from_mask(mask) == expected_indices

    @pytest.mark.parametrize('fixed_indices,expected_indices',
                             [(2, [0, 1, 3, 4]),
                              ([0, 1, 2, 3, 4], []),
                              ([], [0, 1, 2, 3, 4]),
                              ([0, 1], [2, 3, 4])])
    def test_indices_from_constraint(self, fixed_indices, expected_indices):
        atoms = Atoms('CH4', constraint=FixAtoms(fixed_indices))
        cartesian = Atoms('CH4', constraint=FixCartesian(fixed_indices))
        assert VibrationsData.indices_from_constraints(
            atoms) == expected_indices
        assert VibrationsData.indices_from_constraints(
            cartesian) == expected_indices

    def test_tabulate_energies(self):
        # Test the private classmethod _tabulate_from_energies
        # used by public tabulate() method
        energies = np.array([1., complex(2., 1.), complex(1., 1e-3)])

        table = VibrationsData._tabulate_from_energies(energies, im_tol=1e-2)

        for sep_row in 0, 2, 6:
            assert table[sep_row] == '-' * 21
        assert tuple(table[1].strip().split()) == ('#', 'meV', 'cm^-1')

        expected_rows = [
            # energy in eV should be converted to meV and cm-1
            ('0', '1000.0', '8065.5'),
            # Imaginary component over threshold detected
            ('1', '1000.0i', '8065.5i'),
            # Small imaginary component ignored
            ('2', '1000.0', '8065.5')]

        for row, expected in zip(table[3:6], expected_rows):
            assert tuple(row.split()) == expected

        # ZPE = (1 + 2 + 1) / 2  - currently we keep all real parts
        assert table[7].split()[2] == '2.000'
        assert len(table) == 8

    na2 = Atoms('Na2', cell=[2, 2, 2], positions=[[0, 0, 0],
                                                  [1, 1, 1]])
    na2_image_1 = na2.copy()
    na2_image_1.info.update({'mode#': '0',
                             'frequency_cm-1': 8065.5})
    na2_image_1.arrays['mode'] = np.array([[1., 1., 1.],
                                           [0.5, 0.5, 0.5]])

    @pytest.mark.parametrize('kwargs,expected',
                             [(dict(atoms=na2,
                                    energies=[1.],
                                    modes=np.array([[[1., 1., 1.],
                                                     [0.5, 0.5, 0.5]]])),
                               [na2_image_1])
                              ])
    def test_get_jmol_images(self, kwargs, expected):
        # Test the private staticmethod _get_jmol_images
        # used by the public write_jmol_images() method

        jmol_images = list(VibrationsData._get_jmol_images(**kwargs))
        assert len(jmol_images) == len(expected)

        for image, reference in zip(jmol_images, expected):
            assert compare_atoms(image, reference) == []
            for key, value in reference.info.items():
                if key == 'frequency_cm-1':
                    assert float(image.info[key]) == pytest.approx(value,
                                                                   abs=0.1)
                else:
                    assert image.info[key] == value


@pytest.fixture(name="n3_data")
def fixture_n3_data():
    return {
        "atoms": Atoms(
            "N3",
            positions=[
                [-2.38718534e-20, -1.37505272e-18, -9.52602675e-02],
                [4.19083314e-18, 1.20967172e-21, 1.00000000e00],
                [-2.98538406e-20, -6.14940024e-21, 2.09526027e00],
            ],
        ),
        "hessian": np.array(
            [
                [
                    [
                        [-0.051152827404534265, 0.0, 4.515076384965933e-17],
                        [0.10285317614875701, 0.0, -3.4041960635573854e-17],
                        [-0.05180366946913996, 0.0, -1.1112375713637364e-17],
                    ],
                    [
                        [0.0, -0.051152827404534265, 1.4805479027902448e-17],
                        [0.0, 0.10285317614875701, -1.111600225632703e-17],
                        [0.0, -0.05180366946913996, -3.6906433293276164e-18],
                    ],
                    [
                        [
                            4.515076384965933e-17,
                            1.4805479027902448e-17,
                            23.6056441110992,
                        ],
                        [
                            -5.627945322888836e-17,
                            -1.4762943002566173e-17,
                            -17.58123953472097,
                        ],
                        [
                            1.112868937922907e-17,
                            -4.253602533629244e-20,
                            -6.025595710254095,
                        ],
                    ],
                ],
                [
                    [
                        [0.10285317614875701, 0.0, -5.627945322888836e-17],
                        [-0.20549971084767998, 0.0, -4.8316210069774704e-20],
                        [0.10285317614875701, 0.0, 5.632776436845992e-17],
                    ],
                    [
                        [0.0, 0.10285317614875701, -1.4762943002566173e-17],
                        [0.0, -0.20549971084767998, 1.1056563401597684e-17],
                        [0.0, 0.10285317614875701, 3.707539920969443e-18],
                    ],
                    [
                        [
                            -3.4041960635573854e-17,
                            -1.111600225632703e-17,
                            -17.58123953472097,
                        ],
                        [
                            -4.8316210069774704e-20,
                            1.1056563401597684e-17,
                            35.16486133719432,
                        ],
                        [
                            3.409027684564367e-17,
                            5.9438854729351e-20,
                            -17.581239534721107,
                        ],
                    ],
                ],
                [
                    [
                        [-0.05180366946913996, 0.0, 1.112868937922907e-17],
                        [0.10285317614875701, 0.0, 3.409027684564367e-17],
                        [-0.051152827404534265, 0.0, -4.5215388654822585e-17],
                    ],
                    [
                        [0.0, -0.05180366946913996, -4.253602533629244e-20],
                        [0.0, 0.10285317614875701, 5.9438854729351e-20],
                        [0.0, -0.051152827404534265, -1.68965916418262e-20],
                    ],
                    [
                        [
                            -1.1112375713637364e-17,
                            -3.6906433293276164e-18,
                            -6.025595710254095,
                        ],
                        [
                            5.632776436845992e-17,
                            3.707539920969443e-18,
                            -17.581239534721107,
                        ],
                        [
                            -4.5215388654822585e-17,
                            -1.68965916418262e-20,
                            23.605644111098822,
                        ],
                    ],
                ],
            ]
        ),
        "ref_frequencies": np.array(
            [
                77.38456049982454j,
                77.3845604998238j,
                0.032317878013440174j,
                (0.03665752592158175 + 0j),
                (0.036657527792466664 + 0j),
                (3.5546375249195252 + 0j),
                (3.554637524920577 + 0j),
                (758.4595533374285 + 0j),
                (1011.9237550816223 + 0j),
            ]
        ),
        "ref_zpe": 0.11019504062377428,
        "ref_forces": np.array(
            [
                [
                    -6.755811053078452e-17,
                    -2.7326135643188342e-17,
                    -0.19633815231481458,
                ],
                [
                    4.209280606671509e-17,
                    2.722793972467623e-17,
                    0.39270290915945033,
                ],
                [
                    -2.6438967240289884e-17,
                    1.537622420098163e-19,
                    -0.19633815231481405,
                ],
            ]
        ),
    }


@pytest.fixture(name='n3_vibdata')
def fixture_n3_vibdata(n3_data, indices):
    # Trim hessian according to contributing atoms
    # This is only consistent if no constraints are set
    if indices is not None:
        h = n3_data['hessian'][indices][..., indices, :]
        n3_data['hessian'] = h

    return VibrationsData(
        n3_data['atoms'],
        n3_data['hessian'],
        indices=indices,
    )


@pytest.fixture(name="n3_unstable_data")
def fixture_n3_unstable_data():
    n3_unstable_data = {
        "atoms": Atoms(
            "CO2",
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
        ),
        "hessian": np.array(
            [
                [
                    [
                        [-3.45507633e00, 0.00000000e00, 0.00000000e00],
                        [3.46346573e00, 0.00000000e00, 0.00000000e00],
                        [-8.51681929e-03, 0.00000000e00, 0.00000000e00],
                    ],
                    [
                        [0.00000000e00, -3.45507633e00, 0.00000000e00],
                        [0.00000000e00, 3.46346573e00, 0.00000000e00],
                        [0.00000000e00, -8.51681929e-03, 0.00000000e00],
                    ],
                    [
                        [0.00000000e00, 0.00000000e00, 3.64109314e01],
                        [0.00000000e00, 0.00000000e00, -2.93049847e01],
                        [0.00000000e00, 0.00000000e00, -7.10725807e00],
                    ],
                ],
                [
                    [
                        [3.46346573e00, 0.00000000e00, 0.00000000e00],
                        [-6.92667661e00, 0.00000000e00, 0.00000000e00],
                        [3.46346573e00, 0.00000000e00, 0.00000000e00],
                    ],
                    [
                        [0.00000000e00, 3.46346573e00, 0.00000000e00],
                        [0.00000000e00, -6.92667661e00, 0.00000000e00],
                        [0.00000000e00, 3.46346573e00, 0.00000000e00],
                    ],
                    [
                        [0.00000000e00, 0.00000000e00, -2.93049847e01],
                        [0.00000000e00, 0.00000000e00, 5.86125920e01],
                        [0.00000000e00, 0.00000000e00, -2.93049847e01],
                    ],
                ],
                [
                    [
                        [-8.51681929e-03, 0.00000000e00, 0.00000000e00],
                        [3.46346573e00, 0.00000000e00, 0.00000000e00],
                        [-3.45507633e00, 0.00000000e00, 0.00000000e00],
                    ],
                    [
                        [0.00000000e00, -8.51681929e-03, 0.00000000e00],
                        [0.00000000e00, 3.46346573e00, 0.00000000e00],
                        [0.00000000e00, -3.45507633e00, 0.00000000e00],
                    ],
                    [
                        [0.00000000e00, 0.00000000e00, -7.10725807e00],
                        [0.00000000e00, 0.00000000e00, -2.93049847e01],
                        [0.00000000e00, 0.00000000e00, 3.64109314e01],
                    ],
                ],
            ]
        ),
    }

    return n3_unstable_data


@pytest.fixture(name='indices', params=[None])
def fixture_indices(request):
    return request.param


def test_init(n3_data):
    # Check that init runs without error; properties are checked in other
    # methods using the (identical) n2_vibdata fixture
    VibrationsData(n3_data['atoms'], n3_data['hessian'])


def test_energies_and_modes(n3_data, n3_vibdata):
    energies, _ = n3_vibdata.get_energies_and_modes()
    assert_array_almost_equal(n3_data['ref_frequencies'],
                              energies / units.invcm,
                              decimal=5)
    assert_array_almost_equal(n3_data['ref_frequencies'],
                              n3_vibdata.get_energies() / units.invcm,
                              decimal=5)
    assert_array_almost_equal(n3_data['ref_frequencies'],
                              n3_vibdata.get_frequencies(),
                              decimal=5)

    assert (n3_vibdata.get_zero_point_energy()
            == pytest.approx(n3_data['ref_zpe']))

    assert n3_vibdata.tabulate() == (
        '\n'.join(VibrationsData._tabulate_from_energies(energies)) + '\n')

    atoms_with_forces = n3_vibdata.show_as_force(-1, show=False)

    try:
        assert_array_almost_equal(atoms_with_forces.get_forces(),
                                  n3_data['ref_forces'])
    except AssertionError:
        # Eigenvectors may be off by a sign change, which is allowed
        assert_array_almost_equal(atoms_with_forces.get_forces(),
                                  -n3_data['ref_forces'])


def test_imaginary_energies(n3_unstable_data):
    vib_data = VibrationsData(n3_unstable_data['atoms'],
                              n3_unstable_data['hessian'])

    assert vib_data.tabulate() == (
        '\n'.join(VibrationsData._tabulate_from_energies(
            vib_data.get_energies()))
        + '\n')


def test_zero_mass(n3_data):
    atoms = n3_data['atoms']
    atoms.set_masses([0., 1., 1.])
    vib_data = VibrationsData(atoms, n3_data['hessian'])
    with pytest.raises(ValueError):
        vib_data.get_energies_and_modes()


def test_new_mass(n3_data, n3_vibdata):
    original_masses = n3_vibdata.get_atoms().get_masses()
    new_masses = original_masses * 3
    new_vib_data = n3_vibdata.with_new_masses(new_masses)
    assert_array_almost_equal(new_vib_data.get_atoms().get_masses(),
                              new_masses)
    assert_array_almost_equal(n3_vibdata.get_energies() / np.sqrt(3),
                              new_vib_data.get_energies())


def test_fixed_atoms(n3_data):
    vib_data = VibrationsData(n3_data['atoms'],
                              n3_data['hessian'][1:, :, 1:, :],
                              indices=[1, 2])
    assert vib_data.get_indices().tolist() == [1, 2]
    assert vib_data.get_mask().tolist() == [False, True, True]


def test_constrained_atoms(n3_data):
    tmpAtoms = n3_data['atoms'].copy()
    tmpAtoms.set_constraint(FixAtoms(0))
    vib_data = VibrationsData(tmpAtoms,
                              n3_data['hessian'][1:, :, 1:, :])
    assert vib_data.get_indices().tolist() == [1, 2]
    assert vib_data.get_mask().tolist() == [False, True, True]


def test_dos(n3_vibdata):
    with pytest.warns(ComplexWarning):
        dos = n3_vibdata.get_dos()
        real_energies = np.array(n3_vibdata.get_energies(), dtype=float)
    assert_array_almost_equal(dos.get_energies(),
                              real_energies)


def test_pdos(n3_vibdata):
    with pytest.warns(ComplexWarning):
        pdos = n3_vibdata.get_pdos()
        real_energies = np.array(n3_vibdata.get_energies(), dtype=float)
    assert_array_almost_equal(pdos[0].get_energies(),
                              real_energies)
    assert_array_almost_equal(pdos[1].get_energies(),
                              real_energies)
    assert_array_almost_equal(pdos[2].get_energies(),
                              real_energies)
    # 3N states = 9, divided equally over three N atoms = 3.0
    assert sum(pdos[0].get_weights()) == pytest.approx(3.0)


class TestDictMethods:
    @staticmethod
    @pytest.fixture(name="indices", params=[[], None, [0], [0, 1]])
    def fixture_indices(request):
        return request.param

    @staticmethod
    def test_todict(n3_data, n3_vibdata):
        vib_data_dict = n3_vibdata.todict()

        vib_data_dict['indices'] == n3_vibdata._indices
        assert_array_almost_equal(vib_data_dict['atoms'].positions,
                                n3_data['atoms'].positions)
        assert_array_almost_equal(vib_data_dict['hessian'],
                                n3_data['hessian'])

    @staticmethod
    def test_dict_roundtrip(n3_vibdata):
        vib_data_dict = n3_vibdata.todict()
        vib_data_roundtrip = VibrationsData.fromdict(vib_data_dict)

        for getter in ('get_atoms',):
            assert (getattr(n3_vibdata, getter)()
                    == getattr(vib_data_roundtrip, getter)())
        for array_getter in ('get_hessian', 'get_hessian_2d',
                            'get_mask', 'get_indices'):
            assert_array_almost_equal(
                getattr(n3_vibdata, array_getter)(),
                getattr(vib_data_roundtrip, array_getter)())

    @staticmethod
    @pytest.mark.parametrize('indices, expected_mask',
                            [([1], [False, True, False]),
                            (None, [True, True, True])])
    def test_dict_indices(n3_vibdata, indices, expected_mask):
        vib_data_dict = n3_vibdata.todict()
        vib_data_dict['indices'] = indices
        vib_data_fromdict = VibrationsData.fromdict(vib_data_dict)
        assert_array_almost_equal(vib_data_fromdict.get_mask(), expected_mask)


def test_jmol_roundtrip(testdir, n3_data):
    ir_intensities = np.random.RandomState(42).rand(9)

    vib_data = VibrationsData(n3_data['atoms'], n3_data['hessian'])
    jmol_file = 'vib-data.xyz'
    vib_data.write_jmol(jmol_file, ir_intensities=ir_intensities)

    images = ase.io.read(jmol_file, index=':')
    for i, image in enumerate(images):
        assert_array_almost_equal(image.positions,
                                  vib_data.get_atoms().positions)
        assert (image.info['IR_intensity']
                == pytest.approx(ir_intensities[i]))
        assert_array_almost_equal(image.arrays['mode'],
                                  vib_data.get_modes()[i])


def test_bad_hessian(n3_data):
    bad_hessians = (None, 'fish', 1,
                    np.array([1, 2, 3]),
                    np.eye(6),
                    np.array([[[1, 0, 0]],
                              [[0, 0, 1]]]))

    for bad_hessian in bad_hessians:
        with pytest.raises(ValueError):
            VibrationsData(n3_data['atoms'], bad_hessian)


def test_bad_hessian2d(n3_data):
    bad_hessians = (None, 'fish', 1,
                    np.array([1, 2, 3]),
                    n3_data['hessian'],
                    np.array([[[1, 0, 0]],
                              [[0, 0, 1]]]))

    for bad_hessian in bad_hessians:
        with pytest.raises(ValueError):
            VibrationsData.from_2d(n3_data['atoms'], bad_hessian)


def test_vibration_on_surface(testdir):
    "N2 above Ag slab - vibration with frozen molecules"
    ag_slab = fcc111('Ag', (4, 4, 2), a=2)
    n2 = Atoms('N2', positions=[[0., 0., 0.],
                                [0., np.sqrt(2), np.sqrt(2)]])
    add_adsorbate(ag_slab, n2, height=1, position='fcc')

    # Add an interaction between the N atoms
    hessian_bottom_corner = np.zeros((2, 3, 2, 3))
    hessian_bottom_corner[-1, :, -2] = [1, 1, 1]
    hessian_bottom_corner[-2, :, -1] = [1, 1, 1]

    hessian = np.zeros((34, 3, 34, 3))
    hessian[32:, :, 32:, :] = hessian_bottom_corner

    ag_slab.calc = ForceConstantCalculator(hessian.reshape((34 * 3,
                                                            34 * 3)),
                                           ref=ag_slab.copy(),
                                           f0=np.zeros((34, 3)))

    # Check that Vibrations with restricted indices returns correct Hessian
    vibs = Vibrations(ag_slab, indices=[-2, -1])
    vibs.run()
    vibs.read()
    assert len(vibs.get_frequencies()) == 6

    assert_array_almost_equal(vibs.get_vibrations().get_hessian(),
                              hessian_bottom_corner)

    # These should blow up if the vectors don't match number of atoms
    vibs.summary()
    vibs.write_jmol()

    for i in range(6):
        # Frozen atoms should have zero displacement
        assert_array_almost_equal(vibs.get_mode(i)[0], [0., 0., 0.])

        # The N atoms should have finite displacement
        assert np.all(np.any(vibs.get_mode(i)[-2:, :], axis=1))

    # Check that FixAtoms works in the same way
    ag_slab_fixed = ag_slab.copy()
    ag_slab_fixed.calc = ForceConstantCalculator(
        hessian.reshape((34 * 3, 34 * 3)), ref=ag_slab.copy(),
        f0=np.zeros((34, 3))
    )
    ag_slab_fixed.set_constraint(
        FixAtoms(mask=[True] * (len(ag_slab_fixed) - 2) + [False] * 2))
    vibs_fixed_atoms = Vibrations(ag_slab_fixed)
    vibs_fixed_atoms.run()
    vibs_fixed_atoms.read()
    assert_array_equal(vibs_fixed_atoms.indices, np.array([32, 33]))
    assert len(vibs_fixed_atoms.get_frequencies()) == 6
    assert_array_almost_equal(
        vibs.get_frequencies(), vibs_fixed_atoms.get_frequencies()
    )

    # Check that we respect the user
    vibs_fixed_atoms2 = Vibrations(ag_slab_fixed, indices=[0])
    vibs_fixed_atoms2.run()
    vibs_fixed_atoms2.read()
    assert_array_equal(vibs_fixed_atoms2.indices, np.array([0]))
    assert len(vibs_fixed_atoms2.get_frequencies()) == 3
