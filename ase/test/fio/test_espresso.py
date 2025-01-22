"""Quantum ESPRESSO file parsers.

Implemented:
* Input file (pwi)
* Output file (pwo) with vc-relax

"""

import io
from pathlib import Path

import numpy as np
import pytest

import ase.build
from ase import Atoms
from ase.calculators.calculator import compare_atoms
from ase.constraints import FixAtoms, FixCartesian, FixScaled
from ase.io import read
from ase.io.espresso import (
    get_atomic_species,
    read_espresso_in,
    read_fortran_namelist,
    write_espresso_in,
    write_fortran_namelist,
)
from ase.stress import full_3x3_to_voigt_6_stress
from ase.units import create_units

units = create_units('2018')

OUTPUT_DIR = Path(__file__).parent / 'espresso_outputs'

# This file is parsed correctly by pw.x, even though things are
# scattered all over the place with some namelist edge cases
pw_input_text = """
&CONTrol
   prefix           = 'surf_110_H2_md'
   calculation      = 'md'
   restart_mode     = 'from_scratch'
   pseudo_dir       = '.'
   outdir           = './surf_110_!H2_m=d_sc,ratch/'
   verbosity        = 'default'
   tprnfor          = .true.
   tstress          = .True.
!   disk_io          = 'low'
   wf_collect       = .false.
   max_seconds      = 82800
   forc_con!v_thr    = 1e-05
   etot_conv_thr    = 1e-06
   dt               = 41.3 , /

&SYSTEM ecutwfc     = 63,   ecutrho   = 577,  ibrav    = 0,
nat              = 8,   ntyp             = 2,  occupations      = 'smearing',
smearing         = 'marzari-vanderbilt',
degauss          = 0.01,   nspin            = 2,  !  nosym     = .true. ,
    starting_magnetization(2) = 5.12 /
&ELECTRONS
   electron_maxstep = 300
   mixing_beta      = 0.1
   conv_thr         = 1d-07
   mixing_mode      = 'local-TF'
   scf_must_converge = False
/
&IONS
   ion_dynamics     = 'verlet'
   ion_temperature  = 'rescaling'
   tolp             = 50.0
   tempw            = 500.0
/

ATOMIC_SPECIES
H 1.008 H.pbe-rrkjus_psl.0.1.UPF
Fe 55.845 Fe.pbe-spn-rrkjus_psl.0.2.1.UPF

K_POINTS automatic
2 2 2  1 1 1

CELL_PARAMETERS angstrom
5.6672000000000002 0.0000000000000000 0.0000000000000000
0.0000000000000000 8.0146311006808038 0.0000000000000000
0.0000000000000000 0.0000000000000000 27.0219466510212101

ATOMIC_POSITIONS angstrom
Fe 0.0000000000 0.0000000000 0.0000000000 0 0 0
Fe 1.4168000000 2.0036577752 -0.0000000000 0 0 0
Fe 0.0000000000 2.0036577752 2.0036577752 0 0 0
Fe 1.4168000000 0.0000000000 2.0036577752 0 0 0
Fe 0.0000000000 0.0000000000 4.0073155503
Fe 1.4168000000 2.0036577752 4.0073155503
H 0.0000000000 2.0036577752 6.0109733255
H 1.4168000000 0.0000000000 6.0109733255
"""


def test_pw_input():
    """Read pw input file."""
    with open('pw_input.pwi', 'w') as pw_input_f:
        pw_input_f.write(pw_input_text)

    pw_input_atoms = read('pw_input.pwi', format='espresso-in')
    assert len(pw_input_atoms) == 8
    assert pw_input_atoms.get_initial_magnetic_moments() == pytest.approx(
        [5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 0.0, 0.0]
    )


def test_get_atomic_species():
    """Parser for atomic species section"""

    with open('pw_input.pwi', 'w') as pw_input_f:
        pw_input_f.write(pw_input_text)
    with open('pw_input.pwi') as pw_input_f:
        data, card_lines = read_fortran_namelist(pw_input_f)
        species_card = get_atomic_species(
            card_lines, n_species=data['system']['ntyp']
        )

    assert len(species_card) == 2
    assert species_card[0] == (
        'H',
        pytest.approx(1.008),
        'H.pbe-rrkjus_psl.0.1.UPF',
    )
    assert species_card[1] == (
        'Fe',
        pytest.approx(55.845),
        'Fe.pbe-spn-rrkjus_psl.0.2.1.UPF',
    )


def test_pw_input_write():
    """Write a structure and read it back."""
    bulk = ase.build.bulk('NiO', 'rocksalt', 4.813, cubic=True)
    bulk.set_initial_magnetic_moments(
        [2.2 if atom.symbol == 'Ni' else 0.0 for atom in bulk]
    )

    fh = 'espresso_test.pwi'
    pseudos = {'Ni': 'potato', 'O': 'orange'}

    write_espresso_in(fh, bulk, pseudopotentials=pseudos)
    readback = read_espresso_in('espresso_test.pwi')
    assert np.allclose(bulk.positions, readback.positions)

    sections = {
        'system': {'lda_plus_u': True, 'Hubbard_U(1)': 4.0, 'Hubbard_U(2)': 0.0}
    }
    write_espresso_in(
        fh,
        bulk,
        sections,
        pseudopotentials=pseudos,
        additional_cards=['test1', 'test2', 'test3'],
    )

    readback = read_espresso_in('espresso_test.pwi')

    with open('espresso_test.pwi') as f:
        _, cards = read_fortran_namelist(f)

        assert 'K_POINTS gamma' in cards
        assert cards[-3] == 'test1'
        assert cards[-1] == 'test3'

    assert np.allclose(bulk.positions, readback.positions)


def test_pw_input_write_raw_kpts():
    """Write a structure and read it back."""
    bulk = ase.build.bulk('NiO', 'rocksalt', 4.813, cubic=True)
    bulk.set_initial_magnetic_moments(
        [2.2 if atom.symbol == 'Ni' else 0.0 for atom in bulk]
    )

    fh = 'espresso_test.pwi'
    pseudos = {'Ni': 'potato', 'O': 'orange'}
    kpts = np.random.random((10, 4))

    write_espresso_in(fh, bulk, pseudopotentials=pseudos, kpts=kpts)
    readback = read_espresso_in('espresso_test.pwi')
    assert np.allclose(bulk.positions, readback.positions)

    sections = {
        'system': {'lda_plus_u': True, 'Hubbard_U(1)': 4.0, 'Hubbard_U(2)': 0.0}
    }
    write_espresso_in(
        fh,
        bulk,
        sections,
        pseudopotentials=pseudos,
        additional_cards=['test1', 'test2', 'test3'],
        kpts=kpts,
    )

    readback = read_espresso_in('espresso_test.pwi')

    with open('espresso_test.pwi') as f:
        _, cards = read_fortran_namelist(f)

        assert 'K_POINTS crystal' in cards
        assert cards[5].startswith(f'{kpts[0, 0]:.12f}'[:10])
        assert cards[6].startswith(f'{kpts[1, 0]:.12f}'[:10])
        assert cards[-3] == 'test1'
        assert cards[-1] == 'test3'

    assert np.allclose(bulk.positions, readback.positions)


def test_pw_input_write_nested_flat():
    """Write a structure and read it back."""
    bulk = ase.build.bulk('Fe')

    fh = 'espresso_test.pwi'
    pseudos = {'Fe': 'carrot'}

    input_data = {
        'control': {'calculation': 'scf'},
        'unused_keyword1': 'unused_value1',
        'used_sections': {'used_keyword1': 'used_value1'},
    }

    with pytest.raises(DeprecationWarning):
        write_espresso_in(
            fh,
            bulk,
            input_data=input_data,
            pseudopotentials=pseudos,
            mixing_mode='local-TF',
        )

    write_espresso_in(
        fh,
        bulk,
        input_data=input_data,
        pseudopotentials=pseudos,
        unusedkwarg='unused',
    )

    with open(fh) as f:
        new_atoms = read_espresso_in(f)
        f.seek(0)
        readback = read_fortran_namelist(f)

    read_string = readback[0].to_string()

    assert '&USED_SECTIONS\n' in read_string
    assert "   used_keyword1    = 'used_value1'\n" in read_string
    assert np.allclose(bulk.positions, new_atoms.positions)


def test_write_fortran_namelist_any():
    fd = io.StringIO()
    input_data = {
        'environ': {'environ_type': 'vacuum'},
        'electrostatic': {'tol': 1e-10, 'mix': 0.5},
        'boundary': {'solvent_mode': 'full'},
    }

    additional_cards = [
        'EXTERNAL_CHARGES (bohr)',
        '-0.5 0. 0. 25.697 1.0 2 3',
        '-0.5 0. 0. 20.697 1.0 2 3',
    ]

    write_fortran_namelist(fd, input_data, additional_cards=additional_cards)
    result = fd.getvalue()

    expected = (
        '&ENVIRON\n'
        "   environ_type     = 'vacuum'\n"
        '/\n'
        '&ELECTROSTATIC\n'
        '   tol              = 1e-10\n'
        '   mix              = 0.5\n'
        '/\n'
        '&BOUNDARY\n'
        "   solvent_mode     = 'full'\n"
        '/\n'
        'EXTERNAL_CHARGES (bohr)\n'
        '-0.5 0. 0. 25.697 1.0 2 3\n'
        '-0.5 0. 0. 20.697 1.0 2 3\n'
        'EOF'
    )

    assert result == expected
    assert 'ENVIRON' in result
    assert 'ELECTROSTATIC' in result
    assert 'BOUNDARY' in result
    assert result.endswith('EOF')
    fd.seek(0)
    reread = read_fortran_namelist(fd)
    assert reread[1][:-1] == additional_cards
    assert reread[0] == input_data


def test_write_fortran_namelist_pw():
    fd = io.StringIO()
    input_data = {
        'calculation': 'scf',
        'ecutwfc': 30.0,
        'ibrav': 0,
        'nat': 10,
        'nbnd': 8,
        'conv_thr': 1e-6,
        'random': True,
    }
    binary = 'pw'
    write_fortran_namelist(fd, input_data, binary)
    result = fd.getvalue()
    assert 'scf' in result
    assert 'ibrav' in result
    assert 'conv_thr' in result
    assert result.endswith('EOF')
    fd.seek(0)
    reread = read_fortran_namelist(fd)
    assert reread != input_data


def test_write_fortran_namelist_fields():
    fd = io.StringIO()
    input_data = {
        'INPUT': {
            'amass': 28.0855,
            'niter_ph': 50,
            'tr2_ph': 1e-6,
            'flfrc': 'silicon.fc',
        },
    }
    binary = 'q2r'
    write_fortran_namelist(
        fd, input_data, binary, additional_cards='test1\ntest2\ntest3\n'
    )
    result = fd.getvalue()
    expected = (
        '&INPUT\n'
        "   flfrc            = 'silicon.fc'\n"
        '   amass            = 28.0855\n'
        '   niter_ph         = 50\n'
        '   tr2_ph           = 1e-06\n'
        '/\n'
        'test1\n'
        'test2\n'
        'test3\n'
        'EOF'
    )
    assert result == expected


def test_write_fortran_namelist_list_fields():
    fd = io.StringIO()
    input_data = {
        'PRESS_AI': {
            'amass': 28.0855,
            'niter_ph': 50,
            'tr2_ph': 1e-6,
            'flfrc': 'silicon.fc',
        },
    }
    binary = 'cp'
    write_fortran_namelist(
        fd, input_data, binary, additional_cards=['test1', 'test2', 'test3']
    )
    result = fd.getvalue()
    expected = (
        '&CONTROL\n'
        '/\n'
        '&SYSTEM\n'
        '/\n'
        '&ELECTRONS\n'
        '/\n'
        '&IONS\n'
        '/\n'
        '&CELL\n'
        '/\n'
        '&PRESS_AI\n'
        '   amass            = 28.0855\n'
        '   niter_ph         = 50\n'
        '   tr2_ph           = 1e-06\n'
        "   flfrc            = 'silicon.fc'\n"
        '/\n'
        '&WANNIER\n'
        '/\n'
        'test1\n'
        'test2\n'
        'test3\n'
        'EOF'
    )
    assert result == expected


class TestConstraints:
    """Test if the constraint can be recovered when writing and reading.

    Notes
    -----
    Linear constraints in the ATOMIC_POSITIONS block in the quantum ESPRESSO
    `.pwi` format apply to Cartesian coordinates, regardless of whether the
    atomic positions are written in the "angstrom" or the "crystal" units.
    """

    # TODO: test also mask for FixCartesian

    @staticmethod
    def _make_atoms_ref():
        """water molecule"""
        atoms = ase.build.molecule('H2O')
        atoms.cell = 10.0 * np.eye(3)
        atoms.pbc = True
        atoms.set_initial_magnetic_moments(len(atoms) * [0.0])
        return atoms

    def _apply_write_read(self, constraint) -> Atoms:
        atoms_ref = self._make_atoms_ref()
        atoms_ref.set_constraint(constraint)

        pseudopotentials = {
            'H': 'h_lda_v1.2.uspp.F.UPF',
            'O': 'o_lda_v1.2.uspp.F.UPF',
        }
        buf = io.StringIO()
        write_espresso_in(buf, atoms_ref, pseudopotentials=pseudopotentials)
        buf.seek(0)
        atoms = read_espresso_in(buf)

        assert not compare_atoms(atoms_ref, atoms)

        return atoms

    def test_fix_atoms(self):
        """Test FixAtoms"""
        constraint = FixAtoms(indices=(1, 2))
        atoms = self._apply_write_read(constraint)

        assert len(atoms.constraints) == 1
        assert isinstance(atoms.constraints[0], FixAtoms)
        assert all(atoms.constraints[0].index == constraint.index)

    def test_fix_cartesian_line(self):
        """Test FixCartesian along line"""
        # moved only along the z direction
        constraint = FixCartesian(0, mask=(1, 1, 0))
        atoms = self._apply_write_read(constraint)

        assert len(atoms.constraints) == 1
        assert isinstance(atoms.constraints[0], FixCartesian)
        assert all(atoms.constraints[0].index == constraint.index)

    def test_fix_cartesian_plane(self):
        """Test FixCartesian in plane"""
        # moved only in the yz plane
        constraint = FixCartesian((1, 2), mask=(1, 0, 0))
        atoms = self._apply_write_read(constraint)

        assert len(atoms.constraints) == 1
        assert isinstance(atoms.constraints[0], FixCartesian)
        assert all(atoms.constraints[0].index == constraint.index)

    def test_fix_cartesian_multiple(self):
        """Test multiple FixCartesian"""
        constraint = [FixCartesian(1), FixCartesian(2)]
        atoms = self._apply_write_read(constraint)

        assert len(atoms.constraints) == 1
        assert isinstance(atoms.constraints[0], FixAtoms)
        assert atoms.constraints[0].index.tolist() == [1, 2]

    def test_fix_scaled(self):
        """Test FixScaled"""
        constraint = FixScaled(0, mask=(1, 1, 0))
        with pytest.raises(UserWarning):
            self._apply_write_read(constraint)


def test_al001_bc1():
    output_path = OUTPUT_DIR / 'Al001_bc1.out.gz'

    atoms = read(output_path, format='espresso-out')

    num_atoms = 4
    total_energy = -49.51672301 * units['Ry']
    fermi_energy = -4.2462

    assert len(atoms) == num_atoms
    assert atoms.get_potential_energy() == pytest.approx(total_energy)
    assert atoms.calc.get_fermi_level() == pytest.approx(fermi_energy)
    assert np.allclose(atoms.get_forces(), 0.0, atol=1e-5)

    positions = np.array(
        [
            [0.0000000, 0.0000000, 0.0000000],
            [0.5000000, 0.0000000, 0.0000000],
            [0.0000000, 0.5000000, 0.0000000],
            [0.5000000, 0.5000000, 0.0000000],
        ]
    )

    assert np.allclose(atoms.get_scaled_positions(), positions)

    cell = np.array(
        [
            [5.726914, 0.00000, 0.00000],
            [0.00000, 5.726914, 0.00000],
            [0.00000, 0.00000, 12.000027],
        ]
    )

    assert np.allclose(atoms.cell.array, cell)

    kpts = np.array(
        [
            [0.0833333, 0.0833333, 0.0000000],
            [0.0833333, 0.2500000, 0.0000000],
            [0.0833333, 0.4166667, 0.0000000],
            [0.2500000, 0.2500000, 0.0000000],
            [0.2500000, 0.4166667, 0.0000000],
            [0.4166667, 0.4166667, 0.0000000],
        ]
    )

    reader_kpts = np.array([kpts.k for kpts in atoms.calc.kpts])

    assert np.allclose(reader_kpts, kpts, atol=1e-5)

    weights = np.array(
        [0.2222222, 0.4444444, 0.4444444, 0.2222222, 0.4444444, 0.2222222]
    )

    reader_weights = np.array([kpts.weight for kpts in atoms.calc.kpts])

    assert np.allclose(reader_weights, weights)

    bands = np.array(
        [
            # k = 0.0833 0.0833 0.0000
            [
                -11.3467,
                -7.7712,
                -7.7712,
                -6.0626,
                -6.0626,
                -5.3081,
                -4.2853,
                -3.4123,
                -2.1193,
                -2.0726,
            ],
            # k = 0.0833 0.2500 0.0000
            [
                -11.0985,
                -8.9037,
                -7.5270,
                -5.8349,
                -5.3843,
                -5.0704,
                -4.4911,
                -3.9310,
                -3.0029,
                -1.8417,
            ],
            # k = 0.0833 0.4167 0.0000
            [
                -10.6043,
                -9.8688,
                -7.0413,
                -6.3216,
                -5.3834,
                -4.7275,
                -4.5979,
                -3.8990,
                -2.6563,
                -1.3835,
            ],
            # k = 0.2500 0.2500 0.0000
            [
                -10.8508,
                -8.6594,
                -8.6594,
                -6.5048,
                -4.8330,
                -4.2691,
                -4.2691,
                -2.9557,
                -2.7690,
                -2.7689,
            ],
            # k = 0.2500 0.4167 0.0000
            [
                -10.3572,
                -9.6228,
                -8.1730,
                -7.4508,
                -4.3613,
                -3.8390,
                -3.6633,
                -3.2565,
                -2.4566,
                -2.3032,
            ],
            # k = 0.4167 0.4167 0.0000
            [
                -9.8651,
                -9.1333,
                -9.1333,
                -8.4053,
                -3.8903,
                -3.1943,
                -3.1943,
                -2.5008,
                -2.1086,
                -2.0445,
            ],
        ]
    )

    reader_bands = np.array([kpts.eps_n for kpts in atoms.calc.kpts])

    assert np.allclose(reader_bands, bands)


def test_al001_bc2_fcp():
    output_path = OUTPUT_DIR / 'Al001_bc2_FCP_v00.out.gz'
    atoms = read(output_path, format='espresso-out')

    # Test number of atoms and basic properties
    num_atoms = 4
    total_energy = (
        -49.51672315 * units['Ry']
    )  # From "Final grand-energy" in output
    fermi_energy = -4.2503  # From "the Fermi energy is" in output

    assert len(atoms) == num_atoms
    assert atoms.get_potential_energy() == pytest.approx(total_energy)
    assert atoms.calc.get_fermi_level() == pytest.approx(fermi_energy)

    # Test atomic positions (from ATOMIC_POSITIONS section)
    positions = np.array(
        [
            [0.0000000, 0.0000000, 0.0000000],
            [0.5000000, 0.0000000, 0.0000000],
            [0.0000000, 0.5000000, 0.0000000],
            [0.5000000, 0.5000000, 0.0000000],
        ]
    )
    assert np.allclose(atoms.get_scaled_positions(), positions)

    # Test cell parameters (using alat = 10.8223 a.u. from output)
    cell = np.array(
        [
            [5.726914529589178, 0.00000, 0.00000],
            [0.00000, 5.726914529589178, 0.00000],
            [
                0.00000,
                0.00000,
                12.000027805523395,
            ],  # 10.8223 * 2.095374 from output
        ]
    )

    assert np.allclose(atoms.cell.array, cell)

    # Test k-points (from k-points section)
    kpts = np.array(
        [
            [0.0833333, 0.0833333, 0.0000000],
            [0.0833333, 0.2500000, 0.0000000],
            [0.0833333, 0.4166667, 0.0000000],
            [0.2500000, 0.2500000, 0.0000000],
            [0.2500000, 0.4166667, 0.0000000],
            [0.4166667, 0.4166667, 0.0000000],
        ]
    )
    reader_kpts = np.array([kpts.k for kpts in atoms.calc.kpts])
    assert np.allclose(reader_kpts, kpts, atol=1e-5)

    # Test k-point weights
    weights = np.array(
        [0.2222222, 0.4444444, 0.4444444, 0.2222222, 0.4444444, 0.2222222]
    )
    reader_weights = np.array([kpts.weight for kpts in atoms.calc.kpts])
    assert np.allclose(reader_weights, weights)

    # Test band energies (from bands section)
    bands = np.array(
        [
            # k = 0.0833 0.0833 0.0000
            [
                -11.3509,
                -7.7755,
                -7.7755,
                -6.0669,
                -6.0669,
                -5.3118,
                -4.2896,
                -3.4166,
                -2.1236,
                -2.0763,
            ],
            # k = 0.0833 0.2500 0.0000
            [
                -11.1028,
                -8.9080,
                -7.5313,
                -5.8391,
                -5.3886,
                -5.0741,
                -4.4954,
                -3.9353,
                -3.0066,
                -1.8454,
            ],
            # k = 0.0833 0.4167 0.0000
            [
                -10.6086,
                -9.8731,
                -7.0456,
                -6.3259,
                -5.3877,
                -4.7317,
                -4.6016,
                -3.9027,
                -2.6606,
                -1.3872,
            ],
            # k = 0.2500 0.2500 0.0000
            [
                -10.8550,
                -8.6637,
                -8.6637,
                -6.5091,
                -4.8367,
                -4.2734,
                -4.2734,
                -2.9600,
                -2.7727,
                -2.7726,
            ],
            # k = 0.2500 0.4167 0.0000
            [
                -10.3614,
                -9.6271,
                -8.1773,
                -7.4551,
                -4.3649,
                -3.8433,
                -3.6670,
                -3.2608,
                -2.4609,
                -2.3069,
            ],
            # k = 0.4167 0.4167 0.0000
            [
                -9.8694,
                -9.1376,
                -9.1376,
                -8.4096,
                -3.8940,
                -3.1980,
                -3.1980,
                -2.5044,
                -2.1129,
                -2.0488,
            ],
        ]
    )
    reader_bands = np.array([kpts.eps_n for kpts in atoms.calc.kpts])
    assert np.allclose(reader_bands, bands)


def test_arsenic_relax():
    output_path = OUTPUT_DIR / 'As.bfgs500.out.gz'

    atoms = read(output_path, format='espresso-out')

    total_energy = -25.39785255 * units['Ry']
    assert atoms.get_potential_energy() == pytest.approx(total_energy)

    fermi_energy = 13.2092  # eV
    assert atoms.calc.get_fermi_level() == pytest.approx(fermi_energy)

    positions = np.array(
        [
            [0.250000421, 0.250000395, 0.250000395],
            [0.74999958, 0.74999961, 0.74999961],
        ]
    )
    assert np.allclose(atoms.get_scaled_positions(), positions)

    cell = (
        np.array(
            [
                [0.534114296, -0.000000000, 0.747253747],
                [-0.267057062, 0.462556500, 0.747253804],
                [-0.267057062, -0.462556500, 0.747253804],
            ]
        )
        * 7.01033623
        * units['Bohr']
    )
    assert np.allclose(atoms.cell.array, cell)

    forces = (
        np.array(
            [
                [-0.00000000, 0.00000000, -0.00000173],
                [0.00000000, 0.00000000, 0.00000173],
            ]
        )
        * units['Ry']
        / units['Bohr']
    )
    assert np.allclose(atoms.get_forces(), forces)

    # Test stress tensor
    stress = full_3x3_to_voigt_6_stress(
        np.array(
            [
                [0.00340933, -0.00000000, -0.00000000],
                [-0.00000000, 0.00340933, 0.00000000],
                [-0.00000000, 0.00000000, 0.00341208],
            ]
        )
        * units['Ry']
        / units['Bohr'] ** 3
    )
    assert np.allclose(atoms.get_stress(), stress)

    # Test k-points (from k-points section in final SCF)
    kpts = (
        np.array(
            [
                [0.0000000, 0.0000000, 0.1672792],
                [-0.1560215, -0.2702373, 0.2787986],
                [0.3120431, 0.5404745, -0.0557598],
                [0.1560216, 0.2702373, 0.0557597],
                [-0.3120431, 0.0000000, 0.3903181],
                [0.1560216, 0.8107118, 0.0557597],
                [0.0000000, 0.5404745, 0.1672792],
                [0.6240862, 0.0000000, -0.2787987],
                [0.0000000, 0.0000000, 0.5018375],
                [0.4680647, 0.8107118, 0.1672791],
            ]
        )
        / (7.0103 * units['Bohr'])
        @ cell.T
    )
    reader_kpts = np.array([kpt.k for kpt in atoms.calc.kpts])
    assert np.allclose(reader_kpts, kpts, atol=1e-5)

    # Test k-point weights
    weights = np.array(
        [
            0.0625000,
            0.1875000,
            0.1875000,
            0.1875000,
            0.1875000,
            0.3750000,
            0.3750000,
            0.1875000,
            0.0625000,
            0.1875000,
        ]
    )
    reader_weights = np.array([kpt.weight for kpt in atoms.calc.kpts])
    assert np.allclose(reader_weights, weights)

    # Test band energies for first k-point (from bands section in final SCF)
    bands_k1 = np.array(
        [
            -4.8419,
            8.1923,
            10.7715,
            10.7715,
            13.5158,
            17.1689,
            17.1689,
            18.1426,
            18.8970,
        ]
    )  # eV
    reader_bands_k1 = atoms.calc.kpts[0].eps_n
    assert np.allclose(reader_bands_k1, bands_k1)

    all_atoms = read(output_path, format='espresso-out', index=':')

    stresses = (
        np.array(
            [
                [
                    [0.00172500, 0.00000000, 0.00000000],
                    [0.00000000, 0.00172500, 0.00000000],
                    [0.00000000, 0.00000000, 0.00098671],
                ],
                [
                    [0.00332822, 0.00000000, 0.00000000],
                    [0.00000000, 0.00332822, 0.00000000],
                    [0.00000000, 0.00000000, 0.00240156],
                ],
                [
                    [0.00462941, 0.00000000, 0.00000000],
                    [0.00000000, 0.00462941, 0.00000000],
                    [0.00000000, 0.00000000, 0.00472289],
                ],
                [
                    [0.00339932, 0.00000000, 0.00000000],
                    [0.00000000, 0.00339932, 0.00000000],
                    [0.00000000, 0.00000000, 0.00390906],
                ],
                [
                    [0.00314957, 0.00000000, 0.00000000],
                    [0.00000000, 0.00314957, 0.00000000],
                    [0.00000000, 0.00000000, 0.00367669],
                ],
                [
                    [0.00313535, 0.00000000, 0.00000000],
                    [0.00000000, 0.00313535, 0.00000000],
                    [0.00000000, 0.00000000, 0.00360656],
                ],
                [
                    [0.00317873, 0.00000000, 0.00000000],
                    [0.00000000, 0.00317873, 0.00000000],
                    [0.00000000, 0.00000000, 0.00346980],
                ],
                [
                    [0.00340783, 0.00000000, 0.00000000],
                    [0.00000000, 0.00340783, 0.00000000],
                    [0.00000000, 0.00000000, 0.00331114],
                ],
                [
                    [0.00338538, 0.00000000, 0.00000000],
                    [0.00000000, 0.00338538, 0.00000000],
                    [0.00000000, 0.00000000, 0.00339420],
                ],
                [
                    [0.00339683, 0.00000000, 0.00000000],
                    [0.00000000, 0.00339683, 0.00000000],
                    [0.00000000, 0.00000000, 0.00339731],
                ],
                [
                    [0.00340933, 0.00000000, 0.00000000],
                    [0.00000000, 0.00340933, 0.00000000],
                    [0.00000000, 0.00000000, 0.00341208],
                ],
            ]
        )
        * units['Ry']
        / units['Bohr'] ** 3
    )
    stresses = full_3x3_to_voigt_6_stress(stresses)
    reader_stresses = np.array([atoms.get_stress() for atoms in all_atoms])
    assert np.allclose(reader_stresses, stresses)

    cells = (
        np.array(
            [
                [
                    [0.580130, 0.000000, 0.814524],
                    [-0.290065, 0.502407, 0.814524],
                    [-0.290065, -0.502407, 0.814524],
                ],
                [
                    [0.555852086, 0.000000000, 0.765403850],
                    [-0.277925869, 0.481381982, 0.765403856],
                    [-0.277925869, -0.481381982, 0.765403856],
                ],
                [
                    [0.539721503, 0.000000000, 0.701069301],
                    [-0.269860614, 0.467412487, 0.701069329],
                    [-0.269860614, -0.467412487, 0.701069329],
                ],
                [
                    [0.546742119, 0.000000000, 0.711446650],
                    [-0.273370931, 0.473492518, 0.711446685],
                    [-0.273370931, -0.473492518, 0.711446685],
                ],
                [
                    [0.545783176, 0.000000000, 0.718281641],
                    [-0.272891484, 0.472662048, 0.718281692],
                    [-0.272891484, -0.472662048, 0.718281692],
                ],
                [
                    [0.543772827, 0.000000000, 0.725245195],
                    [-0.271886312, 0.470921034, 0.725245246],
                    [-0.271886312, -0.470921034, 0.725245246],
                ],
                [
                    [0.540062685, 0.000000000, 0.735690525],
                    [-0.270031247, 0.467707957, 0.735690578],
                    [-0.270031247, -0.467707957, 0.735690578],
                ],
                [
                    [0.532958020, 0.000000000, 0.751186806],
                    [-0.266478925, 0.461555135, 0.751186864],
                    [-0.266478925, -0.461555135, 0.751186864],
                ],
                [
                    [0.534315508, 0.000000000, 0.747022627],
                    [-0.267157668, 0.462730755, 0.747022684],
                    [-0.267157668, -0.462730755, 0.747022684],
                ],
                [
                    [0.534114296, 0.000000000, 0.747253747],
                    [-0.267057062, 0.462556500, 0.747253804],
                    [-0.267057062, -0.462556500, 0.747253804],
                ],
                [
                    [0.534114296, 0.000000000, 0.747253747],
                    [-0.267057062, 0.462556500, 0.747253804],
                    [-0.267057062, -0.462556500, 0.747253804],
                ],
            ]
        )
        * 7.01033623
        * units['Bohr']
    )

    reader_cells = np.array([atoms.cell.array for atoms in all_atoms])
    assert np.allclose(reader_cells, cells)


def test_licoo2_out():
    output_path = OUTPUT_DIR / 'LiCoO2.scf.out.gz'

    atoms = read(output_path, format='espresso-out')

    assert len(atoms) == 4
    assert atoms.get_potential_energy() == pytest.approx(
        -373.17520246 * units['Ry']
    )
    assert atoms.calc.get_fermi_level() == pytest.approx(9.1031)


def test_si_md_out():
    output_path = OUTPUT_DIR / 'si.md2.out.gz'

    atoms_list = read(output_path, format='espresso-out', index=':-1')

    assert len(atoms_list) == 100

    with pytest.raises(ValueError):
        atoms = read(output_path, format='espresso-out', index=-1)

    atoms = read(
        output_path, format='espresso-out', index=-1, results_required=False
    )

    assert len(atoms) == 2

    positions = (
        np.array(
            [
                [-0.123392744, -0.123392705, -0.123392719],
                [0.123392744, 0.123392705, 0.123392719],
            ]
        )
        * 10.1800
        * units['Bohr']
    )

    assert np.allclose(atoms.get_positions(), positions)


def test_all():
    all_outputs = list(OUTPUT_DIR.glob('*.out'))

    for output in all_outputs:
        try:
            read(output, format='espresso-out')
        except ValueError as e:
            if 'Required properties' in str(e):
                read(
                    output, format='espresso-out', results_required=False
                )
            else:
                raise e
