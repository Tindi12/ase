from dataclasses import dataclass

import pytest

from ase.build import molecule
from ase.io import read
from ase.symmetry import PointGroupAnalyzer


@pytest.mark.parametrize('label, pg, sym', [
    ('Al', 'Kh', 1),
    ('CH3CONH2', 'C1', 1),
    ('HCOOH', 'Cs', 1),
    ('HOCl', 'Cs', 1),
    ('CH3CHO', 'Cs', 1),
    ('N2H4', 'C2', 2),
    ('H2O', 'C2v', 2),
    # C4H4NH and COF2 can be mistakenly identified as symmetric
    # due to mass distribution, which might lead to incorrect
    # identification as Cs
    ('C4H4NH', 'C2v', 2),
    ('COF2', 'C2v', 2),
    ('butadiene', 'C2h', 2),
    ('NH3', 'C3v', 3),
    ('C2H4', 'D2h', 4),
    ('C5H8', 'D2d', 4),
    ('C2H6', 'D3d', 6),
    ('C6H6', 'D6h', 12),
    ('CH4', 'Td', 12),
    ('C60', 'Ih', 60)
])
def test_pointgroup(label, pg, sym):
    mol = molecule(label)
    mol.euler_rotate(30., 30., 30.)
    pga = PointGroupAnalyzer(mol)
    pg_calc = pga.pointgroup
    sym_calc = pga.symmetry_number
    print(label)
    print([s.axis for s in pga._pointgroup_and_symmetries[1]])
    assert pg_calc == pg
    assert sym_calc == sym


@dataclass
class MoleculeData:
    label: str
    pointgroup: str
    symmetry_number: int

    @property
    def datafile(self):
        return f"pointgroup/{self.label}.xyz"


molecules_from_testdata = [
    MoleculeData(label='SF6', pointgroup='Oh', symmetry_number=24),
    MoleculeData(label='B12H12', pointgroup='Ih', symmetry_number=60),
    # HOBr can be incorrectly detected as linear because of its mass
    # distribution
    MoleculeData(label='HOBr', pointgroup='Cs', symmetry_number=1),
    MoleculeData(label='C4H2', pointgroup='D*h', symmetry_number=2),
    MoleculeData(label='C3O2', pointgroup='D*h', symmetry_number=2),
    MoleculeData(label='C10H16', pointgroup='D2', symmetry_number=4),
    MoleculeData(label='C60F36', pointgroup='T', symmetry_number=12),
    MoleculeData(label='C2H2Cl2F2', pointgroup='Ci', symmetry_number=1),
    MoleculeData(label='C8H8', pointgroup='Oh', symmetry_number=24),
    MoleculeData(label='C5H4F4', pointgroup='S4', symmetry_number=2),
    MoleculeData(label='uranocene', pointgroup='D8h', symmetry_number=16),
    MoleculeData(label='S8', pointgroup='D4d', symmetry_number=8),
    MoleculeData(label='corannulene', pointgroup='C5v', symmetry_number=5),
    MoleculeData(label='ferrocene', pointgroup='D5d', symmetry_number=10),
    MoleculeData(label='C70', pointgroup='D5h', symmetry_number=10),
]


@pytest.mark.parametrize('moldata', molecules_from_testdata)
def test_pointgroups_in_testdata(datadir, moldata):
    mol = read(datadir / moldata.datafile)
    mol.euler_rotate(30., 30., 30.)
    pga = PointGroupAnalyzer(mol)
    assert pga.pointgroup == moldata.pointgroup
    assert pga.symmetry_number == moldata.symmetry_number


# Some molecules with slight deviations from perfect symmetry
imperfect_molecules = [
    MoleculeData(label='C20', pointgroup='Ih', symmetry_number=60),
    MoleculeData(label='thorium_nitrate_ion', pointgroup='Th',
        symmetry_number=12),
]


@pytest.mark.parametrize('moldata', imperfect_molecules)
def test_pointgroups_imperfect(datadir, moldata):
    mol = read(datadir / moldata.datafile)
    mol.euler_rotate(30., 30., 30.)
    pga = PointGroupAnalyzer(mol, eigtol=0.015, angtol=6, disttol=0.3)
    assert pga.pointgroup == moldata.pointgroup
    assert pga.symmetry_number == moldata.symmetry_number
