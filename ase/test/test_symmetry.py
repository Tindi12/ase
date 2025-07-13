import numpy as np
import pytest

from ase import Atoms
from ase.build import molecule
from ase.collections import g2
from ase.symmetry import PointGroupAnalyzer

@pytest.mark.parametrize('label, pg, sym', [
    ('Al', 'M', 1),
    ('CH3CONH2', 'C1', 1),
    ('HCOOH', 'Cs', 1),
    ('HOCl', 'Cs', 1),
    ('N2H4', 'C2', 2),
    ('H2O', 'C2v', 2),
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
    assert pga.pointgroup == pg
    assert pga.symmetry_number == sym
    
