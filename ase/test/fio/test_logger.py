from io import StringIO

import numpy as np

from ase.build import bulk, molecule
from ase.calculators.emt import EMT
from ase.io.logger import Logger
from ase.md.bussi import Bussi
from ase.md.logger import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import BFGS
from ase.optimize.logger import OptLogger
from ase.units import fs


def test_logger():
    string_io = StringIO()

    logger = Logger(string_io)
    logger.add_field('Class', lambda: 'MyRandomMDClass', fmt='{:<24s}')
    logger.add_field('Epot[eV]', lambda: 123141.0)
    logger.add_field('Step', lambda: 1, fmt='{:>12d}')
    logger.write_header()

    logger()

    header = string_io.getvalue().split('\n')[0]
    values = string_io.getvalue().split('\n')[1]

    assert header == 'Class' + ' ' * 22 + 'Epot[eV]' + ' ' * 9 + 'Step'
    assert (
        values == 'MyRandomMDClass' + ' ' * 10 + '123141.000' + ' ' * 12 + '1'
    )

    string_io.close()


def test_md_logger():
    atoms = bulk('Cu')
    atoms.calc = EMT()

    string_io = StringIO()

    MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    dyn = Bussi(
        atoms, 1.0 * fs, temperature_K=300, taut=50 * fs, logfile=string_io
    )
    dyn.run(10)

    header = string_io.getvalue().split('\n')[0]

    assert (
        header
        == 'Time[ps]'
        + ' ' * 9
        + 'Etot[eV]'
        + ' ' * 5
        + 'Epot[eV]'
        + ' ' * 5
        + 'Ekin[eV]'
        + ' ' * 7
        + 'T[K]'
        + ' ' * 4
        + 'Econs[eV]'
    )

    string_io.close()


def test_opt_logger():
    atoms = molecule('H2O')
    atoms.rattle(0.1)

    atoms.calc = EMT()

    string_io = StringIO()

    opt = BFGS(atoms, logfile=string_io)

    opt.run(fmax=0.01)

    header = string_io.getvalue().split('\n')[0]

    assert (
        header
        == 'Optimizer'
        + ' ' * 14
        + 'Step'
        + ' ' * 9
        + 'Time'
        + ' ' * 5
        + 'Epot[eV]'
        + ' ' * 3
        + 'Fmax[eV/A]'
    )

    string_io.close()


def test_opt_custom_logger():
    atoms = molecule('H2O')
    atoms.rattle(0.1)
    atoms.calc = EMT()

    string_io = StringIO()

    opt = BFGS(atoms, logfile=None)

    logger = OptLogger(opt, string_io)

    def negative_omega():
        if opt.nsteps > 0:
            return str(np.any(np.linalg.eigh(opt.H)[0] < 0))
        else:
            return 'N/A'

    logger.add_field('NegativeEigenvalues', negative_omega, fmt='{:>22s}')
    opt.attach(logger)

    logger.write_header()

    opt.run(fmax=0.01)

    text = string_io.getvalue()

    assert 'NegativeEigenvalues' in text
    assert 'False' in text
    assert 'Optimizer' in text

    string_io.close()


def test_opt_stress_logger():
    atoms = bulk('Cu') * (3, 3, 3)
    atoms.rattle(0.1)
    atoms.calc = EMT()

    string_io = StringIO()

    MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    dyn = Bussi(atoms, 1.0 * fs, temperature_K=300, taut=50 * fs)

    logger = MDLogger(dyn, string_io)

    logger.add_stress_fields(
        atoms, mask=[False, True, False, True, True, False]
    )

    dyn.attach(logger)

    logger.write_header()

    dyn.run(10)

    string_io.seek(0)
    logger_lines = string_io.read()

    pos = string_io.tell()

    assert 'Stress[yy][GPa]' in logger_lines
    assert 'Stress[xx][GPa]' not in logger_lines
    assert 'Stress[zz][GPa]' not in logger_lines
    assert 'Stress[yz][GPa]' in logger_lines
    assert 'Stress[xz][GPa]' in logger_lines
    assert 'Stress[xy][GPa]' not in logger_lines

    logger.remove_fields('Stress[yz][GPa]')
    logger.add_stress_fields(
        atoms, mask=[False, False, False, False, False, True]
    )

    logger.write_header()

    dyn.run(10)

    string_io.seek(pos)
    new_logger_lines = string_io.read()

    assert 'Stress[yy][GPa]' not in new_logger_lines
    assert 'Stress[xz][GPa]' not in new_logger_lines
    assert 'Stress[xy][GPa]' in new_logger_lines

    string_io.close()
