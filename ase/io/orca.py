import re
from io import StringIO
from pathlib import Path
from typing import List, Optional

import numpy as np

from ase.io import read
from ase.units import Bohr, Hartree
from ase.utils import reader, writer
from ase import Atom, Atoms
from ase.calculators.singlepoint import SinglePointDFTCalculator

# Made from NWChem and FHI-aims interface

class ORCAParseError(Exception):
    """Exception raised if an error occurs when parsing an ORCA output file"""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

@reader
def read_geom_orcainp(fd):
    """Method to read geometry from an ORCA input file."""
    lines = fd.readlines()

    # Find geometry region of input file.
    stopline = 0
    for index, line in enumerate(lines):
        if line[1:].startswith('xyz '):
            startline = index + 1
            stopline = -1
        elif (line.startswith('end') and stopline == -1):
            stopline = index
        elif (line.startswith('*') and stopline == -1):
            stopline = index
    # Format and send to read_xyz.
    xyz_text = '%i\n' % (stopline - startline)
    xyz_text += ' geometry\n'
    for line in lines[startline:stopline]:
        xyz_text += line
    atoms = read(StringIO(xyz_text), format='xyz')
    atoms.set_cell((0., 0., 0.))  # no unit cell defined

    return atoms


@writer
def write_orca(fd, atoms, params):
    # conventional filename: '<name>.inp'
    fd.write(f"! {params['orcasimpleinput']} \n")
    fd.write(f"{params['orcablocks']} \n")

    if 'coords' not in params['orcablocks']:
        fd.write('*xyz')
        fd.write(" %d" % params['charge'])
        fd.write(" %d \n" % params['mult'])
        for atom in atoms:
            if atom.tag == 71:  # 71 is ascii G (Ghost)
                symbol = atom.symbol + ' : '
            else:
                symbol = atom.symbol + '   '
            fd.write(
                symbol
                + str(atom.position[0])
                + " "
                + str(atom.position[1])
                + " "
                + str(atom.position[2])
                + "\n"
            )
        fd.write('*\n')


def read_charge(lines: List[str]) -> Optional[float]:
    """Read sum of atomic charges."""
    charge = None
    for line in lines:
        if 'Sum of atomic charges' in line:
            charge = float(line.split()[-1])
    return charge


def read_energy(lines: List[str]) -> Optional[float]:
    """Read energy."""
    energy = None
    for line in lines:
        if 'FINAL SINGLE POINT ENERGY' in line:
            if "Wavefunction not fully converged" in line:
                energy = float('nan')
            else:
                energy = float(line.split()[-1])
    if energy is not None:
        return energy * Hartree
    return energy


def read_center_of_mass(lines: List[str]) -> Optional[np.ndarray]:
    """ Scan through text for the center of mass """
    # Example:
    # 'The origin for moment calculation is the CENTER OF MASS  =
    # ( 0.002150, -0.296255  0.086315)'
    # Note the missing comma in the output
    com = None
    for line in lines:
        if 'The origin for moment calculation is the CENTER OF MASS' in line:
            line = re.sub(r'[(),]', '', line)
            com = np.array([float(_) for _ in line.split()[-3:]])
    if com is not None:
        return com * Bohr  # return the last match
    return com


def read_dipole(lines: List[str]) -> Optional[np.ndarray]:
    """Read dipole moment.

    Note that the read dipole moment is for the COM frame of reference.
    """
    dipole = np.zeros(3)
    for line in lines:
        if 'Total Dipole Moment' in line:
            dipole = np.array([float(_) for _ in line.split()[-3:]])
    if dipole is not None:
        return dipole * Bohr  # Return the last match
    return dipole

def read_atoms(lines: List[str]) -> Optional[np.ndarray]:
    """Read atomic positions and symbols. Create Atoms object."""
    line_start = -1
    natoms = 0

    for ll, line in enumerate(lines):
        if ('Number of atoms' in line):
            natoms = int(line.split()[4])
        elif ('CARTESIAN COORDINATES (ANGSTROEM)' in line):
            line_start = ll + 2

    # Check if atoms present and if their number is given.
    if (line_start == -1):
        raise ORCAParseError(
            'No information about the atomic structure in the ORCA output file.')
    elif (natoms == 0):
        raise ORCAParseError(
            'No information about number of atoms in the ORCA output file.')

    positions = np.zeros((natoms,3))
    symbols = [""] * natoms

    for ll, line in enumerate(lines[line_start:line_start + natoms]):
        inp = line.split()
        positions[ll, :] = [float(pos) for pos in inp[1:4]]
        symbols[ll] = inp[0]

    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.set_pbc([False, False, False])

    return atoms

def read_forces(lines: List[str]) -> Optional[np.ndarray]:
    """Read forces from output file if available. Else return None.

    Taking the forces from the output files (instead of the engrad-file) to
    be more general. The forces can be present in general output even if
    the engrad file is not there.

    Note: If more than one geometry relaxation step is available, 
          forces do not always exist for the first step. In this case, for
          the first step an array of None will be returned. The following
          relaxation steps will then have forces available.
    """
    line_start = -1
    natoms = 0

    for ll, line in enumerate(lines):
        if ('Number of atoms' in line):
            natoms = int(line.split()[4])
        elif ('CARTESIAN GRADIENT' in line):
            line_start = ll + 3

    # Check if number of atoms is available.
    if (natoms == 0):
        raise ORCAParseError(
            'No information about number of atoms in the ORCA output file.')

    #Forces are not always available. If not available, return None.
    if (line_start == -1):
        forces = np.full((natoms,3), None)
    else:
        forces = np.zeros((natoms,3))

        for ll, line in enumerate(lines[line_start:line_start + natoms]):
            inp = line.split()
            forces[ll, :] = [float(pos) for pos in inp[3:6]]
        forces = -forces * Hartree / Bohr

    return forces

def get_chunks(lines):
    """Separate out the chunks for each geometry relaxation step."""
    finished = False
    relaxation = False

    chunk_lines = []
    for line in lines:
        if ('FINAL SINGLE POINT ENERGY' in line):
            chunk_lines.append(line)
            yield chunk_lines
            chunk_lines = []
        elif ('ORCA TERMINATED NORMALLY' in line):
            finished = True
            # Return the last part of the calculation
            yield chunk_lines
        elif ('ORCA SCF GRADIENT CALCULATION' in line):
            relaxation = True
        elif ('FINAL SINGLE POINT ENERGY' not in line):
            chunk_lines.append(line)
        else:
            raise ORCAParseError(
            'No information about chunk in output file.')

    # Give error if calculation not finished for single-point calculations.
    if (not finished and not relaxation):
        raise ORCAParseError(
            'Error: Calculation did not finish!')
    # Give warning if calculation not finished for geometry optimizations.
    elif (not finished and relaxation):
        print('WARNING: Calculation did not finish!')



@reader
def read_orca_output(fd, index):
    """From the ORCA output file: Read Energy, positions and forces,
       and dipole moment in the frame of reference of the center of mass.

    Create separated atoms object for each geometry frame through
    parsing the output file in chunks.
    """
    images = []
    lines = fd.readlines()

    #Get the chunks of the output file
    chunks = list(get_chunks(lines))

    # Iterate over chunks and create a separate atoms object for each
    for i, chunk in enumerate(chunks[:-1]):
        energy = read_energy(chunk)
        charge = read_charge(chunk)
        com = read_center_of_mass(chunk)
        atoms = read_atoms(chunk)
        forces = read_forces(chunk)

        # Dipole moment only printed at the end of calculation.
        if (i == len(chunks)-2):
            dipole = read_dipole(chunks[-1])
        else:
            dipole = np.zeros(3)

        atoms.calc = SinglePointDFTCalculator(
                atoms,
                energy=energy,
                free_energy=energy,
                forces=forces,
                #stress=self.stress,
                #stresses=self.stresses,
                #magmom=self.magmom,
                dipole=dipole,
                #dielectric_tensor=self.dielectric_tensor,
                #polarization=self.polarization,
            )
        #collect images
        images.append(atoms)

    return images[index]


@reader
def read_orca_engrad(fd):
    """Read Forces from ORCA .engrad file."""
    getgrad = False
    gradients = []
    tempgrad = []
    for _, line in enumerate(fd):
        if line.find('# The current gradient') >= 0:
            getgrad = True
            gradients = []
            tempgrad = []
            continue
        if getgrad and "#" not in line:
            grad = line.split()[-1]
            tempgrad.append(float(grad))
            if len(tempgrad) == 3:
                gradients.append(tempgrad)
                tempgrad = []
        if '# The at' in line:
            getgrad = False

    forces = -np.array(gradients) * Hartree / Bohr
    return forces


def read_orca_outputs(directory, stdout_path):
    # Reproduce old functionality to keep backwards compatability.
    stdout_path = Path(stdout_path)
    results = {}
    atoms = read_orca_output(stdout_path,index=-1)

    results['energy'] = atoms.get_total_energy()
    results['free_energy'] = atoms.get_total_energy()

    if (abs(atoms.get_dipole_moment()[0]) > 0
        and abs(atoms.get_dipole_moment()[1]) > 0
        and abs(atoms.get_dipole_moment()[2]) > 0):
        results['dipole'] = atoms.get_dipole_moment()

    # Does engrad always exist? - No!
    # Will there be other files -No -> We should just take engrad
    # as a direct argument.  Or maybe this function does not even need to
    # exist.
    engrad_path = stdout_path.with_suffix('.engrad')
    if engrad_path.is_file():
        results['forces'] = read_orca_engrad(engrad_path)
    return results
