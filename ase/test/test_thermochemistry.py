# fmt: off
import numpy as np
import pytest

from ase import Atoms
from ase.build import bulk, molecule
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton
from ase.phonons import Phonons
from ase.thermochemistry import (
    CrystalThermo,
    HarmonicThermo,
    HinderedThermo,
    IdealGasThermo,
    get_molecular_symmetry,
)
from ase.vibrations import Vibrations


@pytest.mark.parametrize(('mol', 'ref_geom', 'ref_sym'),
    [('H2', 'linear', 2),
     ('CO', 'linear', 1),
     ('CH3OH', 'nonlinear', 1),
     ('H2O', 'nonlinear', 2),
     ('CH4', 'nonlinear', 12),
     ('C2H6', 'nonlinear', 6),
     ('Li', 'monatomic', 1)
    ])
def test_ideal_gas_thermo_pointgroup(mol, ref_geom, ref_sym):
    atoms = molecule(mol)
    geometry, symmetrynumber = get_molecular_symmetry(atoms)
    assert geometry == ref_geom
    assert symmetrynumber == ref_sym


def test_pointgroup_imperfect_C60(angle=4):
    """Test of angle tolerance.
    Here are coordinates for C60 fullerence, (from Michigan State University)
    that are not perfect; e.g. the distance to the center varies in the order
    of 0.01 - 0.1 Å.

    4 degrees should still be enough tolerance.
    """

    positions = np.array([
        [2.1665, 0.5906, 2.5874],
        [3.0378, 0.1766, 1.5918],
        [1.2786, -0.3098, 3.1679],
        [3.0118, -1.1434, 1.1654],
        [3.1034, -1.4335, -0.1930],
        [3.1503, 1.2106, 0.6682],
        [3.2428, 0.9149, -0.6859],
        [3.2192, -0.4023, -1.1207],
        [-0.4393, 1.3527, 3.1271],
        [0.4363, 2.2618, 2.5542],
        [-0.0296, 0.0633, 3.4379],
        [1.7442, 1.8790, 2.2830],
        [2.3519, 2.2676, 1.0990],
        [-0.2633, 3.0268, 1.6326],
        [0.3374, 3.4054, 0.4373],
        [1.6516, 3.0278, 0.1707],
        [-2.0903, -0.8225, 2.5955],
        [-2.5111, 0.4664, 2.2854],
        [-0.8449, -1.0252, 3.1738],
        [-1.6874, 1.5533, 2.5512],
        [-1.5843, 2.5858, 1.6319],
        [-3.2314, 0.4061, 1.1007],
        [-3.1227, 1.4410, 0.1746],
        [-2.2947, 2.5291, 0.4399],
        [-0.4908, -2.9133, 1.7365],
        [-1.7430, -2.7124, 1.1637],
        [-0.0393, -2.0684, 2.7453],
        [-2.5486, -1.6650, 1.5942],
        [-3.2602, -0.9141, 0.6701],
        [-1.6543, -3.0061, -0.1897],
        [-2.3542, -2.2439, -1.1170],
        [-3.1643, -1.1949, -0.6878],
        [2.1364, -2.0553, 1.7358],
        [1.6895, -2.9009, 0.7293],
        [1.2785, -1.6366, 2.7435],
        [0.3678, -3.3327, 0.7302],
        [-0.3440, -3.3904, -0.4594],
        [2.2889, -2.5250, -0.4640],
        [1.5790, -2.5718, -1.6580],
        [0.2560, -3.0054, -1.6531],
        [-2.1828, -0.5783, -2.5979],
        [-1.7480, -1.8694, -2.3083],
        [-0.4385, -2.2469, -2.5845],
        [-1.2815, 0.3189, -3.1671],
        [-2.1526, 2.0545, -1.7378],
        [-3.0485, 1.1535, -1.1811],
        [-3.0656, -0.1629, -1.6107],
        [-1.2661, 1.6407, -2.7271],
        [0.5039, 2.9361, -1.7418],
        [-0.3788, 3.3561, -0.7513],
        [-1.6943, 2.9186, -0.7491],
        [0.0521, 2.0730, -2.7355],
        [2.0976, 0.8340, -2.6051],
        [2.5517, 1.6923, -1.6107],
        [1.7589, 2.7452, -1.1824],
        [0.8420, 1.0206, -3.1786],
        [0.4461, -1.3495, -3.1661],
        [1.6983, -1.5485, -2.5908],
        [2.5184, -0.4623, -2.3171],
        [0.0218, -0.0645, -3.4585]
    ])

    atoms = Atoms("C60", positions=positions)
    try:
        geometry, sigma = get_molecular_symmetry(atoms, angle_tol=angle)
    except Exception as e:
        print(f'Failed at angle {angle} degrees')
        raise e
    assert geometry == 'nonlinear'
    assert sigma == 60


def test_pointgroup_eigtol(eigtol=0.015):
    """Test of eigenvalue tolerance. A hard case is a triatomic molecule
    with two heavy atoms and one light atom, e.g. HOBr. If the tolerance
    is too high, the moments of inertia will indicate, incorrectly, that
    the molecule is linear
    """

    positions = np.array([
        [0.3181, 0.5001, 0.0000],
        [-0.6262, 0.7055, 0.0000],
        [0.3181, -1.3356, 0.0000],
    ])

    atoms = Atoms('OHBr', positions=positions)
    try:
        geometry, sigma = get_molecular_symmetry(atoms, eig_tol=eigtol)
    except Exception as e:
        print(f'Failed at eigtol {eigtol}')
        raise e
    assert geometry == 'nonlinear'
    assert sigma == 1


def test_ideal_gas_thermo_n2(testdir):
    "We do a basic test on N2"
    atoms = Atoms("N2", positions=[(0, 0, 0), (0, 0, 1.1)])
    atoms.calc = EMT()
    QuasiNewton(atoms).run(fmax=0.01)
    energy = atoms.get_potential_energy()
    vib = Vibrations(atoms, name="igt-vib1")
    vib.run()
    vib_energies = vib.get_energies()
    assert len(vib_energies) == 6
    assert vib_energies[0] == pytest.approx(0.0, abs=1e-8)
    assert vib_energies[-1] == pytest.approx(1.52647479e-01)

    # ---------------------
    #   #    meV     cm^-1
    # ---------------------
    #   0    0.0       0.0 <--- remove!
    #   1    0.0       0.0 <--- remove!
    #   2    0.0       0.0 <--- remove!
    #   3    1.7      13.5 <--- remove!
    #   4    1.7      13.5 <--- remove!
    #   5  152.6    1231.2
    # ---------------------
    thermo = IdealGasThermo(
        vib_energies=vib_energies,
        geometry="linear",
        atoms=atoms,
        symmetrynumber=2,
        spin=0,
        potentialenergy=energy,
    )
    assert len(thermo.vib_energies) == 1
    assert thermo.vib_energies[0] == vib_energies[-1]
    assert thermo.geometry == "linear"
    assert thermo.get_ZPE_correction() == pytest.approx(0.07632373926263808)
    assert thermo.get_enthalpy(1000) == pytest.approx(0.6719935644272014)
    assert thermo.get_entropy(1000, 1e8) == pytest.approx(0.0017861226676818658)
    assert thermo.get_gibbs_energy(1000, 1e8) == pytest.approx(
        thermo.get_enthalpy(1000) - 1000 * thermo.get_entropy(1000, 1e8)
    )


def ideal_gas_thermo_ch3(
    vib_energies,
    geometry="nonlinear",
    atoms=None,
    symmetrynumber=6,
    potentialenergy=0.0,
    spin=0.5,
    ignore_imag_modes=False,
):
    if atoms is None:
        atoms = molecule("CH3")
    return IdealGasThermo(
        vib_energies=vib_energies,
        geometry=geometry,
        atoms=atoms,
        symmetrynumber=symmetrynumber,
        potentialenergy=potentialenergy,
        spin=spin,
        ignore_imag_modes=ignore_imag_modes,
    )


CH3_THERMO = {
    "ZPE": 1.185,
    "enthalpy": 10.610695269124156,
    "entropy": 0.0019310086280219891,
    "gibbs": 8.678687641495167,
}


def test_ideal_gas_thermo_ch3(testdir):
    """
    Now we try something a bit harder. Let's consider a
    CH3 molecule, such that there should be 3*4-6 = 6 modes
    for calculating the thermochemistry. We will also provide
    the modes in an unsorted list to make sure the correct
    values are cut. Note that these vibrational energies
    are simply toy values.

    Input: [1.0, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.35, 0.12]
    Expected: [0.12, 0.2, 0.3, 0.35, 0.4, 1.0]
    """
    thermo = ideal_gas_thermo_ch3(
        vib_energies=[1.0, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.35, 0.12],
        potentialenergy=9,
    )
    assert len(thermo.vib_energies) == 6
    assert list(thermo.vib_energies) == [0.12, 0.2, 0.3, 0.35, 0.4, 1.0]
    assert thermo.geometry == "nonlinear"
    assert thermo.get_ZPE_correction() == pytest.approx(CH3_THERMO["ZPE"])
    assert thermo.get_enthalpy(1000) == pytest.approx(CH3_THERMO["enthalpy"])
    assert thermo.get_entropy(1000, 1e8) == pytest.approx(CH3_THERMO["entropy"])
    assert thermo.get_gibbs_energy(1000, 1e8) == pytest.approx(
        thermo.get_enthalpy(1000) - 1000 * thermo.get_entropy(1000, 1e8)
    )


def test_ideal_gas_thermo_ch3_v2(testdir):
    """
    Let's do the same as above but provide only
    the 6 modes to use.

    Input: [0.12, 0.2, 0.3, 0.35, 0.4, 1.0]
    Expected: [0.12, 0.2, 0.3, 0.35, 0.4, 1.0]
    """
    thermo = ideal_gas_thermo_ch3(
        vib_energies=[0.12, 0.2, 0.3, 0.35, 0.4, 1.0], potentialenergy=9
    )
    assert len(thermo.vib_energies) == 6
    assert list(thermo.vib_energies) == [0.12, 0.2, 0.3, 0.35, 0.4, 1.0]
    assert thermo.geometry == "nonlinear"
    assert thermo.get_ZPE_correction() == pytest.approx(CH3_THERMO["ZPE"])
    assert thermo.get_enthalpy(1000) == pytest.approx(CH3_THERMO["enthalpy"])
    assert thermo.get_entropy(1000, 1e8) == pytest.approx(CH3_THERMO["entropy"])
    assert thermo.get_gibbs_energy(1000, 1e8) == pytest.approx(
        thermo.get_enthalpy(1000) - 1000 * thermo.get_entropy(1000, 1e8)
    )
    assert thermo.n_imag == 0


def test_ideal_gas_thermo_ch3_v3(testdir):
    """
    Now we give the module a more complicated set of
    vibrational frequencies to deal with to make sure
    the correct values are cut. This structure is not a
    minimum or TS and has several imaginary modes. However
    if we cut the first 6 modes, it'd look like all are
    real when they are not. We need to cut based on
    np.abs() of the vibrational energies.
    """

    # ---------------------
    #   #    meV     cm^-1
    # ---------------------
    #   0   63.8i    514.8i
    #   1   63.3i    510.7i
    #   2   42.4i    342.3i
    #   3    5.3i     43.1i <--- remove!
    #   4    0.0       0.0  <--- remove!
    #   5    0.0       0.0  <--- remove!
    #   6    0.0       0.0  <--- remove!
    #   7    5.6      45.5  <--- remove!
    #   8    6.0      48.1  <--- remove!
    #   9  507.9    4096.1
    #  10  547.2    4413.8
    #  11  547.7    4417.3
    # ---------------------
    vib_energies = [
        63.8j,
        63.3j,
        42.4j,
        5.3j,
        0.0,
        0.0,
        0.0,
        5.6,
        6.0,
        507.9,
        547.2,
        547.7,
    ]
    with pytest.raises(ValueError):
        # Imaginary frequencies present!!!
        thermo = ideal_gas_thermo_ch3(vib_energies=vib_energies)

    # Same as above, but let's try ignoring the
    # imag modes. This should just use: 507.9, 547.2, 547.7
    with pytest.warns(UserWarning):
        thermo = ideal_gas_thermo_ch3(vib_energies=vib_energies,
                                      ignore_imag_modes=True)
    assert list(thermo.vib_energies) == [507.9, 547.2, 547.7]
    assert thermo.n_imag == 3


def test_ideal_gas_thermo_ch3_v4(testdir):
    """
    Let's do another test like above, just for thoroughness.
    Again, this is not a minimum or TS and has several
    imaginary modes.
    """
    atoms = molecule("CH3")
    atoms.calc = EMT()
    vib = Vibrations(atoms, name="igt-vib2")
    vib.run()
    vib_energies = vib.get_energies()
    assert len(vib_energies) == 12
    assert vib_energies[0] == pytest.approx(0.09599611291404943j)
    assert vib_energies[-1] == pytest.approx(0.39035516516367375)

    # ---------------------
    #   #    meV     cm^-1
    # ---------------------
    #   0   96.0i    774.3i
    #   1   89.4i    721.0i
    #   2   89.3i    720.4i
    #   3   85.5i    689.7i <-- remove!
    #   4   85.4i    689.1i <-- remove!
    #   5   85.4i    689.1i <-- remove!
    #   6    0.0       0.0 <-- remove!
    #   7    0.0       0.0 <-- remove!
    #   8    0.0       0.0 <-- remove!
    #   9  369.4    2979.1
    #  10  369.4    2979.3
    #  11  390.4    3148.4
    # ---------------------
    with pytest.raises(ValueError):
        ideal_gas_thermo_ch3(vib_energies=vib_energies)

    with pytest.raises(ValueError):
        ideal_gas_thermo_ch3(vib_energies=[100 + 0.1j] * len(vib_energies))


VIB_ENERGIES_HARMONIC = np.array(
    [0.00959394 + 0.0j, 0.00959394 + 0.0j, 0.01741657 + 0.0j]
)


def harmonic_thermo(
    vib_energies=None,
    potentialenergy=4.120517148154894,
    ignore_imag_modes=False,
):
    return HarmonicThermo(
        vib_energies=vib_energies if vib_energies else VIB_ENERGIES_HARMONIC,
        potentialenergy=potentialenergy,
        ignore_imag_modes=ignore_imag_modes,
    )


HELMHOLTZ_HARMONIC = 4.060698673180732


def test_harmonic_thermo(testdir):
    "Basic test of harmonic thermochemistry"
    thermo = harmonic_thermo()
    helmholtz = thermo.get_helmholtz_energy(temperature=298.15)
    assert helmholtz == pytest.approx(HELMHOLTZ_HARMONIC)


def test_harmonic_thermo_v2(testdir):
    "Test with a reversed list giving the same results"
    vib_energies = list(VIB_ENERGIES_HARMONIC)
    vib_energies.sort(reverse=True)
    thermo = harmonic_thermo(vib_energies=vib_energies)
    helmholtz = thermo.get_helmholtz_energy(temperature=298.15)
    assert helmholtz == pytest.approx(HELMHOLTZ_HARMONIC)
    assert thermo.n_imag == 0


def test_harmonic_thermo_v3(testdir):
    "Test that a proper error is raised with imag modes"
    with pytest.raises(ValueError):
        harmonic_thermo(vib_energies=[10j])


def test_harmonic_thermo_v4(testdir):
    "Test that a proper warning is raised with non-crucial imag modes"
    with pytest.warns(UserWarning):
        thermo = harmonic_thermo(
            vib_energies=list(VIB_ENERGIES_HARMONIC) + [10j],
            ignore_imag_modes=True
        )
    helmholtz = thermo.get_helmholtz_energy(temperature=298.15)
    assert helmholtz == pytest.approx(HELMHOLTZ_HARMONIC)
    assert thermo.n_imag == 1


def test_crystal_thermo(asap3, testdir):
    atoms = bulk("Al", "fcc", a=4.05)
    calc = asap3.EMT()
    atoms.calc = calc
    energy = atoms.get_potential_energy()

    # Phonon calculator
    N = 7
    ph = Phonons(atoms, calc, supercell=(N, N, N), delta=0.05)
    ph.run()

    ph.read(acoustic=True)
    dos = ph.get_dos(kpts=(4, 4, 4)).sample_grid(npts=30, width=5e-4)
    phonon_energies = dos.get_energies()
    phonon_DOS = dos.get_weights()

    thermo = CrystalThermo(
        phonon_energies=phonon_energies,
        phonon_DOS=phonon_DOS,
        potentialenergy=energy,
        formula_units=4,
    )
    thermo.get_helmholtz_energy(temperature=298.15)


VIB_ENERGIES_HINDERED = (
    np.array(
        [
            3049.060670,
            3040.796863,
            3001.661338,
            2997.961647,
            2866.153162,
            2750.855460,
            1436.792655,
            1431.413595,
            1415.952186,
            1395.726300,
            1358.412432,
            1335.922737,
            1167.009954,
            1142.126116,
            1013.918680,
            803.400098,
            783.026031,
            310.448278,
            136.112935,
            112.939853,
            103.926392,
            77.262869,
            60.278004,
            25.825447,
        ]
    )
    / 8065.54429
)


def hindered_thermo(
    atoms=None,
    vib_energies=None,
    trans_barrier_energy=0.049313,
    rot_barrier_energy=0.017675,
    sitedensity=1.5e15,
    rotationalminima=6,
    symmetrynumber=1,
    mass=30.07,
    inertia=73.149,
    ignore_imag_modes=False,
):
    return HinderedThermo(
        atoms=atoms,
        vib_energies=vib_energies if vib_energies else VIB_ENERGIES_HINDERED,
        trans_barrier_energy=trans_barrier_energy,
        rot_barrier_energy=rot_barrier_energy,
        sitedensity=sitedensity,
        rotationalminima=rotationalminima,
        symmetrynumber=symmetrynumber,
        mass=mass,
        inertia=inertia,
        ignore_imag_modes=ignore_imag_modes,
    )


HELMHOLTZ_HINDERED = 1.5932242071261076


def test_hindered_thermo1():
    """
    Hindered translator / rotor.
    (Taken directly from the example given in the documentation.)
    """
    thermo = hindered_thermo()
    assert len(thermo.vib_energies) == 21
    helmholtz = thermo.get_helmholtz_energy(temperature=298.15)
    assert helmholtz == pytest.approx(HELMHOLTZ_HINDERED)


def test_hindered_thermo2():
    """
    Now reverse the vib energies and make sure results are the same
    """
    vib_energies = list(VIB_ENERGIES_HINDERED)
    vib_energies.sort(reverse=True)
    thermo = hindered_thermo(vib_energies=vib_energies)

    helmholtz = thermo.get_helmholtz_energy(temperature=298.15)
    assert len(thermo.vib_energies) == 21
    assert helmholtz == pytest.approx(HELMHOLTZ_HINDERED)
    assert thermo.n_imag == 0


def test_hindered_thermo3():
    "Now add an imaginary mode and make sure it is removed"
    with pytest.warns(UserWarning):
        thermo = hindered_thermo(
            vib_energies=list(VIB_ENERGIES_HINDERED) + [10j],
            ignore_imag_modes=True
        )
    assert thermo.get_helmholtz_energy(temperature=298.15) == pytest.approx(
        HELMHOLTZ_HINDERED
    )
    assert thermo.n_imag == 1


def test_hindered_thermo4():
    "Make sure a ValueError is raised if imag modes are present"
    with pytest.raises(ValueError):
        hindered_thermo(vib_energies=[100 + 0.1j] * 24)


def test_hindered_thermo5():
    "Make sure appropriate amount are cut"
    atoms = bulk("Cu") * (2, 2, 2)
    thermo = hindered_thermo(atoms=atoms)
    assert len(thermo.vib_energies) == 3 * len(atoms) - 3
