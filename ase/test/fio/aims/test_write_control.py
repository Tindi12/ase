"""Test writing control.in files for Aims using ase.io.aims.

Control.in file contains calculation parameters such as the functional and
k grid as well as basis set size parameters. We write this file to a string
and assert we find expected values.
"""
# Standard imports.
import io
import re

# Third party imports.
import pytest
pyfhiaims = pytest.importorskip("pyfhiaims")

from pyfhiaims.control.cube import AimsCube

import ase.build
import ase.calculators.aims
import ase.io.aims


@pytest.fixture()
def parameters_dict():
    """Creates a parameters dictionary used to configure Aims simulation."""
    return {
        "xc": "LDA",
        "kpts": [2, 2, 2],
        "smearing": ("gaussian", 0.1),
        "output": ["dos 0.0 10.0 101 0.05", "hirshfeld"],
        "dos_kgrid_factors": [21, 21, 21],
        "vdw_correction_hirshfeld": True,
        "compute_forces": True,
        "output_level": "MD_light",
        "charge": 0.0,
        "plus_u": {"Au": [(1, 0, 0.25), (3, 1, 0.35)]},
    }


def contains(pattern, txt):
    """"Regex search for pattern in the text."""
    return re.search(pattern, txt, re.M | re.DOTALL)


def write_control_to_string(ase_atoms_obj, parameters):
    """Helper function to write control.in file to a stringIO object.

    Args:
        ase_atoms_obj: ASE Atoms object that contains the atoms in the unit
            cell that are to be simulated.
        parameters: Dictionary that contains simulation parameters to be
            written to the control.in FHI-aims file which dictates to the
            aims executable how the simulation should be run.
    """
    string_output = io.StringIO()
    ase.io.aims.write_control(string_output, ase_atoms_obj, parameters)
    return string_output.getvalue()


@pytest.fixture()
def bulk_au():
    """Create an ASE.Atoms bulk object of Gold."""
    return ase.build.bulk("Au")


@pytest.fixture()
def bulk_aucl():
    """Create an ASE AuCl Atoms object"""
    return ase.build.bulk("AuCl",
                          crystalstructure="rocksalt",
                          a=6.32,
                          cubic=True)


AIMS_AU_SPECIES_LIGHT = """\
################################################################################
  species        Au
#     global species definitions
    nucleus             79
    mass                196.966569
#
    l_hartree           4
#
    cut_pot             3.5  1.5  1.0
    basis_dep_cutoff    1e-4
#
    radial_base         73 5.0
    radial_multiplier   1
    angular_grids specified
      division   0.5066   50
      division   0.9861  110
      division   1.2821  194
      division   1.5344  302
#      division   2.0427  434
#      division   2.1690  590
#      division   2.2710  770
#      division   2.3066  974
#      division   2.7597 1202
#      outer_grid 974
      outer_grid 302
################################################################################
#
#  Definition of "minimal" basis
#
################################################################################
#     valence basis states
    valence      6  s   1.
    valence      5  p   6.
    valence      5  d  10.
    valence      4  f  14.
#     ion occupancy
    ion_occ     6  s   0.
    ion_occ     5  p   6.
    ion_occ     5  d   9.
    ion_occ     4  f   14.
################################################################################
#
#  Suggested additional basis functions. For production calculations,
#  uncomment them one after another (the most important basis functions are
#  listed first).
#
#  Constructed for dimers: 2.10, 2.45, 3.00, 4.00 AA
#
################################################################################
#  "First tier" - max. impr. -161.60  meV, min. impr. -4.53 meV
     ionic 6 p auto
     hydro 4 f 7.4
     ionic 6 s auto
#     hydro 5 g 10
#     hydro 6 h 12.8
     hydro 3 d 2.5
#  "Second tier" - max. impr. -2.46  meV, min. impr. -0.28 meV
#     hydro 5 f 14.8
#     hydro 4 d 3.9
#     hydro 3 p 3.3
#     hydro 1 s 0.45
#     hydro 5 g 16.4
#     hydro 6 h 13.6
#  "Third tier" - max. impr. -0.49  meV, min. impr. -0.09 meV
#     hydro 4 f 5.2
#     hydro 4 d 5
#     hydro 5 g 8
#     hydro 5 p 8.2
#     hydro 6 d 12.4
#     hydro 6 s 14.8
#  Further basis functions: -0.08 meV and below
#     hydro 5 f 18.8
#     hydro 5 g 20
#    hydro 5 g 15.2
"""

# removed part of text that is not relevant to basis functions.
AIMS_CL_SPECIES_LIGHT = """\
  species        Cl
#     global species definitions
    nucleus             17
    mass                35.453
#
    l_hartree           4
#
    cut_pot             3.5          1.5  1.0
    basis_dep_cutoff    1e-4
#
    radial_base         45 5.0
    radial_multiplier   1
    angular_grids       specified
      division   0.4412  110
      division   0.5489  194
      division   0.6734  302
#      division   0.7794  434
#      division   0.9402  590
#      division   1.0779  770
#      division   1.1792  974
#      outer_grid  974
      outer_grid  302
################################################################################
#
#  Definition of "minimal" basis
#
################################################################################
#     valence basis states
    valence      3  s   2.
    valence      3  p   5.
#     ion occupancy
    ion_occ      3  s   1.
    ion_occ      3  p   4.
################################################################################
#
#  Suggested additional basis functions. For production calculations,
#  uncomment them one after another (the most important basis functions are
#  listed first).
#
#  Constructed for dimers: 1.65 A, 2.0 A, 2.5 A, 3.25 A, 4.0 A
#
################################################################################
#  "First tier" - improvements: -429.57 meV to -15.03 meV
     ionic 3 d auto
     hydro 2 p 1.9
     hydro 4 f 7.4
     ionic 3 s auto
#     hydro 5 g 10.4
#  "Second tier" - improvements: -7.84 meV to -0.48 meV
#     hydro 3 d 3.3
#     hydro 5 f 9.8
#     hydro 1 s 0.75
#     hydro 5 g 11.2
#     hydro 4 p 10.4
#  "Third tier" - improvements: -1.00 meV to -0.12 meV
#     hydro 4 d 12.8
#     hydro 4 f 4.6
#     hydro 4 d 10.8
#     hydro 2 s 1.8
#     hydro 3 p 3
"""


@pytest.fixture()
def aims_species_dir_light(tmp_path):
    """Create temporary directory to store species files."""
    species_dir_light = tmp_path / "light"
    species_dir_light.mkdir()
    path_au = species_dir_light / "79_Au_default"
    path_au.write_text(AIMS_AU_SPECIES_LIGHT)
    path_cl = species_dir_light / "17_Cl_default"
    path_cl.write_text(AIMS_CL_SPECIES_LIGHT)
    return species_dir_light


@pytest.mark.parametrize(
    "tier,expected_basisset_expr",
    [
        (
            None,
            [
                "             ionic      6  p  auto",
                "#            hydro      6  h  12.8",
                "#            hydro      5  g  10.4",
                "             ionic      3  d  auto"
            ]
        ),
        (
            0,
            [
                "#            ionic      6  p  auto",
                "#            ionic      3  d  auto"
            ],
        ),
        (
            {"Au": 1, "Cl": 3},
            [
                "             hydro      6  h  12.8",
                "#            hydro      5  f  14.8",
                "             hydro      5  g  10.4",
                "             hydro      3  d  3.3",
                "             hydro      4  d  12.8",
            ]
        ),
    ]
)
def test_control(
    bulk_aucl,
    parameters_dict,
    aims_species_dir_light,
    tier,
    expected_basisset_expr
):
    """Tests that control.in for a Gold bulk system works.

    This test tests several things simulationeously, much of
    the aims IO functionality for writing the conrol.in file, such as adding an
    AimsCube to the system.
    """
    # Copy the global parameters dicitonary to avoid rewriting common
    # parameters.
    parameters = parameters_dict
    parameters['species_dir'] = aims_species_dir_light

    # Add AimsCube to the parameter dictionary.
    parameters["cubes"] = [AimsCube(type="delta_density", points=[50, 50, 50])]
    parameters["tier"] = tier
    # Write control.in file to a string which we can directly access for
    # testing.
    control_file_as_string = write_control_to_string(bulk_aucl, parameters)

    assert contains(r"k_grid\s+2 2 2", control_file_as_string)
    assert contains(
        r"k_offset\s+0.25 0.25 0.25", control_file_as_string)
    assert contains(r"occupation_type\s+gaussian 0.1", control_file_as_string)
    assert contains(r"output\s+dos 0.0 10.0 101 0.05", control_file_as_string)
    assert contains(r"output\s+hirshfeld", control_file_as_string)
    assert contains(r"dos_kgrid_factors\s+21 21 21", control_file_as_string)
    assert contains(r"vdw_correction_hirshfeld", control_file_as_string)
    assert contains(r"compute_forces\s+.true.", control_file_as_string)
    assert contains(r"output_level\s+MD_light", control_file_as_string)
    assert contains(r"charge\s+0.0", control_file_as_string)
    assert contains("output cube delta_density", control_file_as_string)

    assert contains(r"plus_u\s+1 0 0.25", control_file_as_string)
    assert contains(r"plus_u\s+3 1 0.35", control_file_as_string)

    assert contains(
        r"cube origin  0.0{12}e\+00  0.0{12}e\+00  0.0{12}e\+00",
        control_file_as_string,
    )
    assert contains(
        r"cube edge 50  1.0{12}e-01  0.0{12}e\+00  0.0{12}e\+00",
        control_file_as_string,
    )
    assert contains(
        r"cube edge 50  0.0{12}e\+00  1.0{12}e-01  0.0{12}e\+00",
        control_file_as_string,
    )
    assert contains(
        r"cube edge 50  0.0{12}e\+00  0.0{12}e\+00  1.0{12}e-01",
        control_file_as_string,
    )

    for expr in expected_basisset_expr:
        assert expr in control_file_as_string


@pytest.mark.parametrize(
    "functional,expected_functional_expression",
    [("PBE", r"xc\s+PBE"), ("LDA", r"xc\s+pw-lda"),
     pytest.param("PBE_06_Fake", None, marks=pytest.mark.xfail)])
def test_control_functional(
        aims_species_dir_light, bulk_au, parameters_dict, functional: str,
        expected_functional_expression: str):
    """Test that the functional written to the control.in file."""
    # Copy the global parameters dicitonary to avoid rewriting common
    # parameters. Then assign functional to parameter dictionary.
    parameters = parameters_dict
    parameters['species_dir'] = aims_species_dir_light
    parameters["xc"] = functional

    control_file_as_string = write_control_to_string(bulk_au, parameters)
    assert contains(expected_functional_expression, control_file_as_string)
