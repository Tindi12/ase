# fmt: off

"""Defines class/functions to write input and parse output for FHI-aims."""
import os
import time
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from ase import Atoms
from ase.calculators.calculator import kpts2mp
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.units import Ang, fs
from ase.utils import deprecated, reader, writer

from pyfhiaims.control.control import AimsControl
from pyfhiaims.geometry.atom import FHIAimsAtom
from pyfhiaims.geometry.geometry import AimsGeometry
from pyfhiaims.outputs.stdout import AimsParseError, AimsStdout

v_unit = Ang / (1000.0 * fs)

LINE_NOT_FOUND = object()


# Read aims geometry files
@reader
def read_aims(fd: TextIO | str | Path, apply_constraints=True) -> Atoms:
    """
    Import FHI-aims geometry type files.

    Reads unitcell, atom positions and constraints from
    a geometry.in file.
    """
    geometry = AimsGeometry.from_file(fd)
    atoms = geometry.ase_atoms

    if apply_constraints:
        atoms.set_positions(atoms.get_positions())
    return atoms


def get_aims_header() -> str:
    """Returns the header for aims input files"""
    lines = ["#" + "=" * 79]
    for line in [
        "Created using the Atomic Simulation Environment (ASE)",
        time.asctime(),
    ]:
        lines.append("# " + line)
    return "\n".join(lines)


def _write_velocities_alias(args: list, kwargs: dict[str, Any]) -> bool:
    arg_position = 5
    if len(args) > arg_position and args[arg_position]:
        args[arg_position - 1] = True
    elif kwargs.get("velocities", False):
        if len(args) < arg_position:
            kwargs["write_velocities"] = True
        else:
            args[arg_position - 1] = True
    else:
        return False
    return True


# Write aims geometry files
@deprecated(
    "Use of `velocities` is deprecated, please use `write_velocities`",
    category=FutureWarning,
    callback=_write_velocities_alias,
)
@writer
def write_aims(
    fd: TextIO | str | Path,
    atoms: Atoms,
    scaled: bool = False,
    geo_constrain: bool = False,
    write_velocities: bool = False,
    velocities: bool = False,
    ghosts: None | Iterable[int] = None,
    info_str: None | str = None,
    wrap: bool = False,
):
    """
    Method to write FHI-aims geometry files.

    Writes the atoms positions and constraints (only FixAtoms is
    supported at the moment).

    Args:
        fd: TextIO | str | Path
            File to output structure to
        atoms: Atoms
            structure to output to the file
        scaled: bool
            If True use fractional coordinates instead of Cartesian coordinates
        geo_constrain: bool
            If true add parameteric constraints
        write_velocities: bool
            If True add the atomic velocity vectors to the file
        velocities: bool
            NOT AN ARRAY OF VELOCITIES, but the legacy version of
            `write_velocities`
        ghosts: list[int]
            A list of indexes, 1 = ghost, regular atom otherwise
        info_str: str
            A string to be added to the header of the file
        wrap: bool
            Wrap atom positions to cell before writing

    .. deprecated:: 3.23.0
        Use of ``velocities`` is deprecated, please use ``write_velocities``.

    """
    make_empty = None
    if ghosts is not None:
        make_empty = [ii for ii, ind in enumerate(ghosts) if ind == 1]

    geometry = AimsGeometry.from_atoms(
        atoms,
        remove_velocities=not write_velocities,
        geo_constrain=geo_constrain,
        make_empty=make_empty,
        scaled=scaled,
        wrap=wrap,
    )

    header = get_aims_header()
    # If writing additional information is requested via info_str:
    if info_str is not None:
        header += "\n# Additional information:\n"
        if isinstance(info_str, list):
            header += "\n".join([f"#  {s}" for s in info_str])
        else:
            header += f"# {info_str}"
        header += "\n"

    header += "#=======================================================\n"
    geometry.write_file(fd, header)


def get_species_directory(species_dir: str | Path | None = None):
    """Get the directory where the basis set information is stored

    If the requested directory does not exist then raise an Error

    Parameters
    ----------
    species_dir: str
        Requested directory to find the basis set info from. E.g.
        `~/aims2022/FHIaims/species_defaults/defaults_2020/light`.

    Returns
    -------
    Path
        The Path to the requested or default species directory.

    Raises
    ------
    RuntimeError
        If both the requested directory and the default one is not defined
        or does not exit.
    """
    if species_dir is None:
        species_dir = os.environ.get("AIMS_SPECIES_DIR")

    if species_dir is None:
        raise RuntimeError(
            "Missing species directory!  Use species_dir "
            + "parameter or set $AIMS_SPECIES_DIR environment variable."
        )

    species_path = Path(species_dir)
    if not species_path.exists():
        raise RuntimeError(
            f"The requested species_dir {species_dir} does not exist")

    return species_path


# Write aims control.in files
@writer
def write_control(
    fd: TextIO | str | Path,
    atoms: Atoms,
    parameters: dict[str, Any],
    verbose_header: bool = False
):
    """
    Write the control.in file for FHI-aims

    Parameters
    ----------
    fd: TextIO | str | Path
        The file object to write to
    atoms: Atoms
        The Atoms object for the requested calculation
    parameters: dict[str, Any]
        The dictionary of all paramters for the calculation
    verbose_header: bool
        If True then explcitly list the paramters used to generate the
        control.in file inside the header

    """
    parameters = dict(parameters)

    if parameters["xc"] == "LDA":
        parameters["xc"] = "pw-lda"

    if "kpts" in parameters:
        mp = kpts2mp(atoms, parameters.pop("kpts"))
        dk = 0.5 - 0.5 / np.array(mp)
        parameters["k_grid"] = tuple(mp)
        parameters["k_offset"] = tuple(dk)

    parameters["species_dir"] = get_species_directory(
        parameters.pop("species_dir")
    )

    control_in = AimsControl(
        parameters=parameters,
        outputs=parameters.pop("output"),
    )

    geometry = AimsGeometry.from_atoms(atoms)

    control_in.write_file(
        geometry=geometry,
        writer=fd,
        header=get_aims_header(),
        verbose_header=verbose_header,
    )


@reader
def read_aims_output(
    fd: TextIO | str | Path,
    index: int | slice = -1,
    non_convergence_ok: bool = False
) -> Atoms | list[Atoms]:
    """
    Import FHI-aims output files with all data available

    Parameters
    ----------
    fd: TextIO | str | Path
        The file object to write to
    index: slice | int
        The images to return
    non_convergence_ok: bool
        True if a non-converged result is okay

    Returns
    -------
    Atoms | list[Atoms]
        The requested Atoms objects
    """
    output = AimsStdout(fd)

    if isinstance(index, int):
        loop_inds = [index]
    else:
        loop_inds = range(output.n_images)
        loop_inds = loop_inds[index]

    atoms_list = []
    for ind in loop_inds:
        image = output[ind]
        if not non_convergence_ok and (not image.converged):
            raise AimsParseError("The calculation did not converge properly.")
        atoms = image.geometry.ase_atoms
        atoms.calc = SinglePointDFTCalculator(
            atoms,
            energy=image.free_energy, # TARP: This is the force-consistent energy
            free_energy=image.free_energy,
            forces=image.forces,
            stress=image.stress,
            stresses=image.stresses,
            magmom=image.magmom,
            dipole=image.dipole,
            dielectric_tensor=image.dielectric_tensor,
            polarization=image.polarization,
        )
        atoms_list.append(atoms)
    if isinstance(index, int):
        return atoms_list[0]

    return atoms_list


def read_aims_results(
    fd: TextIO | str | Path,
    index: int | slice = -1,
    non_convergence_ok: bool = False,
    verbosity: str = "converged",
) -> dict[str, Any] | list[dict[str, Any]]:
    """
    Import FHI-aims output files with all data available as a dict

    Parameters
    ----------
    fd: TextIO | str | Path
        The file object to write to
    index: slice | int
        The images to return
    non_convergence_ok: bool
        True if a non-converged result is okay
    verbosity: str
        Results verbosity level, can be "single_point" or "all"
    Returns
    -------
    dict[str, Any] | list[dict[str, Any]]
        The requested results Dictionaries
    """
    output = AimsStdout(fd)

    if isinstance(index, int):
        loop_inds = [index]
    else:
        loop_inds = range(output.n_images)
        loop_inds = loop_inds[index]

    results = []
    for ind in loop_inds:
        image = output[ind]
        if not non_convergence_ok and (not image.converged):
            raise AimsParseError("The calculation did not converge properly.")
        results.append(image.get_results(verbosity=verbosity))

    if isinstance(index, int):
        return results[0]

    return results
