# fmt: off

"""Defines class/functions to write input and parse output for FHI-aims."""
import os
import time
import warnings
from pathlib import Path
from typing import Any, Iterable, TextIO

import numpy as np
from pyfhiaims.control.control import AimsControlIn
from pyfhiaims.geometry.atom import FHIAimsAtom
from pyfhiaims.geometry.geometry import AimsGeometry
from pyfhiaims.output_parser.aims_outputs import AimsOutput
from pyfhiaims.species_defaults.species import SpeciesDefaults

from ase import Atoms
from ase.calculators.calculator import kpts2mp
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.constraints import (
    FixAtoms,
    FixCartesian,
    FixCartesianParametricRelations,
    FixScaledParametricRelations,
)
from ase.data import atomic_numbers
from ase.units import Ang, fs
from ase.utils import deprecated, reader, writer

v_unit = Ang / (1000.0 * fs)

LINE_NOT_FOUND = object()


class AimsParseError(Exception):
    """Exception raised if an error occurs when parsing an Aims output file"""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def singular(key: str) -> str:
    """Returns the singular form of a string."""
    if key[-6:] == "_atoms":
        return key[:-6]
    if key[-4:] == "sses":
        return key[:-4] + "ss"
    if key[-3:] == "ies":
        return key[:-3] + "y"
    if key[-2:] == "es":
        return key[:-2] + "e"
    if key[-1:] == "s":
        return key[:-1]

    return key


def atoms2aimsgeo(atoms: Atoms) -> AimsGeometry:
    """
    Convert from Atoms to AimsGeometry

    Args:
        atoms (Atoms): The Atoms object to convert

    Returns:
        AimsGeometry: The corresponing AimsGeometry

    """
    geometry_props = {k: v for k, v in atoms.info.items()
                        if k in AimsGeometry._get_property_names()}

    atomic_props = {k: v for k, v in atoms.info.items()
                    if k in AimsGeometry._get_property_names(atomic=True)}

    symmetry_params = [[], []]
    symmetry_n_params = None
    symmetry_lv = None
    symmetry_frac = None
    cart_const = {}
    for const in atoms.constraints:
        if isinstance(const, FixScaledParametricRelations):
            symmetry_params[1] = const.params
            symmetry_frac = const.expressions
        elif isinstance(const, FixCartesianParametricRelations):
            symmetry_params[0] = const.params
            symmetry_lv = const.expressions
        elif isinstance(const, FixAtoms):
            cart_const.update(
                {
                    ind: (True, True, True) for ind in const.get_indices()
                }
            )
        elif isinstance(const, FixCartesian):
            for ind in const.index:
                cart_const[ind] = tuple(mm for mm in const.mask)

    atomic_props["velocities"] = [
        v if any(v) else None for v in atoms.get_velocities()
    ]
    aims_atoms = []
    for i_at, atom in enumerate(atoms):
        atom_props = {
            singular(k): v[i_at] for k, v in atomic_props.items()
        }
        const = atom_props.pop("nuclear_constraint", None)
        aims_atoms.append(FHIAimsAtom(
            symbol=atom.symbol,
            position=atom.position,
            initial_charge=atom.charge,
            initial_moment=atom.magmom,
            constraints=cart_const.get(i_at),
            **atom_props
            )
        )

    if symmetry_lv is not None or symmetry_frac is not None:
        if symmetry_lv is None or symmetry_frac is None:
            raise ValueError(
                "Both symmetry_lv and symmetry_frac must be defined"
            )
        symmetry_n_params = (
            len(symmetry_params[0]),
            len(symmetry_params[1]),
            len(symmetry_params[0]) + len(symmetry_params[1])
        )
        symmetry_params = symmetry_params[0] + symmetry_params[1]
    else:
        symmetry_params = None

    return AimsGeometry(
        atoms=aims_atoms,
        lattice_vectors=None if atoms.cell.rank < 3 else atoms.cell,
        symmetry_n_params=symmetry_n_params,
        symmetry_params=symmetry_params,
        symmetry_lv=symmetry_lv,
        symmetry_frac=symmetry_frac,
        **geometry_props
    )


def aimsgeo2atoms(geo_in: AimsGeometry) -> Atoms:
    """
    Convert from AimsGeometry to Atoms

    Args:
        geo_in (AimsGeometry): geometry object to convert

    Returns:
        Atoms: The correponding Atoms object

    """
    info_entries = [p for p in geo_in._get_property_names()
                    if p not in (
                        "atoms",
                        "lattice_vectors",
                        "species_dict"
                    )] + \
                    [p for p in geo_in._get_property_names(atomic=True)
                    if p not in (
                        "symbols",
                        "numbers",
                        "positions",
                        "fractional_positions",
                        "velocities",
                        "initial_charges",
                        "initial_moments",
                        "species_block",
                        "n_atoms",
                        "masses",
                        "nuclear_charges")
                    ]

    if geo_in.lattice_vectors is not None:
        cell = geo_in.lattice_vectors
    else:
        cell = (0, 0, 0)
    atoms = Atoms(
        numbers=geo_in.numbers,
        positions=geo_in.positions,
        velocities=[v if v is not None else [0, 0, 0]
                    for v in geo_in.velocities],
        magmoms=geo_in.magnetic_moments,
        charges=geo_in.initial_charges,
        pbc=geo_in.lattice_vectors is not None,
        cell=cell,
        info={e: getattr(geo_in, e) for e in info_entries},
    )

    fix_params = []
    if (
        (geo_in.symmetry_n_params is not None) and
        (np.sum(geo_in.symmetry_n_params) > 0)
    ):
        fix_params.append(
            FixCartesianParametricRelations.from_expressions(
                list(range(3)),
                geo_in.symmetry_params[:geo_in.symmetry_n_params[0]],
                [expr for exprs in geo_in.symmetry_lv for expr in exprs],
                use_cell=True,
            )
        )

        fix_params.append(
            FixScaledParametricRelations.from_expressions(
                list(range(len(atoms))),
                geo_in.symmetry_params[geo_in.symmetry_n_params[0]:],
                [expr for exprs in geo_in.symmetry_frac for expr in exprs],
            )
        )

    fix_cart_const = []
    fixed_atoms = []
    for index, constraint in enumerate(geo_in.nuclear_constraints):
        if constraint is None:
            continue

        if all(constraint):
            fixed_atoms.append(index)
        elif any(constraint):
            fix_cart_const.append(FixCartesian(index, constraint))

    if len(fixed_atoms) > 0:
        fix_cart_const.insert(0, FixAtoms(fixed_atoms))

    atoms.set_constraint(fix_cart_const + fix_params)

    return atoms


# Read aims geometry files
@reader
def read_aims(fd: TextIO | str | Path, apply_constraints=True) -> Atoms:
    """
    Import FHI-aims geometry type files.

    Reads unitcell, atom positions and constraints from
    a geometry.in file.
    """
    lines = [line for line in fd.readlines()]
    geometry = AimsGeometry.from_strings(lines)
    atoms = aimsgeo2atoms(geometry)

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
    geometry = atoms2aimsgeo(atoms)
    if scaled and not np.all(atoms.pbc):
        raise ValueError(
            "Requesting scaled for a calculation where scaled=True, but "
            "the system is not periodic")

    if geo_constrain:
        if not scaled and np.all(atoms.pbc):
            warnings.warn(
                "Setting scaled to True because a symmetry_block is detected."
            )
            scaled = True
        elif not np.all(atoms.pbc):
            warnings.warn(
                "Parameteric constraints can only be used in periodic systems."
            )
            geo_constrain = False

    if not geo_constrain:
        geometry.symmetry_frac = None
        geometry.symmetry_lv = None
        geometry.symmetry_params = None
        geometry.symmetry_n_params = None

    if ghosts is not None:
        assert len(ghosts) == len(atoms)
        for gg, ghost in enumerate(ghosts):
            if ghost == 1:
                geometry.atoms[gg].is_empty = True

    wrap = wrap and not geo_constrain
    if scaled:
        for atom in geometry.atoms:
            atom.set_fractional(atoms.cell.array, wrap)

    if not write_velocities:
        for atom in geometry.atoms:
            atom.velocity = None

    fd.write(get_aims_header())

    # If writing additional information is requested via info_str:
    if info_str is not None:
        fd.write("\n# Additional information:\n")
        if isinstance(info_str, list):
            fd.write("\n".join([f"#  {s}" for s in info_str]))
        else:
            fd.write(f"# {info_str}")
        fd.write("\n")

    fd.write("#=======================================================\n")
    fd.write(geometry.to_string())


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

    species_dir = get_species_directory(parameters.pop("species_dir"))
    tiers = parameters.pop("tier", None)
    plus_u = parameters.pop("plus_u", None)

    outputs = parameters.pop("output")

    control_in = AimsControlIn(
        parameters=parameters,
        outputs=outputs,
    )

    geometry = atoms2aimsgeo(atoms)
    if isinstance(tiers, int):
        tiers = {sym: tiers for sym in np.unique(geometry.symbols)}
    elif tiers is not None:
        assert all([sym in tiers for sym in np.unique(geometry.symbols)])

    for sym in np.unique(geometry.symbols):
        sf = f"{species_dir}/{atomic_numbers[sym]:02}_{sym}_default"
        species = SpeciesDefaults.from_file(sf)
        if tiers is not None:
            end_activate = min(tiers[sym], species.basis_set.n_tiers) + 1
            for tt in range(1, end_activate):
                species.basis_set.activate_tier(tt)
            for tt in range(end_activate, species.basis_set.n_tiers + 1):
                species.basis_set.deactivate_tier(tt)

        if plus_u is not None:
            species.plus_u = plus_u.get(sym)
        geometry.set_species(sym, species)

    fd.write(get_aims_header())
    fd.write(control_in.get_content(geometry, verbose_header))


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
    lines = [line.strip() for line in fd.readlines()]
    output = AimsOutput.from_aims_out_content(lines)

    if isinstance(index, int):
        loop_inds = [index]
    else:
        loop_inds = range(output.n_images)
        loop_inds = loop_inds[index]

    atoms_list = []
    for ind in loop_inds:
        image = output.get_image(ind)
        if not non_convergence_ok and (not image.converged):
            raise AimsParseError("The calculation did not converge properly.")
        atoms = aimsgeo2atoms(image.geometry)
        atoms.calc = SinglePointDFTCalculator(
            atoms,
            energy=image["energy"],
            free_energy=image["free_energy"],
            forces=image["forces"],
            stress=image["stress"],
            stresses=image["stresses"],
            magmom=image["magmom"],
            dipole=image["dipole"],
            dielectric_tensor=image["dielectric_tensor"],
            polarization=image["polarization"],
        )
        atoms_list.append(atoms)
    if isinstance(index, int):
        return atoms_list[0]

    return atoms_list


@reader
def read_aims_results(
    fd: TextIO | str | Path,
    index: int | slice = -1,
    non_convergence_ok: bool = False
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

    Returns
    -------
    dict[str, Any] | list[dict[str, Any]]
        The requested results Dictionaries
    """
    lines = [line.strip() for line in fd.readlines()]
    output = AimsOutput.from_aims_out_content(lines)

    if isinstance(index, int):
        loop_inds = [index]
    else:
        loop_inds = range(output.n_images)
        loop_inds = loop_inds[index]

    results = []
    for ind in loop_inds:
        image = output.get_image(ind)
        if not non_convergence_ok and (not image.converged):
            raise AimsParseError("The calculation did not converge properly.")
        results.append(image._results)

    if isinstance(index, int):
        return results[0]

    return results
