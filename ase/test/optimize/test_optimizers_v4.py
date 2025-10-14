# fmt: off
from pathlib import Path

import pytest

from ase._4.atoms import Atoms
from ase._4.calculators.emt import EMT
from ase._4.atoms import PotentialEnergySurface

from ase.build import bulk
from ase.optimize import (
    BFGS,
    FIRE,
    LBFGS,
    BFGSLineSearch,
    GoodOldQuasiNewton,
    GPMin,
    LBFGSLineSearch,
    MDMin,
    ODE12r,
)
from ase.optimize.precon import PreconFIRE, PreconLBFGS, PreconODE12r
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG

optclasses = [
    MDMin,
    FIRE,
    LBFGS,
    LBFGSLineSearch,
    BFGSLineSearch,
    BFGS,
    GoodOldQuasiNewton,
    GPMin,
    SciPyFminCG,
    SciPyFminBFGS,
    PreconLBFGS,
    PreconFIRE,
    ODE12r,
    PreconODE12r,
]


@pytest.fixture(name="ref_atoms")
def fixture_ref_atoms():
    ref_atoms = Atoms.from_v3atoms(bulk("Au"))
    return ref_atoms


@pytest.fixture(name="ref_calc")
def fixture_ref_calc():
    ref_calc = EMT()
    return ref_calc


@pytest.fixture(name="pes")
def fixture_pes(ref_atoms, ref_calc):
    atoms = ref_atoms * (2, 2, 2)
    floor = 0.45

    atoms.rattle(stdev=0.1, seed=7)
    results = ref_calc.evaluate(atoms, properties=["energy"])
    e_unopt = results.properties['energy']
    assert e_unopt > floor

    pes = PotentialEnergySurface(atoms, ref_calc)
    return pes

@pytest.fixture(name="optcls", params=optclasses)
def fixture_optcls(request):
    optcls = request.param
    return optcls


@pytest.fixture(name="kwargs")
def fixture_kwargs(optcls):
    kwargs = {}
    if optcls is PreconLBFGS:
        kwargs["precon"] = None
    yield kwargs
    kwargs = {}


@pytest.mark.optimize()
@pytest.mark.filterwarnings("ignore: estimate_mu")
def test_optimize(optcls, pes, ref_atoms, kwargs):
    """Test if forces can be converged using the optimizer."""
    fmax = 0.01
    with optcls(pes, **kwargs) as opt:
        is_converged = opt.run(fmax=fmax)
    assert is_converged  # check if opt.run() returns True when converged

    results = pes.calc.evaluate(pes.atoms, properties=["forces", "energy"])
    forces = results.properties["forces"]
    final_fmax = max((forces**2).sum(axis=1) ** 0.5)
    ref_energy = pes.calc.evaluate(ref_atoms,
                                   properties=["energy"]).properties["energy"]
    e_opt = results.properties["energy"] * len(ref_atoms) / len(pes.atoms)
    e_err = abs(e_opt - ref_energy)

    print(f"{optcls.__name__:>20}:", end=" ")
    print(f"fmax={final_fmax:.05f} eopt={e_opt:.06f} err={e_err:06e}")

    assert final_fmax < fmax
    assert e_err < 1.75e-5  # (This tolerance is arbitrary)


@pytest.mark.optimize()
def test_unconverged(optcls, pes, kwargs):
    """Test if things work properly when forces are not converged."""
    fmax = 1e-9  # small value to not get converged
    with optcls(pes, **kwargs) as opt:
        opt.run(fmax=fmax, steps=1)  # only one step to not get converged
    gradient = opt.optimizable.get_gradient()
    assert not opt.converged(gradient)
    assert opt.todict()["fmax"] == 1e-9


def test_run_twice(optcls, pes, kwargs):
    """Test if `steps` increments `max_steps` when `run` is called twice."""
    fmax = 1e-9  # small value to not get converged
    steps = 5
    with optcls(pes, **kwargs) as opt:
        opt.run(fmax=fmax, steps=steps)
        opt.run(fmax=fmax, steps=steps)
    assert opt.nsteps == 2 * steps
    assert opt.max_steps == 2 * steps


@pytest.mark.optimize()
@pytest.mark.filterwarnings("ignore: estimate_mu")
def test_path(testdir, optcls, pes, kwargs):
    fmax = 0.01
    traj, log = Path('trajectory.traj'), Path('relax.log')
    with optcls(pes, logfile=log, trajectory=traj, **kwargs) as opt:
        is_converged = opt.run(fmax=fmax)
    assert is_converged  # check if opt.run() returns True when converged
