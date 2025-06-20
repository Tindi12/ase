# fmt: off
from pathlib import Path
from io import StringIO
from typing import IO, Any, Dict, Type

import pytest

from ase import Atoms
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
from ase.optimize.optimize import Optimizer
from ase.optimize.precon import PreconFIRE, PreconLBFGS, PreconODE12r
from ase.optimize.sciopt import (
    SciPyFminBFGS,
    SciPyFminCG,
)

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


FMAX = 1e-3
ENERGY_TOLERANCE = 1e-4


@pytest.fixture(name="optimizer_class", params=optclasses)
def fixture_optimizer_class(request) -> Type[Optimizer]:
    optimizer_class: Type[Optimizer] = request.param
    return optimizer_class


@pytest.fixture(name="kwargs")
def fixture_kwargs(optimizer_class):
    kwargs = {}
    if optimizer_class is PreconLBFGS:
        kwargs["precon"] = None
    yield kwargs
    kwargs = {}


@pytest.fixture(name="logfile")
def fixture_logfile() -> IO:
    return StringIO()


@pytest.fixture(name="optimizer")
def fixture_optimizer(
    optimizer_class: Type[Optimizer],
    request: pytest.FixtureRequest,
    rattled_atoms: Atoms,
    logfile: IO,
    kwargs: Dict[str, Any],
) -> Optimizer:
    return optimizer_class(atoms=rattled_atoms, logfile=logfile, **kwargs)


@pytest.fixture(name="run_optimizer")
def fixture_run_optimizer(optimizer: Optimizer) -> bool:
    return optimizer.run(fmax=FMAX)


@pytest.fixture(name="final_energy")
def fixture_final_energy(rattled_atoms: Atoms, run_optimizer: bool) -> float:
    final_energy: float = rattled_atoms.get_potential_energy()
    return final_energy


@pytest.fixture(name="final_max_force")
def fixture_final_max_force(rattled_atoms: Atoms, run_optimizer: bool) -> float:
    final_max_force = max((rattled_atoms.get_forces() ** 2).sum(axis=1) ** 0.5)
    return final_max_force


@pytest.fixture(name="log_optimizer_run")
def fixture_log_optimizer_run(
    optimizer: Optimizer,
    reference_energy: float,
    final_max_force: float,
    final_energy: float,
) -> None:
    print()
    e_err = abs(final_energy - reference_energy)
    print(
        "{:>20}: fmax={:.05f} eopt={:.06f}, err={:06e}".format(
            optimizer.__class__.__name__, final_max_force, final_energy, e_err
        )
    )


@pytest.mark.usefixtures("log_optimizer_run")
class TestOptimizers:
    @staticmethod
    def test_should_obtain_final_state_with_energy_within_tolerance(
        reference_energy: float, final_energy: float) -> None:
        assert abs(final_energy - reference_energy) < ENERGY_TOLERANCE

    @staticmethod
    def test_should_reduce_forces_below_tolerance(
        final_max_force: float
    ) -> None:
        assert final_max_force < FMAX


@pytest.mark.optimize()
@pytest.mark.filterwarnings("ignore: estimate_mu")
def test_path(testdir, optcls, atoms, kwargs):
    fmax = 0.01
    traj, log = Path('trajectory.traj'), Path('relax.log')
    with optcls(atoms, logfile=log, trajectory=traj, **kwargs) as opt:
        is_converged = opt.run(fmax=fmax)
    assert is_converged  # chec
