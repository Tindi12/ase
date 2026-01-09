import numpy as np
import pytest

from ase._4.optimize.bfgs import BFGSMethod
from ase._4.optimize.frechet import FrechetTarget
from ase._4.optimize.run import (
    Step,
    Target,
    irun,
    read_images,
    read_restartfile,
    run,
    write_restartfile,
    write_to_log,
    write_to_traj,
)
from ase.filters import FrechetCellFilter
from ase.optimize.bfgs import BFGS as OldBFGS
from ase.optimize.cellawarebfgs import CellAwareBFGS
from ase.optimize.optimize import Log
from ase.parallel import world


def setup_surface():
    from ase.build import fcc111
    from ase.calculators.emt import EMT

    rng = np.random.RandomState(42)
    atoms = fcc111('Au', size=(1, 2, 2), vacuum=5.0)
    atoms.rattle(stdev=0.01, rng=rng)
    cell = atoms.get_cell()
    cell[:2, :2] += 0.05 * rng.random((2, 2))
    atoms.set_cell(cell, scale_atoms=True)
    atoms.calc = EMT()
    return atoms


def test_surface():
    atoms = setup_surface()

    bfgs = OldBFGS(atoms)
    for _ in bfgs.irun(fmax=0.01):
        pass


def test_new_bfgs():
    atoms = setup_surface()
    target = Target(atoms, fmax=0.01)
    hessian = target.initial_hessian()
    step = run(target, BFGSMethod(hessian))
    assert step.i == 10
    assert step.gradient_obj.converged


def test_old_frechet():
    atoms = setup_surface()
    bfgs = CellAwareBFGS(
        FrechetCellFilter(atoms, exp_cell_factor=1.0, mask=[1, 1, 0, 0, 0, 1])
    )
    bfgs.run(fmax=0.001, smax=0.0001)


def test_new_bfgs_frechet():
    atoms = setup_surface()
    target = FrechetTarget(atoms, fmax=0.01, smax=0.00001)
    method = BFGSMethod(target.initial_hessian())
    step = run(target=target, method=method)
    assert step.gradient_obj.converged
    assert step.i == 18


def test_new_bfgs_frechet_files(tmp_path):
    comm = world

    atoms = setup_surface()
    fmax = 0.001
    smax = 0.0001
    target = FrechetTarget(atoms, fmax=fmax, smax=smax)
    hessian = target.initial_hessian()
    method = BFGSMethod(hessian)

    log = Log('-', comm)
    restartpath = tmp_path / 'restart.json'
    trajpath = tmp_path / 'opt.traj'
    trajpath.unlink(missing_ok=True)

    def writefiles():
        write_to_log(method, log, step)
        write_to_traj(target, trajpath, comm)
        write_restartfile(restartpath, method, target, step)

    step = Step.start(target)
    writefiles()

    for step in irun(target, method, step):
        writefiles()
        if step.i == 5:
            break

    firstpart_images = read_images(trajpath)

    assert len(firstpart_images) == 6

    halfway = restartpath.with_name('halfway.json')
    write_restartfile(halfway, method, target, step)

    for step in irun(target, method, step):
        writefiles()

    gradient_obj = step.gradient_obj

    ref = pytest.approx(0.837190)

    assert target.get_value() == ref
    assert gradient_obj.fnorm < fmax
    assert gradient_obj.snorm < smax
    assert step.i == 17

    images = read_images(trajpath)
    last_atoms = images[-1]

    assert len(images) == 18
    assert last_atoms.get_potential_energy() == ref

    from ase.calculators.emt import EMT

    print('restart')
    target, method, step = read_restartfile(halfway, EMT())

    trajpath = tmp_path / 'lastpath.traj'
    write_to_log(method, log, step)

    for step in irun(target, method, step):
        writefiles()

    lastpart_images = read_images(trajpath)
    assert len(firstpart_images) + len(lastpart_images) == len(images)
    last_atoms2 = lastpart_images[-1]
    assert last_atoms2.get_potential_energy() == pytest.approx(ref)
    assert last_atoms.positions == pytest.approx(last_atoms2.positions)
    assert last_atoms.cell == pytest.approx(last_atoms2.cell)
