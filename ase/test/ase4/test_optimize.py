import numpy as np
import pytest

from ase._4.optimize.bfgs import BFGSMethod
from ase._4.optimize.frechet import FrechetTarget
from ase._4.optimize.run import (
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


class Optimizer:
    def __init__(
        self,
        target,
        method,
        trajectory=None,
        restartpath=None,
        comm=world,
        logfile='-',
        step=None,
    ):
        self.log = Log(logfile, comm)
        self.comm = comm
        self.target = target
        self.method = method
        self.trajectory = trajectory
        self.restartpath = restartpath
        # We need both "restart from" and "save restart to", somehow.
        # Altough maybe that feature can come via a classmethod
        self.step = step

    def run(self, steps=None):
        for step in self.irun(steps):
            pass
        return step

    def irun(self, steps=None):
        for step in irun(self.target, self.method, step=self.step):
            self.step = step
            self._writefiles(step)
            yield step
            if step.i == steps:
                # What's best: raise or return?
                # steps should be additive probably?
                return

    def _writefiles(self, step):
        write_to_log(self.method, self.log, step)
        if self.trajectory is not None:
            write_to_traj(self.target, self.trajectory, self.comm)
        if self.restartpath is not None:
            write_restartfile(self.restartpath, self.method, self.target, step)

    @classmethod
    def restart(cls, restartfile, calc, **kwargs):
        target, method, step = read_restartfile(restartfile, calc)
        return cls(target=target, method=method, step=step, **kwargs)


def test_new_bfgs_frechet_files(tmp_path):
    atoms = setup_surface()
    fmax = 0.001
    smax = 0.0001
    target = FrechetTarget(atoms, fmax=fmax, smax=smax)
    hessian = target.initial_hessian()
    method = BFGSMethod(hessian)

    restartpath = tmp_path / 'restart.json'
    trajpath = tmp_path / 'opt.traj'
    # trajpath.unlink(missing_ok=True)

    # Three use cases when starting a relaxation:
    #  * Wipe old files, start from scratch
    #  * Load from old files, and append (overwriting restartfile)
    #  * Load from old files, write to some other files
    opt = Optimizer(target, method, trajpath, restartpath)
    opt.run(steps=5)

    firstpart_images = read_images(trajpath)

    assert len(firstpart_images) == 6
    print(' --- first part done and saved, now continue ---')

    halfway = restartpath.with_name('halfway.json')
    write_restartfile(halfway, opt.method, opt.target, opt.step)

    step = opt.run()

    ref = pytest.approx(0.837190)
    gradient_obj = step.gradient_obj
    assert target.get_value() == ref
    assert gradient_obj.fnorm < fmax
    assert gradient_obj.snorm < smax
    assert step.i == 17

    images = read_images(trajpath)
    last_atoms = images[-1]

    assert len(images) == 18
    assert last_atoms.get_potential_energy() == ref

    from ase.calculators.emt import EMT

    print('done relaxing, now restart from checkpoint')
    lastpart_traj = tmp_path / 'lastpart.traj'

    opt = Optimizer.restart(halfway, EMT(), trajectory=lastpart_traj)
    opt.run()

    # If we want to (for example) "get the energy" or "get the calculator"
    # we will need to poke into the opt.target.xxx.  Maybe there should be
    # unified way to export the "domain stuff" similar to iterimages()

    lastpart_images = read_images(lastpart_traj)
    assert len(firstpart_images) + len(lastpart_images) == len(images)
    last_atoms2 = lastpart_images[-1]
    assert last_atoms2.get_potential_energy() == pytest.approx(ref)
    assert last_atoms.positions == pytest.approx(last_atoms2.positions)
    assert last_atoms.cell == pytest.approx(last_atoms2.cell)
