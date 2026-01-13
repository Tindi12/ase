import json
from dataclasses import dataclass

import numpy as np

from ase.io.jsonio import default, object_hook
from ase.io.trajectory import Trajectory
from ase.parallel import world


class Target:
    def __init__(self, atoms, fmax):
        self.optimizable = atoms.__ase_optimizable__()
        self.fmax = fmax

    def get_value(self):
        return self.optimizable.get_value()

    def get_gradient(self):
        forces = self.optimizable.atoms.get_forces()
        gradient = -forces.ravel()
        fnorm = get_maxforce(forces)
        converged = fnorm < self.fmax
        return ForceGradient(
            gradient=gradient,
            forces=forces,
            fnorm=fnorm,
            converged=converged,
        )

    def get_x(self):
        return self.optimizable.get_x()

    def set_x(self, x):
        self.optimizable.set_x(x)

    def gradient_norm(self, gradient):
        return self.optimizable.gradient_norm(gradient)

    def converged(self, gradient) -> bool:
        return self.gradient_norm(gradient) < self.fmax

    def initial_hessian(self, alpha=70.0) -> np.ndarray:
        return initial_position_hessian(self.optimizable.ndofs(), alpha)


def get_maxforce(forces) -> float:
    return np.linalg.norm(forces, axis=1).max()


class Optimizer:
    def __init__(
        self,
        target,
        method,
        trajectory=None,
        restartfile=None,
        comm=world,
        logfile='-',
        step=None,
    ):
        from ase.optimize.optimize import Log

        self.log = Log(logfile, comm)
        self.comm = comm
        self.target = target
        self.method = method
        self.trajectory = trajectory
        self.restartfile = restartfile
        # TODO We need both "restart from" and "save restart to", somehow.
        # Altough maybe that feature can come via a classmethod
        self.step = step

    def run(self, steps=None):
        for step in self.irun(steps):
            pass
        return step

    def irun(self, steps=None):
        if self.step is None:
            self.step = Step.start(self.target)
            yield self.step

        while not step.gradient_obj.converged:
            # (Both method and target change in this update)
            self.step = next_step(self.target, self.method, self.step)
            self._writefiles(step)
            yield self.step
            if self.step.i == steps:
                # What's best: raise or return?
                # steps should be additive probably (if we start from step N)?
                return

    def _writefiles(self, step):
        write_to_log(self.method, self.log, step)
        if self.trajectory is not None:
            write_to_traj(self.target, self.trajectory, self.comm)
        if self.restartfile is not None and self.comm.rank == 0:
            write_restartfile(self.restartfile, self.method, self.target, step)

    @classmethod
    def restart(cls, restartfile, calc, **kwargs):
        # Since this method has "calc", it doesn't really belong on this
        # class (we know about Targets etc. but not calcs).
        # Maybe therefore this should be a standalone function.
        target, method, step = read_restartfile(restartfile, calc)
        return cls(target=target, method=method, step=step, **kwargs)


@dataclass
class ForceGradient:
    gradient: np.ndarray
    forces: np.ndarray
    fnorm: float
    converged: bool

    def loginfo(self):
        return {'fmax': self.fnorm}


def initial_position_hessian(ndofs, alpha=70.0):
    return np.diag(np.full(ndofs, 70.0))


@dataclass
class Step:
    i: int
    x: np.ndarray
    gradient_obj: object
    value: float

    @classmethod
    def start(cls, target):
        step = cls(0, target.get_x(), target.get_gradient(), target.get_value())
        assert step.gradient_obj.gradient.shape == (len(step.x),)
        return step

    def datafy(self):
        return {
            'i': self.i,
            'x': self.x.tolist(),
            'gradient_obj': self.gradient_obj.datafy(),
            'value': self.value,
        }

    @classmethod
    def undatafy(cls, dct, gradient_obj):
        return cls(
            i=dct['i'],
            x=np.array(dct['x']),
            gradient_obj=gradient_obj,
            value=dct['value'],
        )


def next_step(target, method, step) -> Step:
    dx = method.compute_step(step.gradient_obj.gradient)
    # TODO We do not have maxstep right now.  This will not run the same
    # as legacy optimizations until we apply a maxstep.

    # Target may apply constraints or other magic, so we may not
    # get the same x back as the one we set.
    target.set_x(step.x + dx)

    newstep = Step(
        i=step.i + 1,
        x=target.get_x(),
        gradient_obj=target.get_gradient(),
        value=target.get_value(),
    )

    method.update(
        newstep.x,
        newstep.gradient_obj.gradient,
        step.x,
        step.gradient_obj.gradient,
    )
    return newstep


def write_to_log(method, log, step):
    loginfo = step.gradient_obj.loginfo()
    name = method.methodname
    txt = ' '.join(f'{key}={value:e}' for key, value in loginfo.items())
    msg = f'{name} i={step.i:4d} e={step.value:f} {txt}\n'
    log.write(msg)


def write_to_traj(target, trajpath, comm):
    with Trajectory(trajpath, comm=comm, mode='a') as traj:
        # TODO we are not setting metadata (like old optimizers)
        traj.write(target)


def read_images(trajpath):
    with Trajectory(trajpath) as traj:
        return [*traj]


def write_restartfile(restartfile, method, target, step):
    # TODO Unsafe if we just overwrite, we should backup/delete to prevent
    # accidental partial save

    # Still need some things, like maximum iterations.
    # How about trajectory writing, logfile settings, etc.?
    # General observers obviously cannot be saved.
    savedata = {
        'method': [method.iotype, method.datafy()],
        'target': [target.iotype, target.datafy()],
        'step': step.datafy(),
    }
    json_text = json.dumps(savedata, default=default)
    restartfile.write_text(json_text)


def read_restartfile(restartfile, calc):
    json_text = restartfile.read_text()
    dct = json.loads(json_text, object_hook=object_hook)

    assert {*dct} == {'method', 'target', 'step'}
    method_iotype, method_data = dct['method']
    target_iotype, target_dct = dct['target']

    if method_iotype == 'bfgs':
        from ase._4.optimize.bfgs import BFGSMethod
        method = BFGSMethod.undatafy(method_data)
    else:
        raise ValueError(f'No such method: {method_iotype}')

    if target_iotype == 'frechet':
        from ase._4.optimize.frechet import FrechetTarget

        target = FrechetTarget.undatafy(target_dct, calc)
    else:
        raise ValueError(f'No such target: {target_iotype}')

    gradient_obj = target.undatafy_gradient(dct['step']['gradient_obj'])
    step = Step.undatafy(dct['step'], gradient_obj)

    return target, method, step
