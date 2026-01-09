import json
from dataclasses import dataclass

import numpy as np

from ase.io.jsonio import default, object_hook
from ase.io.trajectory import Trajectory


class BFGSMethod:
    methodname = 'BFGS'

    def __init__(self, hessian):
        self.hessian = hessian

    @property
    def H(self):
        return self.hessian

    def compute_step(self, gradient):
        omega, vectors = np.linalg.eigh(self.hessian)
        return -vectors @ (gradient @ vectors / np.fabs(omega))

    def update(self, pos, gradient, pos0, gradient0):
        dpos = pos - pos0

        if np.abs(dpos).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return

        dgradient = gradient - gradient0
        a = dpos @ dgradient
        dg = self.hessian @ dpos
        b = dpos @ dg
        self.hessian -= (
            -np.outer(dgradient, dgradient) / a + np.outer(dg, dg) / b
        )

    def datafy(self):
        return self.hessian.ravel().tolist()

    @classmethod
    def undatafy(cls, hessian):
        n = int(np.round(len(hessian) ** 0.5))
        hessian = np.array(hessian).reshape(n, n)
        return cls(hessian)


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


def new_bfgs(target, method):
    step = Step.start(target)
    assert step.gradient_obj.gradient.shape == (len(step.x),)

    yield step
    yield from irun(target, method, step)


@dataclass
class Step:
    i: int
    x: np.ndarray
    gradient_obj: object
    value: float

    @classmethod
    def start(cls, target):
        return cls(0, target.get_x(), target.get_gradient(), target.get_value())

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


def irun(target, method, step=None):
    if step is None:
        step = Step.start(target)
        yield step

    while not step.gradient_obj.converged:
        # (Both method and target change in this update)
        step = next_step(target, method, step)
        yield step


def next_step(target, method, step) -> Step:
    dx = method.compute_step(step.gradient_obj.gradient)
    # We do not have maxstep right now.  This will not run the same
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
        # XXX we are not setting metadata (like old optimizers)
        traj.write(target)


def read_images(trajpath):
    with Trajectory(trajpath) as traj:
        return [*traj]


def write_restartfile(restartpath, method, target, step):
    # Unsafe if we just overwrite, we should backup/delete to prevent
    # accidental partial save

    # Still need some things, like maximum iterations.
    # How about trajectory writing, logfile settings, etc.?
    # General observers obviously cannot be saved.
    savedata = {
        'method': [method.methodname, method.datafy()],
        'target': target.datafy(),
        'step': step.datafy(),
    }
    json_text = json.dumps(savedata, default=default)
    restartpath.write_text(json_text)


def read_restartfile(restartpath, calc):
    json_text = restartpath.read_text()
    dct = json.loads(json_text, object_hook=object_hook)
    methodname, data = dct['method']
    if methodname == 'BFGS':
        method = BFGSMethod.undatafy(data)
    else:
        raise ValueError(f'No such method: {methodname}')

    # XXX Identity of target must be coded in restartfile as well.
    from ase._4.optimize.frechet import FrechetTarget

    target = FrechetTarget.undatafy(dct['target'], calc)
    gradient_obj = target.undatafy_gradient(dct['step']['gradient_obj'])
    step = Step.undatafy(dct['step'], gradient_obj)

    return target, method, step
