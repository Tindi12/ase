import numpy as np


class BFGSMethod:
    iotype = 'bfgs'
    methodname = 'BFGS'

    def __init__(self, hessian):
        self.hessian = hessian

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
        self.hessian += (
            np.outer(dgradient, dgradient) / a - np.outer(dg, dg) / b
        )

    def datafy(self):
        return self.hessian.ravel().tolist()

    @classmethod
    def undatafy(cls, hessian):
        n = int(np.round(len(hessian) ** 0.5))
        hessian = np.array(hessian).reshape(n, n)
        return cls(hessian)
