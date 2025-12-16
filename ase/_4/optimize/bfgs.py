import numpy as np


class BFGSState:
    def __init__(self, hessian):
        self.hessian = hessian

    @property
    def H(self):
        return self.hessian

    def update(self, pos, forces, pos0, forces0):
        dpos = pos - pos0

        if np.abs(dpos).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return

        dforces = forces - forces0
        a = dpos @ dforces
        dg = self.hessian @ dpos
        b = dpos @ dg
        self.hessian -= np.outer(dforces, dforces) / a + np.outer(dg, dg) / b

    def compute_step(self, gradient):
        omega, vectors = np.linalg.eigh(self.hessian)
        # Maybe we should check for negative eigenvalues
        return -vectors @ (gradient @ vectors / np.fabs(omega))
