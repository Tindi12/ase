#!/usr/bin/env python
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def _prepare_axis(axis, tol):
    axis = np.asarray(axis, dtype=float).copy()
    norm = np.linalg.norm(axis)
    if norm < tol:
        raise ValueError('Axis vector cannot be zero')
    return axis / norm


class SymmetryOperation:
    """
    Base class for symmetry operations, e.g., Rotation

    The matrices are formatted for coordinates in row vectors, so
    X_new = X @ M; X positions in rows, M matrix
    """

    def __init__(self, matrix, order=2, axis=None, tol=1e-10):

        self.order = order
        self.axis = axis

        matrix = np.asarray(matrix)
        if matrix.shape != (3, 3):
            raise ValueError(f"Matrix must be 3x3, got shape {matrix.shape}")
        if not np.allclose(matrix.T @ matrix, np.eye(3), atol=tol):
            raise ValueError("Matrix is not orthogonal")

        self.matrix = matrix

    @staticmethod
    def _mirror_matrix(normal):
        return np.eye(3) - 2 * np.outer(normal, normal)

    @staticmethod
    def _rotation_matrix(axis, order):
        if not isinstance(order, int) or order < 1:
            raise ValueError(f'Invalid symmetry order ({order})')
        return Rot.from_rotvec(axis * 2 * np.pi / order).as_matrix()

    def apply(self, positions, inverse=False):
        return positions @ (self.matrix.T if inverse else self.matrix)

    def is_proper(self, tol):
        # True if proper rotation
        return np.isclose(np.linalg.det(self.matrix), 1.0, atol=tol)


class Identity(SymmetryOperation):
    def __init__(self):
        matrix = np.eye(3)
        super().__init__(matrix, order=1)


class Inversion(SymmetryOperation):
    def __init__(self):
        matrix = -np.eye(3)
        super().__init__(matrix)


class Rotation(SymmetryOperation):
    def __init__(self, axis, order, tol):
        axis = _prepare_axis(axis, tol)
        matrix = SymmetryOperation._rotation_matrix(axis, order).T
        super().__init__(matrix, order, axis=axis)


class Mirror(SymmetryOperation):
    def __init__(self, axis, tol):
        axis = _prepare_axis(axis, tol)
        matrix = SymmetryOperation._mirror_matrix(axis).T
        super().__init__(matrix, axis=axis)


class ImproperRotation(SymmetryOperation):
    def __init__(self, axis, order, tol):
        axis = _prepare_axis(axis, tol)
        matrix = SymmetryOperation._mirror_matrix(axis)
        matrix @= SymmetryOperation._rotation_matrix(axis, order)
        matrix = matrix.T
        super().__init__(matrix, order, axis=axis)
