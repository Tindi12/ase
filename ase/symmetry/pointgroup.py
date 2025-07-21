from functools import cached_property
from itertools import combinations, product

import numpy as np
import scipy as sp

from ase import Atom
from ase.symmetry.operations import (
    Identity,
    ImproperRotation,
    Inversion,
    Mirror,
    Rotation,
)
from ase.visualize import view


class PointGroupAnalyzer:
    """
    Class for analysis of molecular point groups
    Mainly adapted from pymatgen, available under MIT license:
    The MIT License (MIT) Copyright (c) 2011-2012 MIT & The Regents of the
    University of California, through Lawrence Berkeley National Laboratory
    and from Pypi pointgroup available under MIT license:
    Copyright (c) 2023 Efrem Bernuz and Abel Carreras

    atoms : an ASE atoms object
    eigtol : float
        tolerance of inertia eigenvalues, normalized by trace. Half the
        tolerance of other codes like pymatgen or pypi pointgroup, as
        those normalize by half the trace
    angtol : float
        angle tolerance in degrees
    disttol : float
        distance tolerance
    hardtol : float
        other tolerances
    """

    def __init__(self, atoms, eigtol=0.005, angtol=4., disttol=0.2,
                 hardtol=1e-6):

        self.atoms = self._center_on_center_of_mass(atoms)
        self.eigtol = eigtol
        self.angtol = np.deg2rad(angtol)
        self.disttol = disttol
        self.hardtol = hardtol

        self.sintol = np.sin(self.angtol)
        self.costol = np.cos(self.angtol)
        self.pos = atoms.get_positions()
        self._pos_bak = self.pos.copy()  # Devel, test if self.pos is modified
        self._kdtree = sp.spatial.KDTree(self.pos)
        self.symbols = atoms.get_chemical_symbols()
        self._mass_check()

        eigs, eigvecs = self.atoms.get_moments_of_inertia(vectors=True)
        self.principal_axes = eigvecs

        # Normalization by trace. Other codes normalize by 0.5 * trace
        trace = sum(eigs)
        if trace != 0:
            self.normalized_moments_of_inertia = eigs / trace
        else:
            self.normalized_moments_of_inertia = eigs

        # Make atom clusters that cannot be mappe on each other by symmetry
        self.clusters = self._group_atoms_by_symbol_and_norm()

    @property
    def pointgroup(self):
        """Returns the Schoenflies symbol, e.g., C2v"""
        return self._pointgroup_and_symmetries[0]

    @property
    def geometry(self):
        """Monatomic, linear or nonlinear"""
        if len(self.atoms) == 1:
            return 'monatomic'
        elif self.pointgroup in {'C*v', 'D*h'}:
            return 'linear'
        return 'nonlinear'

    @property
    def symmetry_number(self):
        """Rotation symmetry number"""
        n_rot_mappings = 1
        for symm_op in self._pointgroup_and_symmetries[1]:
            if isinstance(symm_op, Rotation):
                n_rot_mappings += symm_op.order - 1
        return n_rot_mappings

    @property
    def symmetry_operations(self):
        """List of SymmetryOperations"""
        return [Identity()] + self._pointgroup_and_symmetries[1]

    @cached_property
    def _pointgroup_and_symmetries(self):
        schoenflies, symm_ops = self._calc_schoenflies_and_symmetries()
        pointgroup = self._rename_pointgroup(schoenflies)
        return pointgroup, symm_ops

    def visualize_symmetry_axes(self, symm_ops):
        if not isinstance(symm_ops, list):
            symm_ops = [symm_ops]
        atoms = self.atoms.copy()
        for symm_op in symm_ops:
            axis = symm_op.axis
            if axis is None:
                continue
            atoms.append(Atom('Ar', 4 * axis))
            atoms.append(Atom('Ar', -4 * axis))
        view(atoms)

    def _rename_pointgroup(self, schoenflies):
        """
        Equivalent function of other codes
        Probably not necessary
        """
        return schoenflies

    def _calc_schoenflies_and_symmetries(self):
        """
        Main calculation of schoenflies symbol and symmetry operations

        Returns:

        Schoenflies symbol : str
            symbol for point group, e.g. D2h
        Symmetries : list of SymmetryOperation

        """

        if len(self.atoms) == 1:
            return 'Kh', []

        if self.normalized_moments_of_inertia[0] < self.eigtol:
            return self._linear()

        diffs = np.diff(self.normalized_moments_of_inertia)
        if np.all(diffs >= self.eigtol):
            return self._asymmetric_top()
        if np.all(diffs < self.eigtol):
            return self._spherical_top()
        if diffs[0] < self.eigtol:
            return self._symmetric_top(main_ind=2)
        if diffs[1] < self.eigtol:
            return self._symmetric_top(main_ind=0)

    def _is_valid(self, symm_op, transform_tol=None):
        """
        Check if SymmetryOperation is valid

        Inputs:

        symm_op : SymmetryOperation
        transform_tol : float
            tolerance for position deviation after symmetry transformation.
            Needs to be larger than general distance tolerance and defaults
            to twice as large
        """

        if transform_tol is None:
            if isinstance(symm_op, (Rotation, ImproperRotation)):
                transform_tol = 2 * self.disttol
            else:
                transform_tol = self.disttol

        transformed_positions = self.pos.copy()
        for step in range(symm_op.order - 1):
            transformed_positions = symm_op.apply(transformed_positions)
            distances, inds = self._kdtree.query(transformed_positions)
            if np.all(distances <= transform_tol):
                symbols_rotated = [self.symbols[i] for i in inds]

                if self.symbols != symbols_rotated:
                    return False
            else:
                return False

        return True

    def _linear(self):
        """
        Inertia indicates linear molecule.
        It is possible for a molecule with two heavy atoms and a light one,
        like HOBr, can have a close-to-zero moment of inertia
        """

        # Check that all atoms are within angle tolerance on principal axis
        if not np.all(self._on_line(self.principal_axes[0])):
            return self._asymmetric_top()

        rot = Rotation(self.principal_axes[2], 2, tol=self.hardtol)
        if self._is_valid(rot):

            return 'D*h', [rot]
        else:
            return 'C*v', []

    def _asymmetric_top(self):
        """Three distinct moments of inertia"""

        rots = []

        # Look for C2 axes
        for vec in self.principal_axes:
            rot = Rotation(vec, 2, tol=self.hardtol)
            if self._is_valid(rot):
                rots.append(rot)

        if len(rots) == 3:
            schoenflies, mirrors = self._dihedral(rots)
            return schoenflies, rots + mirrors

        # We have to search for axes not aligned with principal axes
        symmetry_axes = [r.axis.copy() for r in rots]
        # groups = self._group_atoms_by_symbol_and_norm()
        for cluster in self.clusters:
            for i1, i2 in combinations(cluster, 2):
                axis = 0.5 * (self.pos[i1] + self.pos[i2])
                if self._is_close_to_symmetry_axis(axis, symmetry_axes):
                    continue
                rot = Rotation(axis, order=2, tol=self.hardtol)
                if self._is_valid(rot):
                    rots.append(rot)
                    symmetry_axes.append(rot.axis.copy())

        if len(rots) == 3:
            schoenflies, mirrors = self._dihedral(rots)
            return schoenflies, rots + mirrors
        elif len(rots) == 0:
            schoenflies, symm_ops = self._no_rot_sym()
            return schoenflies, symm_ops
        else:
            schoenflies, symm_ops = self._cyclic(rots)
            return schoenflies, rots + symm_ops

    def _get_smallest_cluster_off_line(self, axis):
        """
        Returns the smallest atom cluster after atoms are removed on the axis
        """
        mask = self._on_line(axis)
        off_inds = np.arange(len(self.pos))[~mask]
        filtered_clusters = [
            cl[np.isin(cl, off_inds)] for cl in self.clusters
            if len(cl[np.isin(cl, off_inds)]) > 0
        ]
        sorted_clusters = sorted(filtered_clusters, key=lambda x: len(x))
        return sorted_clusters[0]

    def _symmetric_top(self, main_ind):
        """Two moments of inertia equal, third distinct"""

        main_axis = self.principal_axes[main_ind]
        rots = []
        smallest_cluster = self._get_smallest_cluster_off_line(main_axis)

        # Get main Cn axis
        for order in range(len(smallest_cluster) + 1, 1, -1):
            if len(smallest_cluster) % order == 0:
                rot = Rotation(main_axis, order, tol=self.hardtol)
                if self._is_valid(rot):
                    rots.append(rot)
                    break

        # Get perpendicular C2 axes
        if len(rots) > 0:
            found_nC2s = 0
            max_nC2s = order
            symmetry_axes = []

            if len(smallest_cluster) == 2:
                i1, i2 = smallest_cluster
                # Two possibilites: Map on itself or map on the other
                axis1 = self.pos[i1] - self.pos[i2]
                axis2 = np.cross(axis1, main_axis)
                for axis in [axis1, axis2]:
                    axis /= np.linalg.norm(axis)
                    rot = Rotation(axis, 2, tol=self.hardtol)
                    if self._is_valid(rot):
                        rots.append(rot)
                        found_nC2s += 1
            else:
                # Rotation axes must be defined by atoms of equal distance
                # to the plane through the center of mass whose normal is the
                # main axis. Thus, project on the axis
                subclusters = self._subdivide_group_by_projection(
                    smallest_cluster,
                    main_axis,
                    pairs=True,
                )
                subcl_pairs, subcl_central = subclusters
                if subcl_central is not None:
                    # Atoms in the central plane
                    ind_pairs = combinations(subcl_central, 2)
                    if len(subcl_central) == 4:
                        # Special case where midpoints can't cover the axes
                        # that go through atoms
                        fictional_pairs = [(i, i) for i in subcl_central]
                        ind_pairs = list(ind_pairs) + fictional_pairs
                elif len(subcl_pairs) != 0:
                    # Atoms in two planes of equal distance to central plane
                    ind_pairs = product(*subcl_pairs[0])
                else:
                    ind_pairs = []

                for i1, i2 in ind_pairs:
                    axis = 0.5 * (self.pos[i1] + self.pos[i2])
                    if self._is_close_to_symmetry_axis(axis, symmetry_axes):
                        continue
                    axis /= np.linalg.norm(axis)
                    if np.dot(axis, main_axis) > self.sintol:
                        continue
                    rot = Rotation(axis, 2, tol=self.hardtol)
                    if self._is_valid(rot):
                        rots.append(rot)
                        symmetry_axes.append(rot.axis.copy())
                        if found_nC2s >= max_nC2s:
                            break

        if len(rots) >= 2:
            schoenflies, mirrors = self._dihedral(rots)
            return schoenflies, rots + mirrors
        elif len(rots) == 1:
            schoenflies, symm_ops = self._cyclic(rots)
            return schoenflies, rots + symm_ops
        else:
            # Either no rotations exist, or accidental asymmetric top.
            # Better to check asymmetric top
            return self._asymmetric_top()

    def _no_rot_sym(self):
        """No rotation symmetries (C1, Cs, Ci)"""
        inv = Inversion()
        if self._is_valid(inv):
            return 'Ci', [inv]
        else:
            mirrors = []
            normals = []
            for axis in self.principal_axes:
                _, new_mirrors = self._find_mirrors(axis)
                for m in new_mirrors:
                    if not self._is_close_to_symmetry_axis(m.axis, normals):
                        mirrors.append(m)
                        normals.append(m.axis.copy())
            if len(mirrors) > 1:
                raise Exception(f'Too many mirrors ({len(mirrors)})')
            elif len(mirrors) == 1:
                return 'Cs', mirrors
            else:
                return 'C1', []

    def _dihedral(self, rotations):
        """Dihedral molecules - main axis plus perpendicular C2 axes"""

        main_axis = rotations[0].axis
        schoenflies = f'D{rotations[0].order}'
        mirror_type, mirrors = self._find_mirrors(main_axis, rots=rotations)
        schoenflies += mirror_type
        return schoenflies, mirrors

    def _cyclic(self, rotations):
        """Cyclic symmetry"""

        main_axis = rotations[0].axis
        order = rotations[0].order
        schoenflies = f'C{order}'
        mirror_type, mirrors = self._find_mirrors(main_axis)
        if mirror_type == '':
            imrot = ImproperRotation(main_axis, 2 * order, tol=self.hardtol)
            if self._is_valid(imrot):
                return f'S{2 * order}', [imrot]
            else:
                return schoenflies, mirrors
        else:
            return schoenflies + mirror_type, mirrors

    def _find_mirrors(self, main_axis, rots=[]):
        """
        Possible types are 'h', 'v', 'd', ''. Horizontal (h) mirrors are
        perpendicular to the axis while vertical (v) or dihedral (d) mirrors
        are parallel. d mirrors bisect two C2 axes.
        """

        # if groups is None:
        #    groups = self._group_atoms_by_symbol_and_norm()
        mirror_type = None
        mirrors = []

        # First test whether the axis itself is the normal to a mirror plane.
        mirror = Mirror(main_axis, tol=self.hardtol)
        if self._is_valid(mirror):
            mirrors.append(mirror)
            mirror_type = 'h'

        cluster = self._get_smallest_cluster_off_line(main_axis)
        subclusters = self._subdivide_group_by_projection(cluster, main_axis)

        vertical_axes = []
        for i1, i2 in combinations(subclusters[0], 2):
            normal = self.pos[i1] - self.pos[i2]
            normal /= np.linalg.norm(normal)
            if np.dot(normal, main_axis) > self.sintol:
                continue
            if self._is_close_to_symmetry_axis(normal, vertical_axes):
                continue
            mirror = Mirror(normal, tol=self.hardtol)
            if self._is_valid(mirror):
                mirrors.append(mirror)
                vertical_axes.append(mirror.axis.copy())

        if mirror_type == 'h':
            return mirror_type, mirrors
        elif len(mirrors) == 0:
            return '', mirrors
        else:
            C2_axes = [rot.axis for rot in rots
                if rot.order == 2]
            if len(C2_axes) < 2:
                return 'v', mirrors
            for ax1, ax2 in combinations(C2_axes, 2):
                bisec = 0.5 * (ax1 + ax2)
                bisec /= np.linalg.norm(bisec)
                if any([np.dot(bisec, m.axis) < self.costol for m in mirrors]):
                    return 'd', mirrors
            return 'v', mirrors

    def _spherical_top(self):
        """High symmetry (T, O, I)"""

        # Get rotations
        # groups = self._group_atoms_by_symbol_and_norm()
        symmetry_axes = []
        max_rot_order = 1
        symm_ops = []

        for cluster in self.clusters:
            neighbor_list = self._get_neighbor_list(cluster)
            rots, partial_max_rot_order = self._get_spherical_rotations(
                neighbor_list,
                symmetry_axes)
            symm_ops += rots
            max_rot_order = max(max_rot_order, partial_max_rot_order)
            if max_rot_order > 3:
                break

        if len(symm_ops) == 0:
            raise Exception('No rotations')

        main_axis = [op.axis for op in symm_ops if op.order == max_rot_order][0]

        if max_rot_order == 2:
            # Accidental symmetric top
            schoenflies, more_symm_ops = self._symmetric_top(0)
            return schoenflies, symm_ops + more_symm_ops

        inv = Inversion()
        if self._is_valid(inv):
            symm_ops.append(inv)
            sub = 'h'
        else:
            sub = ''

        if max_rot_order == 3:
            # Tetrahedral
            if not self._has_rots(symm_ops, nC2=3, nC3=4):
                raise Exception('T: Incorrect number of axes')

            if sub == 'h':
                return 'Th', symm_ops

            mirror_type, mirrors = self._find_mirrors(main_axis, symm_ops)
            symm_ops += mirrors
            if mirror_type == '':
                return 'T', symm_ops
            else:
                return 'Td', symm_ops

        if max_rot_order == 4:
            # Octahedral
            if not self._has_rots(symm_ops, nC2=6, nC3=4, nC4=3):
                raise Exception('O: Incorrect number of axes')
            return 'O' + sub, symm_ops

        elif max_rot_order == 5:
            # Icosahedral
            if not self._has_rots(symm_ops, nC2=15, nC3=10, nC5=6):
                raise Exception('I: Incorrect number of axes')
            return 'I' + sub, symm_ops
        else:
            raise Exception('Error in spherical top')

        return schoenflies, symm_ops

    def _has_rots(self, symm_ops, nC2=0, nC3=0, nC4=0, nC5=0):
        """Check that list of SymmetryOperations has the correct
        number of Cn axes"""

        n_axes = {2: nC2, 3: nC3, 4: nC4, 5: nC5}
        for op in symm_ops:
            if isinstance(op, Rotation):
                n_axes[op.order] -= 1
        return all(val == 0 for val in n_axes.values())

    def _is_close_to_symmetry_axis(self, axis, symmetry_axes):
        """
        Check if axis is non-zero and non-parallel to any of searched_axes

        Inputs:

        axis : (3,) numpy.ndarray
        symmetry_axes : list of (3,) numpy.ndarray
        """

        if np.shape(axis) != (3,):
            raise Exception(f'Incorrect dimensions of axis: {np.shape(axis)}')
        norm = np.linalg.norm(axis)
        if norm < self.hardtol:
            return True
        axis /= norm

        close = any(abs(np.dot(a, axis)) > self.costol for a in symmetry_axes)

        return close

    def _get_spherical_rotations(self, neighbor_list, symmetry_axes):
        """
        Find rotation axes by checking axis
        1) through atoms (might be superfluous)
        2) midpoints between two atoms
        3) normals to regular polygons with three atoms as vertices
        It is assumed that nearest neighbors are enough to check.

        Inputs:

        neighbor_list : list of lists of int
            sublists are [i, j0, j1, ...], where jn are indices of nearest
            neighbors to atom i
        searched_axes : list of numpy.ndarray
            axes already checked for rotations
        """

        rots = []
        for nbr_row in neighbor_list:
            main_ind = nbr_row[0]
            axis = self.pos[main_ind].copy()

            # Axis through atoms. Might not be needed
            # Order must be a factor of number of nearest neighbors
            if not self._is_close_to_symmetry_axis(axis, symmetry_axes):
                num_nbrs = len(nbr_row) - 1
                for order in [5, 4, 3, 2]:
                    if num_nbrs % order == 0:
                        rot = Rotation(axis, order, tol=self.hardtol)
                        if self._is_valid(rot):
                            rots.append(rot)
                            symmetry_axes.append(rot.axis.copy())
                            break

            other_inds = [i for i in nbr_row[1:] if i > main_ind]

            # Midpoint between pairs of atoms
            pairs = [(main_ind, x) for x in other_inds]
            for i1, i2 in pairs:
                axis = 0.5 * (self.pos[i1] + self.pos[i2])
                if self._is_close_to_symmetry_axis(axis, symmetry_axes):
                    continue
                rot = Rotation(axis, order=2, tol=self.hardtol)
                if self._is_valid(rot):
                    rots.append(rot)
                    symmetry_axes.append(rot.axis.copy())

            # Normal to plane of triple of atoms
            triples = [(main_ind, x, y) for x, y in
                combinations(other_inds, 2)]

            for triple in triples:
                pos = self.pos[list(triple)]
                vec1 = pos[1] - pos[0]
                vec2 = pos[2] - pos[0]
                axis = np.cross(vec1, vec2)

                if self._is_close_to_symmetry_axis(axis, symmetry_axes):
                    continue

                dot = np.dot(vec1, vec2)
                cos_angle = dot / np.linalg.norm(vec1) / np.linalg.norm(vec2)
                n_vertices = 2 * np.pi / np.arccos(-cos_angle)
                if abs(n_vertices - round(n_vertices)) > self.disttol:
                    continue
                n_vertices = int(round(n_vertices))
                for order in [5, 4, 3]:
                    if n_vertices % order == 0:
                        rot = Rotation(axis, order, tol=self.hardtol)
                        if self._is_valid(rot):
                            rots.append(rot)
                            symmetry_axes.append(rot.axis.copy())
                            break

        max_rot_order = max([rot.order for rot in rots], default=1)
        return rots, max_rot_order

    def _get_neighbor_list(self, atom_indices, n=10, nearest=True):
        """
        Get neighbor lists for atoms with indices

        Inputs:

        atom_indices : list of int
            indices of atoms for which neighbor lists are built (typically
            a cluster in self.clusters)
        n : int
            number of neighbors to include (before any distance check)
        nearest : bool
            if True, include only nearest neighbors, with the smallest distance
        """

        neighbor_list = []
        group_positions = self.pos[atom_indices]
        tree = sp.spatial.KDTree(group_positions)
        dists, neighbors = tree.query(group_positions, k=n + 1)
        for dist_row, nbr_row in zip(dists, neighbors):
            min_dist = dist_row[1] if len(dist_row) > 1 else np.inf
            valid_mask = dist_row <= min_dist + self.disttol
            valid_nbrs = nbr_row[valid_mask]

            nbrs = [atom_indices[nbr] for nbr in valid_nbrs]
            neighbor_list.append(nbrs)

        return sorted(neighbor_list)

    def _on_line(self, vector):
        """
        Returns mask for on the line vector through origin

        Inputs:

        vector : numpy.ndarray
            defines the line points are on or off
        """

        if np.linalg.norm(vector) < self.hardtol:
            raise ValueError("Direction vector cannot be zero.")
        vector /= np.linalg.norm(vector)

        pos_norms = np.linalg.norm(self.pos, axis=1)
        dots = self.pos @ vector

        cos_angles = np.ones_like(dots)  # default to 1.0 for atom at origin
        nonzero = pos_norms > self.disttol

        # |pos| * |vector| * cos(angle) = dot(pos, vector); |vector| = 1
        cos_angles[nonzero] = dots[nonzero] / pos_norms[nonzero]

        # if on the `vector` line, `cos_angles` should be close to +1 or -1
        mask = (np.abs(cos_angles) > self.costol)

        return mask

    def _group_atoms_by_symbol_and_norm(self, inds=None, d=4):
        """
        Inputs:

        inds : list of int
            indices of atoms to group
        d : int
            number of digits for distance in dictionary key. Mainly for
            debugging

        Returns:

        groups : list of numpy.ndarray
            the arrays are indices of atoms of same element and same distance
            to the origin
        """

        if inds is None:
            inds = np.arange(len(self.pos))
        else:
            inds = np.array(inds)

        positions = np.array(self.pos)[inds]
        symbols = np.array(self.symbols)[inds]
        dists = np.linalg.norm(positions, axis=1)

        groups = {}

        for symbol in np.unique(symbols):
            # Filter and skip central atom
            mask = (symbols == symbol) & (dists > self.disttol)
            sym_inds = inds[mask]
            sym_dists = dists[mask]

            if len(sym_dists) == 0:
                continue
            if len(sym_dists) == 1:
                groups[(symbol, round(sym_dists[0], d))] = np.array(sym_inds)
                continue

            # Cluster the 1D distance data
            labels = sp.cluster.hierarchy.fclusterdata(
                sym_dists[:, None],
                t=self.disttol,
                criterion='distance'
            )

            for label in np.unique(labels):
                cluster_mask = labels == label
                cluster_inds = sym_inds[cluster_mask]
                mean_dist = float(np.mean(sym_dists[cluster_mask]))
                key = (str(symbol), round(mean_dist, d))
                groups[key] = cluster_inds

        group_vals = sorted(list(groups.values()), key=lambda x: len(x))
        return group_vals

    def _subdivide_group_by_projection(self, inds, proj_axis, pairs=False):
        """
        Project the position vectors on an axis, i.e. group them in planes
        perpendicular to the axis

        Inputs:

        inds : np.ndarray of int | list of int
            atom indices. Must be of same element
        proj_axis : np.ndarray
            vector to project on
        pairs : bool
            pair up the output groups of equal distance to the central plane
            through the origin, perpendicular to proj_axis.

        Returns:

        If pairs is False:
            groups : list of numpy.ndarray of int
        Else
            pairs : list of list (len 2) of numpy.ndarray of int
            central_group : numpy.ndarray of int

        """
        if len(inds) <= 1:
            # raise Exception('Group too small')
            return [inds]
        positions = np.array(self.pos)[inds]
        symbols = np.array(self.symbols)[inds]
        if len(np.unique(symbols)) != 1:
            raise Exception('Only one element for subdivision projection')
        dists = np.dot(positions, proj_axis) / np.linalg.norm(proj_axis)

        labels = sp.cluster.hierarchy.fclusterdata(
            dists[:, None],
            t=self.disttol,
            criterion='distance'
        )

        groups = {}
        for label in np.unique(labels):
            cluster_mask = labels == label
            cluster_inds = inds[cluster_mask]
            mean_dist = float(np.mean(dists[cluster_mask]))
            key = round(mean_dist, 4)
            groups[key] = cluster_inds

        if not pairs:
            group_vals = sorted(list(groups.values()), key=lambda x: len(x))
            return group_vals

        keys = sorted(groups.keys(), key=abs)
        used_keys = set()
        pairs = []

        for i, k1 in enumerate(keys):
            for k2 in keys[i + 1:]:
                if k2 in used_keys:
                    continue
                if abs(k1 + k2) < self.disttol:
                    pairs.append([groups[k1], groups[k2]])
                    used_keys.update({k1, k2})
                if abs(k2) - abs(k1) > self.disttol:
                    break
        if keys[0] not in used_keys and abs(keys[0]) < self.disttol:
            # pairs.append([groups[keys[0]], groups[keys[0]]])
            central = groups[keys[0]]
        else:
            central = None

        pairs.sort(key=lambda pair: len(pair[0]))
        return pairs, central

    def _mass_check(self):
        """Checks that atoms of elements have the same mass, else exception"""

        unique_symbols = np.unique(self.symbols)
        masses = self.atoms.get_masses()
        for symbol in unique_symbols:
            mask = [symbol == s for s in self.symbols]
            masses_for_symbol = masses[mask]

            ref_mass = masses_for_symbol[0]
            if not np.allclose(masses_for_symbol, ref_mass,
                               atol=self.hardtol):
                print(f'Masses not equal for {symbol}')
                print('Current implementation requires equal mass')
                raise Exception('Unique masses required')
        return

    def _center_on_center_of_mass(self, atoms):
        """Center atoms on center of mass

        Inputs:

        atoms : an ASE atoms object
            the atoms to be centered
        """

        pos_start = atoms.get_positions()
        com = atoms.get_center_of_mass()
        atoms.set_positions(pos_start - com, apply_constraint=False)
        return atoms
