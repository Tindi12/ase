import copy
import os
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np

from ase import Atoms as V3Atoms
from ase._4.calculators.results import CalculationResults
from ase.calculators.calculator import BaseCalculator as V3BaseCalculator
from ase.outputs import all_outputs

all_properties = [
    'energy',
    'forces',
    'stress',
    'stresses',
    'dipole',
    'charges',
    'magmom',
    'magmoms',
    'free_energy',
    'energies',
    'dielectric_tensor',
    'born_effective_charges',
    'polarization',
]


special = {
    'emt': 'EMT',
}


def equal(a, b, tol=None, rtol=None, atol=None):
    """ndarray-enabled comparison function."""
    # XXX Known bugs:
    #  * Comparing cell objects (pbc not part of array representation)
    #  * Infinite recursion for cyclic dicts
    #  * Can of worms is open
    if tol is not None:
        msg = 'Use `equal(a, b, rtol=..., atol=...)` instead of `tol=...`'
        warnings.warn(msg, DeprecationWarning)
        assert rtol is None and atol is None, (
            'Do not use deprecated `tol` with `atol` and/or `rtol`'
        )
        rtol = tol
        atol = tol

    a_is_dict = isinstance(a, dict)
    b_is_dict = isinstance(b, dict)
    if a_is_dict or b_is_dict:
        # Check that both a and b are dicts
        if not (a_is_dict and b_is_dict):
            return False
        if a.keys() != b.keys():
            return False
        return all(equal(a[key], b[key], rtol=rtol, atol=atol) for key in a)

    if np.shape(a) != np.shape(b):
        return False

    if rtol is None and atol is None:
        return np.array_equal(a, b)

    if rtol is None:
        rtol = 0
    if atol is None:
        atol = 0

    return np.allclose(a, b, rtol=rtol, atol=atol)


class Parameters(dict):
    """Dictionary for parameters.

    Special feature: If param is a Parameters instance, then param.xc
    is a shorthand for param['xc'].
    """

    def __getattr__(self, key):
        if key not in self:
            return dict.__getattribute__(self, key)
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    @classmethod
    def read(cls, filename):
        """Read parameters from file."""
        # We use ast to evaluate literals, avoiding eval()
        # for security reasons.
        import ast

        with open(filename) as fd:
            txt = fd.read().strip()
        assert txt.startswith('dict(')
        assert txt.endswith(')')
        txt = txt[5:-1]

        # The tostring() representation "dict(...)" is not actually
        # a literal, so we manually parse that along with the other
        # formatting that we did manually:
        dct = {}
        for line in txt.splitlines():
            key, val = line.split('=', 1)
            key = key.strip()
            val = val.strip()
            if val[-1] == ',':
                val = val[:-1]
            dct[key] = ast.literal_eval(val)

        parameters = cls(dct)
        return parameters

    def tostring(self):
        keys = sorted(self)
        return (
            'dict('
            + ',\n     '.join(f'{key}={self[key]!r}' for key in keys)
            + ')\n'
        )

    def write(self, filename):
        Path(filename).write_text(self.tostring())


class BaseCalculator(ABC):
    implemented_properties: List[str] = []
    'Properties calculator can handle (energy, forces, ...)'

    # Placeholder object for deprecated arguments.  Let deprecated keywords
    # default to _deprecated and then issue a warning if the user passed
    # any other object (such as None).
    _deprecated = object()

    def __init__(self, parameters=None):
        if parameters is None:
            parameters = {}

        self.parameters = dict(parameters)
        self.atoms = None

    @abstractmethod
    def evaluate(self, atoms, properties): ...

    def _get_name(self) -> str:  # child class can override this
        return self.__class__.__name__.lower()

    @property
    def name(self) -> str:
        return self._get_name()

    def todict(self) -> dict[str, Any]:
        """Obtain a dictionary of parameter information"""
        return {}


class Calculator(BaseCalculator):
    """Base-class for all ASE calculators.

    A calculator must raise PropertyNotImplementedError if asked for a
    property that it can't calculate.  So, if calculation of the
    stress tensor has not been implemented, get_stress(atoms) should
    raise PropertyNotImplementedError.  This can be achieved simply by not
    including the string 'stress' in the list implemented_properties
    which is a class member.  These are the names of the standard
    properties: 'energy', 'forces', 'stress', 'dipole', 'charges',
    'magmom' and 'magmoms'.
    """

    default_parameters: Dict[str, Any] = {}
    'Default parameters'

    def __init__(
        self,
        restart=None,
        label=None,
        directory='.',
        **kwargs,
    ):
        """Basic calculator implementation.

        restart: str
            Prefix for restart file.  May contain a directory. Default
            is None: don't restart.
        directory: str or PurePath
            Working directory in which to read and write files and
            perform calculations.
        label: str
            Name used for all files.  Not supported by all calculators.
            May contain a directory, but please use the directory parameter
            for that instead.
        """
        self.parameters = None  # calculational parameters
        self._directory = None  # Initialize

        if restart is not None:
            # duplicated in transition implementation
            self.read(restart)  # read parameters, atoms and results

        self.directory = directory
        self.prefix = None
        if label is not None:
            if self.directory == '.' and '/' in label:
                # We specified directory in label, and nothing in the directory
                # key
                self.label = label
            elif '/' not in label:
                # We specified our directory in the directory keyword
                # or not at all
                self.label = '/'.join((self.directory, label))
            else:
                raise ValueError(
                    'Directory redundantly specified though '
                    'directory="{}" and label="{}".  '
                    'Please omit "/" in label.'.format(self.directory, label)
                )

        if self.parameters is None:
            # Use default parameters if they were not read from file:
            self.parameters = self.get_default_parameters()

        self.set_check_parameter_changes(**kwargs)

        if not hasattr(self, 'get_spin_polarized'):
            self.get_spin_polarized = self._deprecated_get_spin_polarized
        # XXX We are very naughty and do not call super constructor!

    @property
    def directory(self) -> str:
        return self._directory

    @directory.setter
    def directory(self, directory: Union[str, os.PathLike]):
        self._directory = str(Path(directory))  # Normalize path.

    @property
    def label(self):
        if self.directory == '.':
            return self.prefix

        # Generally, label ~ directory/prefix
        #
        # We use '/' rather than os.pathsep because
        #   1) directory/prefix does not represent any actual path
        #   2) We want the same string to work the same on all platforms
        if self.prefix is None:
            return self.directory + '/'

        return f'{self.directory}/{self.prefix}'

    @label.setter
    def label(self, label):
        if label is None:
            self.directory = '.'
            self.prefix = None
            return

        tokens = label.rsplit('/', 1)
        if len(tokens) == 2:
            directory, prefix = tokens
        else:
            assert len(tokens) == 1
            directory = '.'
            prefix = tokens[0]
        if prefix == '':
            prefix = None
        self.directory = directory
        self.prefix = prefix

    def set_label(self, label):
        """Set label and convert label to directory and prefix.

        Examples:

        * label='abc': (directory='.', prefix='abc')
        * label='dir1/abc': (directory='dir1', prefix='abc')
        * label=None: (directory='.', prefix=None)
        """
        self.label = label

    def get_default_parameters(self):
        return Parameters(copy.deepcopy(self.default_parameters))

    def todict(self, skip_default=True):
        defaults = self.get_default_parameters()
        dct = {}
        for key, value in self.parameters.items():
            if hasattr(value, 'todict'):
                value = value.todict()
            if skip_default:
                default = defaults.get(key, '_no_default_')
                if default != '_no_default_' and equal(value, default):
                    continue
            dct[key] = value
        return dct

    def read(self, label):
        """Read atoms, parameters and calculated properties from output file.

        Read result from self.label file.  Raise ReadError if the file
        is not there.  If the file is corrupted or contains an error
        message from the calculation, a ReadError should also be
        raised.  In case of success, these attributes must set:

        atoms: Atoms object
            The state of the atoms from last calculation.
        parameters: Parameters object
            The parameter dictionary.
        results: dict
            Calculated properties like energy and forces.

        The FileIOCalculator.read() method will typically read atoms
        and parameters and get the results dict by calling the
        read_results() method."""

        self.set_label(label)

    # EG: not yet sure how to handle this.
    # The ase4 calculator needs a parameter setter function
    # if we only want to split out Atoms, but preserve the rest
    # of the behavirour unchanged. But ase3 calculator needs
    # to check whether the parameters have changed before updating
    # so that the calculator may be reset. How should that be
    # handled in the ASEv4 + ASEv3Mixin split during the transition?
    def set_check_parameter_changes(self, **kwargs):
        """Set parameters like set(key1=value1, key2=value2, ...).

        A dictionary containing the parameters that have been changed
        is returned.

        The special keyword 'parameters' can be used to read
        parameters from a file."""

        if 'parameters' in kwargs:
            filename = kwargs.pop('parameters')
            parameters = Parameters.read(filename)
            parameters.update(kwargs)
            kwargs = parameters

        changed_parameters = {}

        for key, value in kwargs.items():
            oldvalue = self.parameters.get(key)
            if key not in self.parameters or not equal(value, oldvalue):
                changed_parameters[key] = value
                # set only here in v4 base class
                self.parameters[key] = value
        # also returned by the transition class
        return changed_parameters

    def evaluate(self, atoms, properties=['energy']):
        """Use the calculator to evaluate the structure and obtain properties.

        atoms: Atoms
            Structure to be evaluated.
        properties: list of str
            List of what needs to be calculated.  Can be any combination
            of 'energy', 'forces', 'stress', 'dipole', 'charges', 'magmom'
            and 'magmoms'.

        Subclasses need to implement this, but can ignore properties
        if they want. Calculated properties should
        be returned as a CalculationResults object.

        The subclass implementation should first call this
        implementation to set the atoms attribute and create any missing
        directories.
        """
        if not os.path.isdir(self._directory):
            try:
                os.makedirs(self._directory)
            except FileExistsError as e:
                # We can only end up here in case of a race condition if
                # multiple Calculators are running concurrently *and* use the
                # same _directory, which cannot be expected to work anyway.
                msg = (
                    'Concurrent use of directory '
                    + self._directory
                    + 'by multiple Calculator instances detected. Please '
                    'use one directory per instance.'
                )
                raise RuntimeError(msg) from e

    def _deprecated_get_spin_polarized(self):
        msg = (
            'This calculator does not implement get_spin_polarized().  '
            'In the future, calc.get_spin_polarized() will work only on '
            'calculator classes that explicitly implement this method or '
            'inherit the method via specialized subclasses.'
        )
        warnings.warn(msg, FutureWarning)
        return False

    def band_structure(self):
        """Create band-structure object for plotting."""
        from ase.spectrum.band_structure import get_band_structure

        # XXX This calculator is supposed to just have done a band structure
        # calculation, but the calculator may not have the correct Fermi level
        # if it updated the Fermi level after changing k-points.
        # This will be a problem with some calculators (currently GPAW), and
        # the user would have to override this by providing the Fermi level
        # from the selfconsistent calculation.
        return get_band_structure(calc=self)


class Version4Adaptor(BaseCalculator):
    """A generic wrapper to make ASE 3.x/old/legacy calculators work
    with ASE 4.x interface.
    """

    wrapped_class: type[V3BaseCalculator]

    def __init__(self, *args, **kwargs):
        self._v3_calculator = self.wrapped_class(*args, **kwargs)

    def evaluate(
        self, atoms: V3Atoms, properties: list[str] = ['energy']
    ) -> CalculationResults:
        # enforce no modification of the input atoms
        atoms = atoms.copy()
        self._v3_calculator.calculate(atoms=atoms, properties=properties)
        results = CalculationResults()

        for prop, val in self._v3_calculator.results.items():
            if prop in all_outputs:
                results.add_property(prop, val)
            if prop not in all_outputs:
                warnings.warn(
                    f'Property {prop} was found in calculation results '
                    f'but is not one of the standard properties of '
                    f'ase.outputs.all_outputs, skipping.'
                )

        return results
