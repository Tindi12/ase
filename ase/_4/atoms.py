from functools import cached_property
from ase._4.calculators.calculator import BaseCalculator
from ase._4.calculators.results import CalculationResults
from ase.atoms import Atoms as V3Atoms, _LimitedAtoms
from ase.outputs import ArrayProperty, all_outputs
from ase.utils.abc import Optimizable

# consider renaming this to V4Atoms during transition period
# 
class Atoms(_LimitedAtoms):
    """
    Dummy class to illustrate how `Atoms.store` works with
    `CalculationResults`. Will be adjusted accordingly when
    merging with the branch that implements the updated
    Atoms interface.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @classmethod
    def from_v3atoms(cls, v3atoms: V3Atoms):
        return Atoms(
            symbols=v3atoms.symbols,
            positions=v3atoms.positions,
            cell=v3atoms.cell,
            pbc=v3atoms.pbc,
        )

    def store(self, results: CalculationResults, label: str = '') -> None:
        """
        Stores properties from CalculationResults in Atoms.info and
        Atoms.arrays under "labelproperty".
        """
        properties = results.properties
        if len(properties) == 0:
            raise ValueError(
                'The CalculationResults instance has no properties to save'
            )

        # All properties in CalculationResults are compatible with
        # `ase.outputs.all_outputs.
        for prop_name, prop_val in properties.items():
            output_type = all_outputs[prop_name]
            if (
                isinstance(output_type, ArrayProperty)
                and output_type.shapespec[0] == 'natoms'
            ):
                self.arrays[label + prop_name] = prop_val
            else:
                self.info[label + prop_name] = prop_val


class PotentialEnergySurface(Optimizable):
    def __init__(self, atoms: Atoms, calc: BaseCalculator):
        self.atoms = atoms
        self.calc = calc

    def get_x(self):
        return self.atoms.get_positions().ravel()

    def set_x(self, x):
        self.atoms.set_positions(x.reshape(-1, 3))

    def get_gradient(self):
        results = self.calc.evaluate(self.atoms, properties="forces")
        return results.properties["forces"].ravel()

    @cached_property
    def _value_property(self):
        # This boolean is in principle invalidated if the
        # calculator changes.  This can lead to weird things
        # in multi-step optimizations.
        if 'free_energy' in self.calc.implemented_properties:
            return 'free_energy'
        else:
            return 'energy'

    def get_value(self):
        results = self.calc.evaluate(self.atoms,
                                     properties=self._value_property)
        return results.properties[self._value_property]

    def iterimages(self):
        # XXX document purpose of iterimages
        return self.atoms.iterimages()

    def ndofs(self):
        return 3 * len(self.atoms)
