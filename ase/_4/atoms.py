from ase._4.calculators.results import CalculationResults
from ase.atoms import Atoms as V3Atoms, _LimitedAtoms
from ase.outputs import ArrayProperty, all_outputs
from ase.test._4.optimize import OptimizableAtomsv4

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

    def __ase_optimizable__(self):
        from ase._4.optimize import OptimizableV4Atoms
        return OptimizableV4Atoms(self)
