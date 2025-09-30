from ase._4.calculators.results import CalculationResults
from ase.atoms import Atoms as v3Atoms
from ase.outputs import ArrayProperty, all_outputs


class Atoms(v3Atoms):
    """
    Dummy class to illustrate how `Atoms.store` works with
    `CalculationResults`. Will be adjusted accordingly when
    merging with the branch that implements the updated
    Atoms interface.
    """

    def __init__(self, kwargs) -> None:
        super.__init__(**kwargs)

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
