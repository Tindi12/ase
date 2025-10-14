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

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @classmethod
    def from_v3atoms(cls, v3atoms: v3Atoms):
        return Atoms(symbols=v3atoms.symbols,
                     positions=v3atoms.positions,
                     cell=v3atoms.cell,
                     pbc=v3atoms.pbc)

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

    def get_potential_energy(self, force_consistent=False,
                             apply_constraint=True):
        raise NotImplementedError("moved to Calculator.evalute() in ASEv4")

    def get_potential_energies(self, force_consistent=False,
                             apply_constraint=True):
        raise NotImplementedError("moved to Calculator.evalute() in ASEv4")

    def get_total_energy(self):
        raise NotImplementedError("moved to Calculator.evalute() in ASEv4")

    def get_forces(self, apply_constraint=True, md=False):
        raise NotImplementedError("moved to Calculator.evalute() in ASEv4")

    def get_stress(self, voigt=True, apply_constraint=True,
                   include_ideal_gas=False):
        raise NotImplementedError("moved to Calculator.evalute() in ASEv4")

    def get_stresses(self, include_ideal_gas=False, voigt=True):
        raise NotImplementedError("moved to Calculator.evalute() in ASEv4")

    @property
    def calc(self):
        raise NotImplementedError("Calculator no longer lives in Atoms")

    @calc.setter
    def calc(self, calc):
        raise NotImplementedError("Calculator no longer lives in Atoms")

    @calc.deleter
    def calc(self):
        raise NotImplementedError("Calculator no longer lives in Atoms")
    
