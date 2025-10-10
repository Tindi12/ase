import ase.calculators.emt
from ase._4.calculators.calculator import Version4Adaptor


class EMT(Version4Adaptor):
    wrapped_class = ase.calculators.emt.EMT
