"""
========
Geometry
========
"""

from ase.atoms import BaseAtoms
from ase.outputs import Properties


class Geometry(BaseAtoms):
    """Geometry object of ASE4 to store properties of multiple calculators.

    Attributes
    ----------
    results : dict[str, :class:`~ase.outputs.Properties`]
        Dictionary to store the properties of calculators.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.results: dict[str, Properties] = {}

    def store(self, properties: Properties, *, label: str = '') -> None:
        """Store the properties obtained with a calculator.

        Parameters
        ----------
        properties : :class:`~ase.outputs.Properties`
            :class:`~ase.outputs.Properties` obtained by a calculator.
        label : str, default = ''
            Label to the properties.

        Examples
        --------
        >>> from ase.build import bulk
        >>> from ase.calculators.emt import EMT
        >>> from ase.dev.geometry import Geometry
        >>> geom = Geometry(bulk('Cu'))
        >>> calc = EMT()
        >>> properties = ['energy']
        >>> geom.store(calc.calculate_properties(geom, properties), label='EMT')
        >>> print(f"{geom.results['EMT']['energy']:f}")
        -0.005682
        """
        self.results[label] = properties
