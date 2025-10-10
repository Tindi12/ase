from copy import deepcopy
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from ase.outputs import Properties, all_outputs


class CalculationResults:
    """
    Container for the results of all Calculator.evaluate() methods.

    Currently only supports properties defined in `ase.outputs.all_outputs`
    and assumes that Atoms.store(CalculationResults) will write the
    properties to Atoms.info and Atoms.arrays.

    Storred data is immutable - setter functions don't allow overwriting
    non-empty attributes and getter functions return a read-only proxy.

    Attributes
    ----------
    metadata
    properties

    """

    _metadata: dict[str, Any]
    _properties: Properties
    recognised_properties = all_outputs

    def __init__(
        self, metadata: dict = None, properties: dict | Properties | None = None
    ) -> None:
        """
        Parameters
        ----------
        metadata : , optional
            Arbitrary information about the calculation.
        properties : ase.outputs.Properties or dict, optional
            Calculated properties, compliant with `ase.outputs.all_outputs`.
        """
        self._metadata = {}
        if metadata is not None:
            self.metadata = metadata

        if properties is None:
            properties = Properties({})
        elif isinstance(properties, dict):
            properties = Properties(properties)
        self._properties = properties

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Read-only Mapping[str, Any] to store calculation metadata."""
        return MappingProxyType(self._metadata)

    @metadata.setter
    def metadata(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise TypeError(f'`metadata` has to be a dict, not {type(data)}.')
        if len(self.metadata) > 0:
            raise AttributeError(
                f'Not overwriting already set metadata: {self.metadata}.'
            )
        self._metadata = deepcopy(data)

    @property
    def properties(self) -> Properties:
        """ase.outputs.Properties to store calculation results."""
        return self._properties

    @properties.setter
    def properties(self, data: dict | Properties) -> None:
        if not isinstance(data, dict) and not isinstance(data, Properties):
            raise TypeError(
                '`properties` has to be an instance of '
                f'`ase.outputs.Properties` or `dict`, not {type(data)}.'
            )
        if len(self.properties) > 0:
            raise AttributeError(
                f'Not overwriting already set properties: {self.properties}.'
            )

        if isinstance(data, dict):
            data = Properties(data)
        self._properties = data

    def add_metadata(self, key: str, val: Any) -> None:
        if key in self._metadata:
            raise AttributeError(
                f'{key} is already present in the metadata '
                f'with {key}={self.metadata[key]}.'
            )
        self._metadata[key] = val

    def add_property(self, key: str, val: int | float | np.ndarray) -> None:
        if key not in self.recognised_properties:
            raise KeyError(
                f'{key} is not one of the recognised '
                f'properties {self.recognised_properties.keys()}.'
            )

        self._properties._setvalue(key, val)

    def __repr__(self) -> str:
        clsname = type(self).__name__
        info = f'{clsname}\n'

        # metadata
        info += '    Metadata:\n'
        for key, val in self.metadata.items():
            info += f'        {key}={val}\n'

        # properties
        info += '    Properties:\n'
        for key, val in self.properties.items():
            info += f'        {key}={val}\n'

        return info
