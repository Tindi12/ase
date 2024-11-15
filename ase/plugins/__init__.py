"""Atomic Simulation Environment plugin package. In this package is all the
stuff related to the ASE plugins - mechanism how to enrich ASE package with
new formats, calculators and/or viewers.

To create a plugin, please, add ``ase.plugins`` entry point (e.g. to
pyproject.toml file) to your package. The entry point should have the name of
the plugin, and the value the module, that contains a ``ase_register``
function, that accepts plugin as argument.
This function should call ``register_io_format``,
``register_calculator`` and/or ``register_viewer``
plugin method to register the new calculators/viewers/... E.g.

```toml
[project.entry-points."ase.plugins"]
my_plugin_name = "my_package.module_with_ase_register"
```

``my_package.module_with_ase_register`` than could contain e.g.:
```
def ase_register():
    from ase.plugins import register_calculator, register_io_format
    register_calculator('ase2sprkkr.SPRKKR')
    register_io_format('ase2sprkkr.ase.io', 'SPRKKR potential file',
                       '1F', name='sprkkr', ext='pot')
```
"""

__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from ase.io import formats as _formats
from ase.visualize import viewers as _viewers

from .pluggables import CalculatorPluggables
from .plugin import Plugins
from .registering import register_plugins  # NOQA

plugins = Plugins({
    'calculators': CalculatorPluggables,
    'io_formats': _formats.IOFormatPluggables,
    'viewers': _viewers.ViewerPluggables
})

calculators: CalculatorPluggables = plugins.calculators
io_formats: _formats.IOFormatPluggables = plugins.io_formats
viewers: _viewers.ViewerPluggables = plugins.viewers

# set up the legacy ways how to get the pluggables
# it has to be here to avoid circular import
_viewers.VIEWERS = viewers
_viewers.CLI_VIEWERS = viewers.cli_viewers
_viewers.PY_VIEWERS = viewers.python_viewers
_formats.ioformats = io_formats
# Aliased for compatibility only. Please do not use.
_formats.all_formats = io_formats
_formats.extension2format = io_formats.by_extension


# install the plugins
register_plugins()


__all__ = [
    'plugins',
    'calculators',
    'io_formats',
    'viewers',
]
