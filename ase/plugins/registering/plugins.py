"""This meta-plugin takes care of registering all external plugins.
To create a plugin, do a two thing. First, create a registeration module,
(e.g. ``my_cool_package/ase/register.py"), with ``ase_register`` funtion within.
```
from ase.plugins import register_calculator, register_io_format

ase_register():
    register_calculator("mypackage.calculator.MyCalculator")
    register_io_format("mypackage.io", "My IO format",
                       name="my_cool_format", ext="cool")
    ...
```
The function will be served as entry point, which have to
be defined in ``pyproject.toml``:
```

[project.entry-points."ase.plugins"]
my_cool_package="my_cool_package.ase.register"
```
"""

import importlib
import sys
import warnings
from importlib.metadata import entry_points

from .. import plugins


def ase_register_ex():
    def import_plugin_module(name, value):
        try:
            module = importlib.import_module(value)
            return module
        except Exception as e:
            spec = importlib.util.find_spec(value)
            inn = f" located in '{spec.origin}'" if spec else ''
            warnings.warn(
                f'Can not import module {value}{inn} from entry '
                f' point ase.plugins.{name} or call its '
                f' ase_register() function. '
                f'This ASE plugin is probably broken. \nReason: {e}'
            )

    if sys.version_info < (3, 10):
        epoints = entry_points()
        if 'ase.plugins' not in epoints:
            return
        epoints = epoints['ase.plugins']
    else:
        epoints = entry_points(group='ase.plugins')
    for epoint in epoints:
        module = import_plugin_module(epoint.name, epoint.value)
        if module:
            plugins.create_plugin(module, epoint.name).register()
