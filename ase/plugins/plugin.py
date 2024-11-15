""" This module contains the classes for listing plugins and
the pluggables (Calculators, IOs, etc...) provided by the plugins.


The structure is as follows

```
-----------                     --------------------------------
|         |                     |                              |
| Plugins |--------1:n----------|   Plugin (can provide some   |
|         |  (for each plugin   |   calculator, formats...)    |
|         |      package)       |                              |
^^^^^^^^^^                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |                                           |
   1:n  for each type                         1:n - plugin can return lists
    |   (i.e viewer, calculator)                | of pluggables
    |                                           | of each pluggable type
    |                                           | (calculator, viewer)
    |                                           | that it provides
    |                                           |
    |                                           |
    |                                           |
-------------------------                ------------------------
|                       |                |                      |
|    Pluggables:        |                |    Pluggable:        |
|      list all the     |------1:n-------|    holds inform.     |
|   available 'items'   |                |    about just one    |
|   (calcs, viewers)    |                |    calculator        |
|   of the given type   |                |    or viewer         |
|                       |                |                      |
^^^^^^^^^^^^^^^^^^^^^^^^^                ^^^^^^^^^^^^^^^^^^^^^^^^
                                                  |
                                                 1:1
                                                  |
                                         ------------------------
                                         |                      |
                                         |   Implementation     |
                                         |   (e.g. a Calculator |
                                         |    subclass)         |
                                         |                      |
                                         """"""""""""""""""""""""
```

To register a plugin, see a docstring of
:module:`ase.plugins.builtin.plugins`
"""

import warnings
from typing import List, Optional, Tuple, Union

import numpy as np

from ase.utils import lazyproperty

from .listing import Listing


class Plugins(Listing):
    """ A class, that holds all the installed plugins in
        the given namespace package.
        Plugins are registered by importing register subpackage.
    """

    def __init__(self, pluggable_types):
        self._pluggables = {
            k: cls(k)
            for k, cls in pluggable_types.items()
        }
        self._items = {}

    def register(self, plugin):
        self._items[plugin.name] = plugin

    def pluggables_of(self, class_type):
        return self._pluggables[class_type]

    def all_pluggables(self):
        return self._pluggables.values()

    @lazyproperty
    def calculators(self):
        return self.pluggables_of('calculators')

    @lazyproperty
    def viewers(self):
        return self.pluggables_of('viewers')

    @lazyproperty
    def io_formats(self):
        return self.pluggables_of('io_formats')

    def __repr__(self):
        return "<ASE plugins>"

    def info(self, prefix='', opts={}, filter=None):
        return "Plugins:\n"\
               "--------\n" + super().info(prefix, opts, filter)

    def create_plugin(self, module, name=None):
        """ A factory method to create a plugin. """
        return Plugin(self, module, name)


class Plugin:
    """ A class, that encapsulates a plugin package """

    def __init__(self, plugins, package, name=None):
        self.plugins = plugins
        self.package = package
        self.pluggables = {
            i.class_type: {} for i in plugins.all_pluggables()
        }
        self.modules = {}
        self._pluggables = {}
        self.registered = False
        if name is None:
            name = self.package.__name__[12:]   # get rig 'ase.plugins'
        self.name = name
        self.broken = False

    def add_pluggable(self, pluggable):
        """ Called by Pluggable.register() """
        self.pluggables[pluggable.class_type][pluggable.name] = pluggable
        self.plugins.pluggables_of(pluggable.class_type).add(pluggable)

    @property
    def lowercase_names(self):
        return (self.name.lower(), )

    def register(self):
        """ Register the pluggables in the plugin:
        Call the ase_register function from the
        __init__.py of the plugin package
        """
        if not self.registered:
            try:
                if hasattr(self.package, 'ase_register'):
                    self.package.ase_register(self)
            except Exception as e:
                self.broken = True
                warnings.warn(f"Can not register plugin {self} because of {e}")
            self.registered = True
            self.plugins.register(self)

    def info(self, prefix='', opts={}):
        info = f'{prefix}{self.name}'

        prefix += '  '
        opts = opts.copy()
        opts['plugin'] = False

        if self.broken:
            info += ' (broken)'

        for pluggables in self.plugins.all_pluggables():
            itype = pluggables.class_type
            inst = self.pluggables[itype]
            if inst:
                p = f'\n{prefix}{pluggables.singular_name}: '
                for i in inst.values():
                    info += i.info(p, opts)
        return info

    def __repr__(self):
        return f"<ASE plugin: {self.package}>"

    def _register_pluggable(self, pluggable_type: str, cls: str, name=None):
        """ Register a calculator or other pluggable exposed by a plugin.
        The name can be derived from the cls name

        Parameters
        ----------
        pluggable_type
          Which to register. E.g. 'calculators', 'io_formats' and so on.
          However, io_formats have its old routine to register, so they
          do not use this mechanism.

        cls: str
          Which class implements the pluggable (e.g.
          ``ase.plugins.emt.EMTCalculator``)
          The class goes by its name only to avoid importing too myuch stuff.
        """
        if not name:
            name = cls.rsplit(".", 1)[-1]
        p_cls = self.plugins.pluggables_of(pluggable_type).item_type
        pluggable = p_cls(pluggable_type, name, cls)
        pluggable.register(self)

    def register_calculator(self, cls: str, name=None):
        """ Register a calculator exposed by a plugin.
        The name can be derived from the cls name
        """
        self._register_pluggable('calculators', cls, name)

    def register_io_format(self, module, desc, code, *, name=None, ext=None,
                           glob=None, magic=None, encoding=None,
                           magic_regex=None, external=True,
                           allowed_pbc: Optional[List[
                               Union[str, bytes, np.ndarray, List, Tuple]
                           ]] = None):
        """ Use this method to define and register an input/output format,
        instead of old :func:`ioformats.define_io_format`.
        The order of parameters is however slightly different here,
        to be as much as possible similiar to the :func:`register_calculator`

        If not external is set, define_io_format add ase.io to the module.
        """
        if not name:
            name = module.rsplit(".", 1)[-1]
        fmt = formats.define_io_format(name, desc, code,
                                       module=module,
                                       ext=ext,
                                       glob=glob,
                                       magic=magic,
                                       encoding=encoding,
                                       magic_regex=magic_regex,
                                       external=external,
                                       allowed_pbc=allowed_pbc
                                       )
        fmt.register(self)

    def register_viewer(self, name, desc, *, module=None, cli=False, fmt=None,
                        argv=None, external=True):
        """ Register a new viewer """
        viewer = viewers.define_viewer(name, desc, module=module, cli=cli,
                                     fmt=fmt, argv=argv, external=external)
        viewer.register(self)


import ase.io.formats as formats  # NOQA
import ase.visualize.viewers as viewers        # NOQA
