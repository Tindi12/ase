from collections.abc import Mapping
from functools import cached_property
from importlib.metadata import entry_points


class Plugin:
    def __init__(self, entrypoint):
        self._entrypoint = entrypoint

    @property
    def name(self):
        return self._entrypoint.name

    def __repr__(self):
        return f'Plugin({self._entrypoint})'

    def load(self):
        # Should probably be called by some lazy mechanism
        module = self._entrypoint.load()
        register_plugin = getattr(module, '__ase_register_plugin__')
        obj = register_plugin()
        assert obj == 'STRING FROM PLUGIN'
        print('plugin was loaded')


class Plugins(Mapping):
    @cached_property
    def _plugins(self):
        return {
            entrypoint.name: Plugin(entrypoint)
            for entrypoint in entry_points(group='ase.plugin.extensions')
        }

    def __iter__(self):
        return iter(self._plugins)

    def __getitem__(self, name) -> Plugin:
        return self._plugins[name]

    def __len__(self):
        return len(self._plugins)

    def tostring(self):
        return '\n'.join(
            [
                'Plugins',
                '-' * 78,
                *[f'  {plugin}' for plugin in self.values()],
                '-' * 78,
            ]
        )


plugins = Plugins()


# We can have a number of mappings at module-level, or possibly elsewhere,
# which provide access to ioformats, calculators, etc.  For example:

# ioformats: Mapping[str, IOFormat] = IOFormats(plugins)
# calculators: Mapping[str, CalculatorMetadata] = Calculators(plugins)
# etc.


def main():
    print(plugins.tostring())
    plugin = plugins['ase-mytestplugin']
    print(plugin)

    plugin.load()

if __name__ == '__main__':
    main()
