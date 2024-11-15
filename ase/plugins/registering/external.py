"""
In this plugins, external calculators/viewers/formats
(either the old long known by ASE, or the ones registered using entry-points)
are present. This way of registering calculators/... is deprecated.

Formerly, view plugins could be registered through the entrypoint system in
with the following in a module, such as a `viewer.py` file:

```python3
VIEWER_ENTRYPOINT = ExternalViewer(
    desc="Visualization using <my package>",
    module="my_package.viewer"
)
```

Where module `my_package.viewer` contains a `view_my_viewer` function taking
and `ase.Atoms` object as the first argument, and also `**kwargs`.

Then ones needs to register an entry point in `pyproject.toml` with

```toml
[project.entry-points."ase.visualize"]
my_viewer = "my_package.viewer:VIEWER_ENTRYPOINT"
```

After this, call to `ase.visualize.view(atoms, viewer='my_viewer')` will be
forwarded to `my_package.viewer.view_my_viewer` function.
"""

import warnings
from importlib.metadata import entry_points

from ase.utils.plugins import ExternalIOFormat, ExternalViewer

plugin_name = 'external'


def register_external_io_format(plugin, entry_point):
    """ Used in test """

    fmt = entry_point.load()
    # if entry_point.name in ioformats:
    #    raise ValueError(f'Format {entry_point.name} already defined')
    if not isinstance(fmt, ExternalIOFormat):
        raise TypeError('Wrong type for registering external IO formats '
                        f'in format {entry_point.name}, expected '
                        'ExternalIOFormat')
    return plugin.register_io_format(name=entry_point.name, **fmt._asdict(),
                                     external=True)


def _register_external_io_formats(plugin, group):

    if hasattr(entry_points(), 'select'):
        fmt_entry_points = entry_points().select(group=group)
    else:
        fmt_entry_points = entry_points().get(group, ())

    for entry_point in fmt_entry_points:
        try:
            register_external_io_format(plugin, entry_point)
        except Exception as exc:
            warnings.warn(
                'Failed to register external '
                f'IO format {entry_point.name}: {exc}'
            )


def _register_external_viewer_formats(plugin, group):

    def define_external_viewer(entry_point):
        """Define external viewer"""

        viewer_def = entry_point.load()
        # if entry_point.name in VIEWERS:
        #    raise ValueError(f"Format {entry_point.name} already defined")
        if not isinstance(viewer_def, ExternalViewer):
            raise TypeError(
                "Wrong type for registering external Viewer"
                f"in format {entry_point.name}, expected "
                "ExternalViewer"
            )
        return plugin.register_viewer(name=entry_point.name,
                                      **viewer_def._asdict(),
                                      external=True)

    if hasattr(entry_points(), "select"):
        viewer_entry_points = entry_points().select(group=group)
    else:
        viewer_entry_points = entry_points().get(group, ())

    for entry_point in viewer_entry_points:
        try:
            define_external_viewer(entry_point)
        except Exception as exc:
            warnings.warn(
                "Failed to register external "
                f"Viewer {entry_point.name}: {exc}"
            )


def ase_register(plugin):
    plugin.register_calculator("asap3.EMT", 'asap')
    plugin.register_calculator("gpaw.GPAW")
    plugin.register_calculator("hotbit.Calculator", name='hotbit')

    # Register IO formats exposed through the ase.ioformats entry point
    _register_external_io_formats(plugin, 'ase.ioformats')
    _register_external_viewer_formats(plugin, 'ase.visualize')
