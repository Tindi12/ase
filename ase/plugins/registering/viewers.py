""" This module registers all built-in viewer. """
import sys


def ase_register(plugin):

    def F(*args, **kwargs):
        plugin.register_viewer(*args, **kwargs, external=False)

    F("ase", "View atoms using ase gui.")
    F("ngl", "View atoms using nglview.")
    F("mlab", "View atoms using matplotlib.")
    F("sage", "View atoms using sage.")
    F("x3d", "View atoms using x3d.")

    # CLI viweers that are internally supported
    F(
        "avogadro", "View atoms using avogradro.", cli=True, fmt="cube",
        argv=["avogadro"]
    )
    F(
        "ase_gui_cli", "View atoms using ase gui.", cli=True, fmt="traj",
        argv=[sys.executable, '-m', 'ase.gui'],
    )
    F(
        "gopenmol",
        "View atoms using gopenmol.",
        cli=True,
        fmt="extxyz",
        argv=["runGOpenMol"],
    )
    F(
        "rasmol",
        "View atoms using rasmol.",
        cli=True,
        fmt="proteindatabank",
        argv=["rasmol", "-pdb"],
    )
    F("vmd", "View atoms using vmd.", cli=True, fmt="cube", argv=["vmd"])
    F(
        "xmakemol",
        "View atoms using xmakemol.",
        cli=True,
        fmt="extxyz",
        argv=["xmakemol", "-f"],
    )
