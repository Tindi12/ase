""".. _ext_calc_asap:

External calculators - ASAP
---------------------------

.. _ASAP: https://asap3.readthedocs.io/en/latest/

ASAP is a calculator for doing large-scale classical molecular dynamics
within the Atomic Simulation Environment (ASE).

ASAP currently implements:

- The Effective Medium Theory (Effective Medium Theory (EMT)) potential
  for the elements Ni, Cu, Pd, Ag, Pt and Au (and their alloys).
  There is also experimental support for CuMg and CuZr alloys.
- Support for all models published by the OpenKIM.org project.
  Currently, more than 150 potentials are available this way.
  See the page OpenKIM support.
- A number of experimental or in-development potentials.

"""

# %%
#
#
#

from asap3 import EMT

from ase.build import bulk

atoms = bulk('Cu', 'fcc', a=3.614)
atoms.calc = EMT()
energy = atoms.get_potential_energy()
print(energy)
