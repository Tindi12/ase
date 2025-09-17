""".. _ext_calc_gpaw:

External calculators - GPAW
---------------------------

Many external calculators can be used with ASE, including GPAW_, Abinit_,
Vasp_, Siesta_, `Quantum ESPRESSO`_, Asap_, LAMMPS_ ,
FhiAIMS_, MACE_, and many more, see :ref:`supported calculators` for the
full list.

.. _Asap: http://wiki.fysik.dtu.dk/asap
.. _GPAW: http://gpaw.readthedocs.io
.. _Siesta: http://www.icmab.es/siesta
.. _Abinit: https://www.abinit.org
.. _Vasp: https://www.vasp.at
.. _Quantum ESPRESSO: http://www.quantum-espresso.org/
.. _LAMMPS: http://lammps.sandia.gov/
.. _FhiAIMS: https://fhi-aims.org/
.. _MACE: https://github.com/ACEsuit/mace
.. _PET-MAD: https://github.com/lab-cosmo/pet-mad
"""

# %%
# Setting up an external calculator with ASE
# ==========================================
#
# This tutorial will cover how to set up a basic calculation in ASE, using
# several external  calculator, GPAW_, `Quantum ESPRESSO`_ and ASAP_ and
# their tutorials :ref:`ext_calc_qe` and :ref:`ext_calc_vasp`
# All these are various external calculators, performing DFT GPAW and
# Quantum Espresso or using classical potentials, via ASAP
#
# We will start by looking at GPAW_,
#
#
# GPAW
# ====
# - GPAW is an electronic structure code implemented as a Python
#    library with C backend.
# - GPAW includes a few command line tools, but is generally
#    always used with ASE.
# - Due to this close relationship many GPAW developers are also
#    ASE developers.
# - GPAW implements the projector-augmented wave (PAW) method which requires
#    atomic “setups” with pseudopotentials.
#
# To begin with, we run a single-point energy calculation using
# Kohn-Sham density-functional theory (DFT). First, we import GPAW
# and create the atoms object

from gpaw import GPAW, PW

from ase.build import bulk

atoms = bulk('Cu')

# %%
# Second, we attach GPAW as a calculator
# - We specify a number of parameters:
#
#   - xc sets the exchange-correlation functional
#   - kpts sets the Brillouin-zone sampling
#
# - mode sets the basis set; in this case a 400 eV cutoff plane-wave
#    basis is used.
# - For more information about valid parameters, see the GPAW docs.
#
atoms.calc = GPAW(xc='PBE', kpts=(3, 3, 3), mode=PW(400))

energy = atoms.get_potential_energy()
print(energy)
