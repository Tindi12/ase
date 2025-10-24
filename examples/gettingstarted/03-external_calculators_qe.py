""".. _ext_calc_qe:

External calculators - QE
-------------------------

.. _Quantum ESPRESSO: http://www.quantum-espresso.org/

- Unlike GPAW, we are going to call the program with MPI from within our
   regular Python interpreter.
- We define the mpi command when instantiating the calculator: the command
   might need to be tweaked for different machines with different
   parallel environments.
- This information is captured in a “Profile” object.

Getting the data
================

We will use the SSSP-efficiency pseudopotential set.

"""

# %%
#
# Profiles are a fairly new ASE feature and not yet used by all such
# Calculators. An alternative way to manage these commands is by setting
# environment variables, e.g. ASE_ESPRESSO_COMMAND. Check the docs for
# each calculator to see what is currently implemented.

from ase.build import bulk
from ase.calculators.espresso import Espresso

# pseudo_dir = datapaths.DataFiles().paths['espresso'][0]
# if pseudo_dir.exists():
#     print(f'using pseudopotentials from {pseudo_dir}')
# command = 'mpirun /home/ase/calculators/espresso/bin/pw.x'
# profile = EspressoProfile(command=command, pseudo_dir=pseudo_dir)

# %%
# Each Calculator has its own keywords to match the input syntax of the
# corresponding software code. You can see below that the keywords for
# the Espresso() class are different to those from the GPAW() class.
# This is because each software code requires different input parameters.
# For QE, the content of input_data contains the parameters for the
# calculation input file.


calc = Espresso(
    kpts=(3, 3, 3),
    input_data={
        'control': {'tprnfor': True, 'tstress': True},
        'system': {'ecutwfc': 50.0},
    },
    pseudopotentials={'Si': 'si_lda_v1.uspp.F.UPF'},
)

# %%
# Once we have setup the calculator we use the same three step process to
# retrieve a property
#
# - The difficult part is setting up the calculator!
# - Once setup is complete, we can get a total energy using the same
#   three-step process.

atoms = bulk('Si')
atoms.calc = calc
energy = atoms.get_potential_energy()
print(energy)
