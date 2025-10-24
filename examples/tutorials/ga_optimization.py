""".. _genetic_algorithm_optimization:

Genetic Algorithm - Optimization of Structures
==============================================
This example sketches a simple *screening* stage before relaxation:
we compute quick descriptor(s) for a child and estimate its energy
via a linear model fitted on previous DB entries. Only promising
children are relaxed (others are skipped), saving CPU.

For the tutorial, we:
- read existing candidates from ``gadb.db``,
- fit a tiny linear model on a naive descriptor (pair distances),
- generate a handful of new children,
- relax only those with predicted ``raw_score``
competitive with the current best.

This keeps the code/idea simple; replace the descriptor/model as needed.

The method was first described in the supplemental material of

L. B. Vilhelmsen and B. Hammer
:doi:`Systematic Study of Au6 to Au12 Gold Clusters on MgO(100)
F Centers Using Density-Functional Theory <10.1103/PhysRevLett.108.126101>`
Physical Review Letters, Vol. 108 (Mar 2012), 126101

and a full account of the method is given in

L. B. Vilhelmsen and B. Hammer
:doi:`A genetic algorithm for first principles global optimization of
supported nano structures <10.1063/1.4886337>`
Journal of Chemical Physics, Vol 141, 044711 (2014)

A Brief Overview of the Implementation
--------------------------------------

The GA relies on the ase.db module for tracking which structures have
been found. Before the GA optimization starts the user therefore needs
to prepare this database and appropriate folders. This is done through
an initialization script as the one described in the next section. In
this initialization the starting population is generated and
added to the database.


After initialization the main script is run. This script defines
objects responsible for the different parts of the GA and then creates
and locally relaxes new candidates. It is up to the user to define
when the main script should terminate. An example of a main script is
given in the next section.  Notice that because of the persistent data
storage the main script can be executed multiple times to generate new
candidates.

The GA implementation generally follows a responsibility driven
approach. This means that each part of the GA is isolated into
individual classes making it possible to put together an optimizer
satisfying the needs of a specific optimization problem.

This tutorial will use the following parts of the GA:

* A population responsible for proposing new candidates to pair
  together.
* A paring operator which combines two candidates.
* A set of mutations.
* A comparator which determines if two structures are different.
* A starting population generator.

Each of the above components are described in the supplemental
material of the first reference given above and will not be discussed
here. The example will instead focus on the technical aspect of
executing the GA.

A Basic Example
---------------
The user needs to specify the following three properties about the
structure that needs to be optimized.

* A list of atomic numbers for the structure to be optimized

* A super cell in which to do the optimization. If the structure to
  optimize resides on a surface or in a support this supercell
  contains the atoms which should not be considered explicitly by the
  GA.

* A box defining the volume of the super cell in which to randomly
  distribute the starting population.

As an example we will find the structure of a
:mol:`Ag_2Au_2` cluster on a Au(111) surface using the
EMT optimizer.

The script doing all the initialisations should be run in the folder
in which the GA optimisation is to take place. The script looks as follows:
"""

import sys
import time
from random import random

import numpy as np

from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.ga import get_parametrization
from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.data import DataConnection, PrepareDB
from ase.ga.offspring_creator import OperationSelector
from ase.ga.parallellocalrun import ParallelLocalRun
from ase.ga.pbs_queue_run import PBSQueueRun
from ase.ga.population import Population
from ase.ga.relax_attaches import VariansBreak
from ase.ga.standard_comparators import InteratomicDistanceComparator
from ase.ga.standardmutations import (
    MirrorMutation,
    PermutationMutation,
    RattleMutation,
)
from ase.ga.startgenerator import StartGenerator
from ase.ga.utilities import (
    closest_distances_generator,
    get_all_atom_types,
    get_angles_distribution,
    get_atoms_connections,
    get_atoms_distribution,
    get_rings,
)
from ase.io import read, write
from ase.optimize import BFGS

db_file = 'gadb.db'

# create the surface
slab = fcc111('Au', size=(4, 4, 1), vacuum=10.0, orthogonal=True)
slab.set_constraint(FixAtoms(mask=len(slab) * [True]))

# %%
# This generates the surface slab that we want to optimize.
# So far, it is all gold atoms, with the lowest layer fixed.
# We can visualize it with:

import matplotlib.pyplot as plt

from ase.visualize.plot import plot_atoms

fig, (ax1, ax2) = plt.subplots(1, 2)
plot_atoms(slab, ax1, rotation=('0x,0y,0z'))
plot_atoms(slab, ax2, rotation=('270x,0y,0z'))
ax1.text(1, -1, 'xy-axis view')
ax2.text(1, -1, 'xz-axis view')
ax1.set_axis_off()
ax2.set_axis_off()

# %%

# define the volume in which the adsorbed cluster is optimized
# the volume is defined by a corner position (p0)
# and three spanning vectors (v1, v2, v3)
pos = slab.get_positions()
cell = slab.get_cell()
p0 = np.array([0.0, 0.0, max(pos[:, 2]) + 2.0])
v1 = cell[0, :] * 0.8
v2 = cell[1, :] * 0.8
v3 = cell[2, :]
v3[2] = 3.0

# Define the composition of the atoms to optimize
atom_numbers = 2 * [47] + 2 * [79]

# define the closest distance two atoms of a given species can be to each other
unique_atom_types = get_all_atom_types(slab, atom_numbers)
blmin = closest_distances_generator(
    atom_numbers=unique_atom_types, ratio_of_covalent_radii=0.7
)

# create the starting population
sg = StartGenerator(
    slab, atom_numbers, blmin, box_to_place_in=[p0, [v1, v2, v3]]
)

# generate the starting population
population_size = 20
starting_population = [sg.get_new_candidate() for i in range(population_size)]


# %%
# Let's visualize the first 4 structure of the starting population:

fig, axs = plt.subplots(2, 2)
for iax, ax in enumerate(axs.reshape(-1)):
    plot_atoms(starting_population[iax], ax, rotation=('0x,0y,0z'))
    ax.set_axis_off()

# Alternatively, you can uncomment the following lines:
# from ase.visualize import view   # uncomment these lines
# view(starting_population)        # to see the starting population


# create the database to store information in
d = PrepareDB(
    db_file_name=db_file, simulation_cell=slab, stoichiometry=atom_numbers
)

for a in starting_population:
    d.add_unrelaxed_candidate(a)


# %%
# Having initialized the GA optimization we now need to actually run the
# GA. The main script running the GA consists of first an initialization
# part, and then a loop proposing new structures and locally optimizing
# them. The main script can look as follows:


# Change the following three parameters to suit your needs
population_size = 20
mutation_probability = 0.3
n_to_test = 10

# Initialize the different components of the GA
da = DataConnection('gadb.db')
atom_numbers_to_optimize = da.get_atom_numbers_to_optimize()
n_to_optimize = len(atom_numbers_to_optimize)
slab = da.get_slab()
all_atom_types = get_all_atom_types(slab, atom_numbers_to_optimize)
blmin = closest_distances_generator(all_atom_types, ratio_of_covalent_radii=0.7)

comp = InteratomicDistanceComparator(
    n_top=n_to_optimize,
    pair_cor_cum_diff=0.015,
    pair_cor_max=0.7,
    dE=0.02,
    mic=False,
)

pairing = CutAndSplicePairing(slab, n_to_optimize, blmin)
mutations = OperationSelector(
    [1.0, 1.0, 1.0],
    [
        MirrorMutation(blmin, n_to_optimize),
        RattleMutation(blmin, n_to_optimize),
        PermutationMutation(n_to_optimize),
    ],
)

# Relax all unrelaxed structures (e.g. the starting population)
while da.get_number_of_unrelaxed_candidates() > 0:
    a = da.get_an_unrelaxed_candidate()
    a.calc = EMT()
    print('Relaxing starting candidate {}'.format(a.info['confid']))
    dyn = BFGS(a, trajectory=None, logfile=None)
    dyn.run(fmax=0.05, steps=100)
    a.info['key_value_pairs']['raw_score'] = -a.get_potential_energy()
    da.add_relaxed_step(a)

# create the population
population = Population(
    data_connection=da, population_size=population_size, comparator=comp
)

# test n_to_test new candidates
for i in range(n_to_test):
    print(f'Now starting configuration number {i}')
    a1, a2 = population.get_two_candidates()
    a3, desc = pairing.get_new_individual([a1, a2])
    if a3 is None:
        continue
    da.add_unrelaxed_candidate(a3, description=desc)

    # Check if we want to do a mutation
    if random() < mutation_probability:
        a3_mut, desc = mutations.get_new_individual([a3])
        if a3_mut is not None:
            da.add_unrelaxed_step(a3_mut, desc)
            a3 = a3_mut

    # Relax the new candidate
    a3.calc = EMT()
    dyn = BFGS(a3, trajectory=None, logfile=None)
    dyn.run(fmax=0.05, steps=100)
    a3.info['key_value_pairs']['raw_score'] = -a3.get_potential_energy()
    da.add_relaxed_step(a3)
    population.update()

write('all_candidates.traj', da.get_all_relaxed_candidates())


# %%
# The above script proposes and locally relaxes 20 new candidates. To
# speed up the execution of this sample the local relaxations are
# limited to 10 steps. This restriction should not be set in a real
# application. *Note* it is important to set the ``raw_score``, as
# it is what is being optimized (maximized). It is really an input in the
# ``atoms.info['key_value_pairs']`` dictionary.
#
# The GA progress can be monitored by running the tool
# ``ase/ga/tools/get_all_candidates`` in the
# same folder as the GA. This will create a trajectory file
# ``all_candidates.traj`` which includes all locally relaxed candidates
# the GA has tried. This script can be run at the same time as the main
# script is running. This is possible because the ase.db database
# is being updated as the GA progresses.


# %%
# Running the GA in Parallel
# ==========================
#
# One of the great advantages of a GA is that many structures can be
# relaxed in parallel. This GA implementation includes two classes which
# facilitates running the GA in parallel. One class can be used for
# running several single threaded optimizations simultaneously on the
# same compute node, and the other class integrates the GA into the PBS
# queuing system used at many high performance computer clusters.
#
#
# Relaxations in Parallel on the Same Computer
# --------------------------------------------
#
# In order to relax several structures simultaneously on the same
# computer a separate script relaxing one structure needs to be
# created. Continuing the example from above we therefore create a
# script taking as input the filename of the structure to relax and
# which as output saves a trajectory file with the locally optimized
# structure. It is important that the relaxed structure is named as in
# this script, since the parallel integration assumes this file naming
# scheme.
#
# For the example described above this script could look like
# below, however, we are using the formerly written file and not the
# system argument for demonstration purooses.
# Comment the first line
# and uncomment the line below if you want to use the system argument.
# You can also directly download this
# :download:`here <ga_basic_calc_download.py>`


fname = (
    'all_candidates.traj'  # comment this if you want to use the system argument
)
# fname = sys.argv[1] #uncomment

print(f'Now relaxing {fname}')
a = read(fname)

a.calc = EMT()
dyn = BFGS(a, trajectory=None, logfile=None)
vb = VariansBreak(a, dyn)
dyn.attach(vb.write)
dyn.run(fmax=0.05)

a.info['key_value_pairs']['raw_score'] = -a.get_potential_energy()

write(fname[:-5] + '_done.traj', a)

print(f'Done relaxing {fname}')

# %%
# The main script needs to initialize the parallel controller and then
# the script needs to be changed the two places where structures are
# relaxed. The changed main script now looks like


population_size = 20
mutation_probability = 0.3
n_to_test = 10  # here you propbably want to test more


# Initialize the different components of the GA
da = DataConnection('gadb.db')
tmp_folder = 'tmp_folder/'

# An extra object is needed to handle the parallel execution
parallel_local_run = ParallelLocalRun(
    data_connection=da,
    tmp_folder=tmp_folder,
    n_simul=4,
    calc_script='ga_basic_calc_download.py',
)

atom_numbers_to_optimize = da.get_atom_numbers_to_optimize()
n_to_optimize = len(atom_numbers_to_optimize)
slab = da.get_slab()
all_atom_types = get_all_atom_types(slab, atom_numbers_to_optimize)
blmin = closest_distances_generator(all_atom_types, ratio_of_covalent_radii=0.7)

comp = InteratomicDistanceComparator(
    n_top=n_to_optimize,
    pair_cor_cum_diff=0.015,
    pair_cor_max=0.7,
    dE=0.02,
    mic=False,
)
pairing = CutAndSplicePairing(slab, n_to_optimize, blmin)
mutations = OperationSelector(
    [1.0, 1.0, 1.0],
    [
        MirrorMutation(blmin, n_to_optimize),
        RattleMutation(blmin, n_to_optimize),
        PermutationMutation(n_to_optimize),
    ],
)

# Relax all unrelaxed structures (e.g. the starting population)
while da.get_number_of_unrelaxed_candidates() > 0:
    a = da.get_an_unrelaxed_candidate()
    parallel_local_run.relax(a)

# Wait until the starting population is relaxed
while parallel_local_run.get_number_of_jobs_running() > 0:
    time.sleep(5.0)

# create the population
population = Population(
    data_connection=da, population_size=population_size, comparator=comp
)

# test n_to_test new candidates
for i in range(n_to_test):
    print(f'Now starting configuration number {i}')
    a1, a2 = population.get_two_candidates()
    a3, desc = pairing.get_new_individual([a1, a2])
    if a3 is None:
        continue
    da.add_unrelaxed_candidate(a3, description=desc)

    # Check if we want to do a mutation
    if random() < mutation_probability:
        a3_mut, desc = mutations.get_new_individual([a3])
        if a3_mut is not None:
            da.add_unrelaxed_step(a3_mut, desc)
            a3 = a3_mut

    # Relax the new candidate
    parallel_local_run.relax(a3)
    population.update()

# Wait until the last candidates are relaxed
while parallel_local_run.get_number_of_jobs_running() > 0:
    time.sleep(5.0)

write('all_candidates.traj', da.get_all_relaxed_candidates())


# %%
# Notice how the main script is not cluttered by the local optimization
# logic and is therefore now also easier to read. ``n_simul`` controls
# the number of simultaneous relaxations, and can of course also be set
# to 1 effectively giving the same result as in the non parallel
# situation.
#
# The ``relax`` method on the ``ParallelLocalRun`` class only returns
# control to the main script when there is an execution thread
# available. In the above example the relax method immediately returns
# control to the main script the first 4 times it is called, but the
# fifth time control is first returned when one of the first four
# relaxations have been completed.


# %%
# Running the GA together with a queing system
# ============================================
#
# The GA has been implemented with first principles structure
# optimization in mind. When using for instance DFT calculations for the
# local relaxations relaxing one structure can take many hours. For this
# reason the GA has been made so that it can work together with queing
# systems where each candidate is relaxed in a separate job. With this
# in mind the main script of the GA can thus also be considered a
# controller script which every time it is invoked gathers the current
# population, checks with a queing system for the number of jobs
# submitted, and submits new jobs. For a typical application the main
# script can thus be invoked by a crontab once every hour.
#
# To run the GA together with a queing system the user needs to specify
# a function which takes as input a job name and the path to the
# trajectory file that needs to be submitted (the ``jtg`` function in
# the sample script below). From this the function generates a PBS job
# file which is submitted to the queing system. The calculator script
# specified in the jobfile needs to obey the same naming scheme as the
# sample calculator script in the previous section. The sample
# relaxation script given in the previous can be used as starting point
# for a relaxation script.
#
# Handling of the parallel logic is in this case in the main script. The
# parameter n_simul given to the ``PBSQueueRun`` object determines how
# many relaxations should be in the queuing system simultaneously. The
# main script now looks the following:
#
#
# .. code-block:: python
#
#    def jtg(job_name, traj_file):
#        s = '#!/bin/sh\n'
#        s += '#PBS -l nodes=1:ppn=12\n'
#        s += '#PBS -l walltime=48:00:00\n'
#        s += f'#PBS -N {job_name}\n'
#        s += '#PBS -q q12\n'
#        s += 'cd $PBS_O_WORKDIR\n'
#        s += f'python calc.py {traj_file}\n'
#        return s
#
#
#    population_size = 20
#    mutation_probability = 0.3
#
#    # Initialize the different components of the GA
#    da = DataConnection('gadb.db')
#    tmp_folder = 'tmp_folder/'
#    # The PBS queing interface is created
#    pbs_run = PBSQueueRun(
#        da,
#        tmp_folder=tmp_folder,
#        job_prefix='Ag2Au2_opt',
#        n_simul=5,
#        job_template_generator=jtg,
#    )
#
#    atom_numbers_to_optimize = da.get_atom_numbers_to_optimize()
#    n_to_optimize = len(atom_numbers_to_optimize)
#    slab = da.get_slab()
#    all_atom_types = get_all_atom_types(slab, atom_numbers_to_optimize)
#    blmin = closest_distances_generator(all_atom_types,
#                                        ratio_of_covalent_radii=0.7)
#
#    comp = InteratomicDistanceComparator(
#        n_top=n_to_optimize,
#        pair_cor_cum_diff=0.015,
#        pair_cor_max=0.7,
#        dE=0.02,
#        mic=False,
#    )
#    pairing = CutAndSplicePairing(slab, n_to_optimize, blmin)
#    mutations = OperationSelector(
#        [1.0, 1.0, 1.0],
#        [
#            MirrorMutation(blmin, n_to_optimize),
#            RattleMutation(blmin, n_to_optimize),
#            PermutationMutation(n_to_optimize),
#        ],
#    )
#
#    # Relax all unrelaxed structures (e.g. the starting population)
#    while (
#        da.get_number_of_unrelaxed_candidates() > 0
#        and not pbs_run.enough_jobs_running()
#    ):
#        a = da.get_an_unrelaxed_candidate()
#        pbs_run.relax(a)
#
#    # create the population
#    population = Population(
#        data_connection=da, population_size=population_size, comparator=comp
#    )
#
#    # Submit new candidates until enough are running
#    while (
#        not pbs_run.enough_jobs_running()
#        and len(population.get_current_population()) > 2
#    ):
#        a1, a2 = population.get_two_candidates()
#        a3, desc = pairing.get_new_individual([a1, a2])
#        if a3 is None:
#            continue
#        da.add_unrelaxed_candidate(a3, description=desc)
#
#        if random() < mutation_probability:
#            a3_mut, desc = mutations.get_new_individual([a3])
#            if a3_mut is not None:
#                da.add_unrelaxed_step(a3_mut, desc)
#                a3 = a3_mut
#        pbs_run.relax(a3)
#
#    write('all_candidates.traj', da.get_all_relaxed_candidates())
#
# Parameterising the GA search for structure screening
# ====================================================
# Relaxing every candidate suggested by the GA is very inefficient. Many
# of these structures are poor suggestions and are immediately discarded
# when they are compared to the current population. For this reason it
# can be very effective to screen the candidate before relaxation to have
# a guess whether the candidate has a chance to enter the population or
# not. If this is not the case they can be rejected without the need for
# a costly DFT calculation. By doing this you could, for example, use a
# more drastic mutation resulting in both potentially very good but also
# very bad candidates without having to waste a lot of CPU power
# evaluating the poor suggestions.
#
# Parameterising the whole database of structures and relating the
# parameters for the individual structures to their DFT energy is one
# example of how to handle this. As the database of structures grows doing
# the GA search, the fit parameters and the guessed energy becomes more
# refined. As a result, the screening becomes more precise.
#
# Below is a sample script of how this method can be implemented and used.
# The script is a direct extension of the above tutorial. A number of
# predefined parameterising methods are available and its implementation
# is by no means restricted to the use of one of those. In the example a
# linear relationship is expected between every parameter and the DFT
# energy. The main script for the GA run hence could look like:
#
#
# .. code-block:: python
#
#    def jtg(job_name, traj_file):
#        s = '#!/bin/sh\n'
#        s += '#PBS -l nodes=1:ppn=16\n'
#        s += '#PBS -l walltime=100:00:00\n'
#        s += f'#PBS -N {job_name}\n'
#        s += '#PBS -q q16\n'
#        s += 'cd $PBS_O_WORKDIR\n'
#        s += 'NPROCS==`wc -l < $PBS_NODEFILE`\n'
#        s += 'mpirun --mca mpi_warn_on_fork 0 -np $NPROCS '
#        s += f'gpaw-python calc_gpaw.py {traj_file}\n'
#        return s
#
#
#    def combine_parameters(conf):
#        # Get and combine selected parameters
#        parameters = []
#        gets = [
#            get_atoms_connections(conf)
#            + get_rings(conf)
#            + get_angles_distribution(conf)
#            + get_atoms_distribution(conf)
#        ]
#        for get in gets:
#            parameters += get
#        return parameters
#
#
#    def should_we_skip(conf, comparison_energy, weights):
#        parameters = combine_parameters(conf)
#        # Return if weights not defined (too few completed
#        # calculated structures to make a good fit)
#        if weights is None:
#            return False
#        regression_energy = sum(p * q for p, q in zip(weights, parameters))
#        # Skip with 90% likelihood if energy appears to go up 5 eV or more
#        if (regression_energy - comparison_energy) > 5 and random() < 0.9:
#            return True
#        else:
#            return False
#
#
#    population_size = 20
#    mutation_probability = 0.3
#
#    # Initialize the different components of the GA
#    da = DataConnection('gadb.db')
#    tmp_folder = 'work_folder/'
#    # The PBS queing interface is created
#    pbs_run = PBSQueueRun(
#        da,
#        tmp_folder=tmp_folder,
#        job_prefix='Ag2Au2_opt',
#        n_simul=5,
#        job_template_generator=jtg,
#        find_neighbors=get_neighborlist,
#        perform_parametrization=combine_parameters,
#    )
#
#    atom_numbers_to_optimize = da.get_atom_numbers_to_optimize()
#    n_to_optimize = len(atom_numbers_to_optimize)
#    slab = da.get_slab()
#    all_atom_types = get_all_atom_types(slab, atom_numbers_to_optimize)
#    blmin = closest_distances_generator(all_atom_types,
#                                        ratio_of_covalent_radii=0.7)
#
#    comp = InteratomicDistanceComparator(
#        n_top=n_to_optimize,
#        pair_cor_cum_diff=0.015,
#        pair_cor_max=0.7,
#        dE=0.02,
#        mic=False,
#    )
#    pairing = CutAndSplicePairing(slab, n_to_optimize, blmin)
#    mutations = OperationSelector(
#        [1.0, 1.0, 1.0],
#        [
#            MirrorMutation(blmin, n_to_optimize),
#            RattleMutation(blmin, n_to_optimize),
#            PermutationMutation(n_to_optimize),
#        ],
#    )
#
#    # Relax all unrelaxed structures (e.g. the starting population)
#    while (
#        da.get_number_of_unrelaxed_candidates() > 0
#        and not pbs_run.enough_jobs_running()
#    ):
#        a = da.get_an_unrelaxed_candidate()
#        pbs_run.relax(a)
#
#
#    # create the population
#    population = Population(
#        data_connection=da, population_size=population_size,
#                        comparator=comp
#    )
#
#    # create the regression expression for estimating the energy
#    all_trajs = da.get_all_relaxed_candidates()
#    sampled_points = []
#    sampled_energies = []
#    for conf in all_trajs:
#        no_of_conn = list(get_parametrization(conf))
#        if no_of_conn not in sampled_points:
#            sampled_points.append(no_of_conn)
#            sampled_energies.append(conf.get_potential_energy())
#
#    sampled_points = np.array(sampled_points)
#    sampled_energies = np.array(sampled_energies)
#
#    if len(sampled_points) > 0 and
#        len(sampled_energies) >= len(sampled_points[0]):
#        weights = np.linalg.lstsq(sampled_points,
#                                  sampled_energies, rcond=-1)[0]
#    else:
#        weights = None
#
#    # Submit new candidates until enough are running
#    while (
#        not pbs_run.enough_jobs_running()
#        and len(population.get_current_population()) > 2
#    ):
#        a1, a2 = population.get_two_candidates()
#
#        # Selecting the "worst" parent energy
#        # which the child should be compared to
#        ce_a1 = da.get_atoms(a1.info['relax_id']).get_potential_energy()
#        ce_a2 = da.get_atoms(a2.info['relax_id']).get_potential_energy()
#        comparison_energy = min(ce_a1, ce_a2)
#
#        a3, desc = pairing.get_new_individual([a1, a2])
#        if a3 is None:
#            continue
#        if should_we_skip(a3, comparison_energy, weights):
#            continue
#        da.add_unrelaxed_candidate(a3, description=desc)
#
#        if random() < mutation_probability:
#            a3_mut, desc_mut = mutations.get_new_individual([a3])
#            if a3_mut is not None and not should_we_skip(
#                a3_mut, comparison_energy, weights
#            ):
#                da.add_unrelaxed_step(a3_mut, desc_mut)
#                a3 = a3_mut
#        pbs_run.relax(a3)
#
#    write('all_candidates.traj', da.get_all_relaxed_candidates())
