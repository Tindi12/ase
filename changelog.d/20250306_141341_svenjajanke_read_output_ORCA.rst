.. A new scriv changelog fragment.
..
.. Uncomment the header that is right (remove the leading dots).
..
I/O
---

- ase.io.orca: implemented creation of atoms objects from ORCA output
- ase.io.orca: implemented processing of ORCA output in chunks to process
  geometry optimizations
- ase.io.orca: implemented read-in of forces from ORCA general output file
- **BREAKING** ase.io.orca: old `read_orca_output` renamed to
  `read_orca_output_results`. Old function returns dictionary, making it
  necessary to create new `read_orca_output` function that returns atoms
  object
- ase.io.formats: modified to allow processing of ORCA output via ase.io.read
- ase.test.test_orca: modified tests to reflect changes and added test
  of ase.io.read for an ORCA output.
..
.. Calculators
.. -----------
..
.. - A bullet item for the Calculators category.
..
.. Optimizers
.. ----------
..
.. - A bullet item for the Optimizers category.
..
.. Molecular dynamics
.. ------------------
..
.. - A bullet item for the Molecular dynamics category.
..
.. GUI
.. ---
..
.. - A bullet item for the GUI category.
..
.. Development
.. -----------
..
.. - A bullet item for the Development category.
..
.. Other changes
.. -------------
..
.. - A bullet item for the Other changes category.
..
.. Bugfixes
.. --------
..
.. - A bullet item for the Bugfixes category.
..
