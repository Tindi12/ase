.. _coding conventions:

==================
Coding Conventions
==================

Please follow the below conventions when writing code.

* Generally speaking, follow PEP8_.

* Use :ref:`ruff` to lint and autoformat new code.

* Use "StudlyCaps" for class names.

* Use "lowercase" or "lowercase_with_underscores" for function,
  method, and variable names.  For short names,
  joined lowercase may be used (e.g. "tagname").  Choose what is most
  readable.

* No single-character variable names, except indices in loops
  that encompass a very small number of lines
  (``for i in range(5): ...``).

* It is normally best to use named functions instead of lambda expressions.

* Use list comprehensions insteaed of functional constructs
  (filter, map, etc.).

.. _PEP8: https://www.python.org/dev/peps/pep-0008/

.. attention::

   Thus spake the Lord: Thou shalt indent with four spaces. No more, no less.
   Four shall be the number of spaces thou shalt indent, and the number of thy
   indenting shall be four. Eight shalt thou not indent, nor either indent thou
   two, excepting that thou then proceed to four. Tabs are right out.

                                          Georg Brandl


General advice
==============

 * Get rid of as many ``break`` and ``continue`` statements as possible.

 * Write short functions.  All functions should fit within a standard screen.

 * Use descriptive variable names.

Docstrings
==========

ASE follows the NumPy/SciPy convention for docstrings:

  https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard


.. _ruff:


Ruff formatter and linter
-------------------------

ASE is moving towards using `ruff <https://docs.astral.sh/ruff/>`__ to
autoformat and check all code.  New modules must be autoformatted with
ruff whereas older code may still have autoformatting
disabled by ``# fmt: off`` in order to avoid git conflicts.
The goal is to enable autoformatting on all the whole codebase.

The ASE source code must pass these checks::

  $ ruff format --check
  $ ruff check

To autoformat your code (which modifies the files!), run::

  $ ruff format
  $ ruff check --fix

.. attention::

  Please do not mix significant automated changes by ruff with other
  changes since this is makes code review difficult.
