.. _architecture-data-contracts:

Data contracts
==============

These contracts are shared across operators, solvers, and algorithms. Prefer
the public types in :doc:`../../api_reference/core` when linking from other
pages.

CellField
---------

``CellField.data`` has shape ``[C, k]``, where ``C`` is the number of cells and
``k`` is the number of components. A field registers itself with its grid and
loads configured initial and boundary values. Fields with
``FieldRole.TRANSIENT`` keep a cloned previous time level; other roles do not.

FaceField
---------

Face values are separated by topology:

* ``single_data`` stores ordinary, single-sided internal faces.
* ``domain_bnd_data`` stores domain-boundary faces.
* ``immersed_upper`` and ``immersed_lower`` store the two sides of immersed
  faces on an ``AxisProjectedGrid``.

Operators must preserve this distinction. In particular, immersed faces must
not be treated as ordinary internal faces.

FvMatrix and Equation
---------------------

``FvMatrix`` stores LDU coefficients, a source tensor, and an optional explicit
face-flux correction. Matrix algebra composes equation terms, while
``Equation`` pairs the assembled matrix with its target ``CellField`` for a
linear solver.
