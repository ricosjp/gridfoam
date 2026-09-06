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

:meth:`~gridfoam.core.field.FaceField.pack` concatenates these blocks along
axis 0 in the order
``[single | domain_bnd | immersed_upper | immersed_lower]``.
Immersed blocks are omitted on non-axis-projected grids.
:func:`~gridfoam.core.field.packed_face_n_rows` returns the packed row count
from the grid alone.

Operators must preserve this distinction. In particular, immersed faces must
not be treated as ordinary internal faces.

FV caches
---------

Derived FV data is owned by the object it describes, not by module globals:

* ``grid.fv_cache`` (:class:`~gridfoam.core.FvGridCache`) holds static face
  geometry (:class:`~gridfoam.fv.kernels.FaceGeometry`).
* ``field.fv_cache`` (:class:`~gridfoam.core.FvFieldCache`) holds boundary
  batches and evaluated boundary states.

Call :meth:`~gridfoam.core.grid.base.IGridBase.invalidate_derived_caches`
after topology, immersed-boundary, or device changes (``remesh``,
``update_ib``, ``to``).

FvMatrix and Equation
---------------------

``FvMatrix`` stores LDU coefficients, a source tensor, and an optional explicit
face-flux correction. Matrix algebra composes equation terms, while
``Equation`` pairs the assembled matrix with its target ``CellField`` for a
linear solver.
