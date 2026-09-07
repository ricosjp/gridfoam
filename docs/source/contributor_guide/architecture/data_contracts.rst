.. _architecture-data-contracts:

Data contracts
==============

These contracts are shared across operators, solvers, and algorithms. Prefer
the public types in :doc:`../../api_reference/core` when linking from other
pages.

CellField
---------

``CellField.data`` has shape ``(C,) + component_shape``, where ``C`` is the
number of cells and ``component_shape`` contains the physical axes. A field
registers itself with its grid and loads configured initial and boundary
values. Fields with
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

Physical tensor layout
----------------------

Every physical array has shape ``(N,) + component_shape``. The entity axis is
first; there is no feature axis in gridfoam. Spatial dimension is currently
three, including cases with empty patches that represent two-dimensional
flow.

.. list-table:: Physical shapes
   :header-rows: 1

   * - Quantity
     - ``component_shape``
     - Array shape
   * - Scalar
     - ``()``
     - ``(N,)``
   * - Vector
     - ``(3,)``
     - ``(N, 3)``
   * - Rank-two tensor
     - ``(3, 3)``
     - ``(N, 3, 3)``

``component_shape`` is the sole shape metadata; ``tensor_rank`` and
``num_components`` are derived. Constructors take ``component_shape`` instead
of ``num_components``. Setters and packed imports reject incompatible shapes;
there is no automatic conversion from scalar columns or flattened tensors.

.. code-block:: python

   p = CellField(grid, "p", FieldRole.LOCAL, component_shape=())
   U = CellField(grid, "U", FieldRole.TRANSIENT, component_shape=(3,))
   stress = CellField(grid, "stress", FieldRole.LOCAL, component_shape=(3, 3))
   p.reset_data(0.0)
   U.reset_data([1.0, 0.0, 0.0])
   stress.reset_data([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])

Initial and prescribed boundary values omit the entity axis too. In YAML,
write ``internal: 0.0`` and ``value: 0.0`` for scalars, and nested lists for
tensors. A uniform scalar tensor, such as molecular viscosity or a trainable
scalar boundary value, has shape ``()``.

``grad`` appends the differentiation axis:
``grad(U).data[n, i, j] = dU_i/dx_j``. Interpolation, normal derivatives,
time derivatives and scalar-coefficient diffusion preserve physical tensor
axes. ``fvc.div`` integrates an already projected face flux per cell volume,
preserving that flux's tensor axes; it does not perform an additional spatial
contraction. ``fvm.div`` takes scalar advecting flux and a transported field.
Velocity reconstruction and flux construction require a vector velocity and
scalar volumetric flux. Slip boundaries support scalars and vectors; fixed
flux pressure requires a scalar pressure.

AP interpolation stores boundary and cell weights separately as
``ap_owner_bnd_weight`` / ``ap_owner_cell_weight`` and the corresponding
neighbour properties; each has shape ``(F_immersed,)``.

Volumes, area magnitudes, distances, interpolation weights, boundary mixing
fractions, and LDU coefficients are entity scalars ``(N,)``. Boundary reference
values and gradients carry the target field's tensor axes. Matrix sources,
solutions, face corrections, histories and cached values preserve those axes.
Solvers iterate physical component indices and solve each scalar component
as ``(N_cell,)``. Adjoint gradients sum over all physical tensor axes for the
shared scalar LDU coefficients.

Separate shape alignment from physical formulas. Use
``gridfoam.core.shapes.broadcast_entity`` to validate an entity scalar
``(N,)`` and create a view with singleton axes matching the physical field.
Then write the formula with ordinary arithmetic:

.. code-block:: python

   f = broadcast_entity(fraction, psi_owner)
   distance = broadcast_entity(mag_d, psi_owner)
   neumann_value = psi_owner + distance * ref_grad
   psi_boundary = f * ref_value + (1.0 - f) * neumann_value

For known axes, show the axis layout explicitly: ``coeff[:, None]`` changes
``(N,)`` to ``(N, 1)`` for vectors, and ``coeff[:, None, None]`` produces
``(N, 1, 1)`` for rank-two tensors. Use reductions such as
``(a * b).sum(dim=-1)`` to obtain the desired shape directly. Select a single
component by indexing instead of creating and then removing singleton axes;
keep the entity axis even when ``N == 1``. Add shape comments where the axis
relationship is more involved.

For rank-generic contractions and outer products,
use explicit ``einsum`` indices when they make the axis relationships clearer:
``torch.einsum("n...j,nj->n...", gradient, displacement)`` contracts the last
differentiation axis, while ``torch.einsum("n...,nj->n...j", value, area_vector)``
appends a spatial axis. Keep physical formulas visible at the call site instead
of wrapping multiplication or tensor products in helpers.

Singleton broadcast views are local temporaries, never stored field or
geometry layouts. Retain the original ``(N,)`` coefficients for scalar matrix
assembly and cached boundary states.
Conversions required by external formats belong at their boundary; for
example, VTK export flattens tensor components in the writer only.

For machine learning, ``field.data[..., None]`` appends a single feature
axis. ``torch.stack([a.data, b.data], dim=-1)`` groups fields sharing a grid
and tensor shape into features. ``ml_values[..., feature_index]`` selects a
feature without changing the physical tensor axes; use it when returning
values to gridfoam.
