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
  geometry (:class:`~gridfoam.fv.kernels.FaceGeometry`) and read-only patch
  masks with their CPU face counts.
* ``field.fv_cache`` (:class:`~gridfoam.core.FvFieldCache`) holds boundary
  batches (including area vectors and magnitudes), evaluated boundary states,
  and fixed Dirichlet constraint selections. Constraint selections cache only
  geometry choices; prescribed values and their autograd graph are rebuilt.

Call :meth:`~gridfoam.core.grid.base.GridBase.invalidate_derived_caches`
after topology, immersed-boundary, or device changes (``remesh``,
``update_ib``, ``to``).

FvMatrix and Equation
---------------------

``FvMatrix`` stores LDU coefficients, a source tensor, and an optional explicit
face-flux correction. Matrix algebra composes equation terms, while
``Equation`` pairs the assembled matrix with its target ``CellField`` for a
linear solver. Assemble every term and explicit source before calling
``equation(field, matrix)``. This applies immersed Dirichlet cell constraints
to the solve matrix without modifying the input; without constraints, the
input is reused. Keep the original matrix for physical flux reconstruction.
See `Immersed Dirichlet constraints`_ for the constraint and flux contracts.

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

Boundary convection
-------------------

``fvm.div`` assembles boundary convection using the value-fraction contract.
On domain and immersed boundaries, it uses the value prescribed by the
boundary condition for either flux direction. Dirichlet conditions supply
the fixed value; zero-gradient conditions supply the adjacent cell value.
``inletOutlet`` itself selects zero gradient on outflow and the prescribed
value on reverse inflow. A nonzero Neumann gradient contributes through
the cell-to-boundary extrapolation, including on outflow.

Immersed Dirichlet constraints
------------------------------

AP geometry stores actual ray distances in ``ap_dist_owner_to_bnd`` and
``ap_dist_neighbour_to_bnd``. The boolean arrays ``ap_owner_near_boundary``
and ``ap_neighbour_near_boundary`` identify candidate boundary cells. All
four arrays have shape ``(F_immersed,)``. They replace the former boundary
and cell ghost weights; Fluxel and gridfoam must be updated together.

For cell width ``dx`` and the largest domain extent ``L``, Fluxel marks
``distance / dx <= dx / L``. This is the condition in Gibou et al.,
*A Second Order Accurate Symmetric Discretization of the Poisson Equation
on Irregular Domains*, p. 8, expressed in coordinates normalized by ``L``.
Distances are never replaced by cell widths, including at zero distance.
The flags describe geometry only: Neumann or partially mixed conditions
are not converted into Dirichlet constraints. A partially mixed condition
at exactly zero distance raises ``ValueError`` because its value contribution
cannot be evaluated by division by the distance.

``equation(field, matrix)`` applies the near-boundary Dirichlet constraints
**after** assembling the full equation and its explicit sources. When
constraints are present, it creates a separate solve matrix using
``FvMatrix.with_fixed_values``: prescribed columns move to the source, both
row and column couplings are removed, and the prescribed row becomes a
scaled identity row. Symmetry is preserved. Nonzero diagonal entries are
retained; zero entries use a unit value with the sign of the diagonal sum
(positive if zero). Tensor components share the same constrained cells.
When multiple Dirichlet faces constrain one cell, the closest hit supplies
the value (ties follow boundary-batch/face order).

The original matrix remains available for physical flux reconstruction;
using the eliminated solve matrix for this would lose internal face fluxes.
At ordinary immersed faces, diffusion uses ``gamma * area / distance``
exactly once. On snapped Dirichlet cells the Poisson row no longer specifies
a flux balance: the pressure correction recovers the total near-boundary
flux from that cell's mass balance, distributing it over its near Dirichlet
faces in proportion to area. This preserves cell continuity but is not a
pointwise, second-order approximation of the boundary-normal derivative.
The placeholder zero returned by ``sn_grad`` on snapped Dirichlet faces
must not be interpreted as an imposed Neumann condition. The original
paper's second-order result concerns the solution of its Dirichlet Poisson
problem, not a general guarantee for gradients or Navier--Stokes solutions.

Momentum assembly retains the pressure-free equation. ``with_source`` creates
an independent predictor matrix with the integrated pressure force added to
its right-hand side, preserving autograd connections and face corrections.
Cell constraints are applied to the predictor after adding pressure, and
separately to the pressure-free equation used for ``H/A``.
Explicit velocity correction and pressure relaxation reapply the prescribed
cell values. Boundary-snapping choices are discrete geometry decisions;
autograd differentiates the equation and prescribed values for fixed choices.

Reference: https://physbam.stanford.edu/papers/cam2000-37.pdf
