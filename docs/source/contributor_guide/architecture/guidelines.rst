.. _architecture-guidelines:

Working guidelines
==================

Choose the owning layer
-----------------------

``meta``
   A user-visible configuration value, enum, or shared type.

``core``
   A mesh, field, equation, or matrix data contract used by several higher
   layers.

``boundaries``
   Boundary-state evaluation. Put a condition in ``basic`` when it depends only
   on its configured value and local geometry; use ``derived`` when it looks up
   other registered fields.

``fv``
   Reusable discretization. Put explicit evaluation in ``fvc``, implicit matrix
   assembly in ``fvm``, selectable numerical kernels in ``schemes``, and shared
   tensor operations in ``kernels``. Schemes must not import ``fvc`` or
   ``fvm``.

``models``
   Physical-property and closure models, such as transport and turbulence.

``solvers``
   Linear-system solution, preconditioning, convergence statistics, and the
   solver adjoint.

``algorithms``
   Equation sequencing, pressure--velocity coupling, correction loops, and
   time-step-level convergence.

``pre``, ``post``, and ``io``
   Initialization, analysis/diagnostics, and serialization respectively.

``runner``
   Top-level orchestration only. Reusable numerical logic should live below
   this layer.

Dependency guidelines
---------------------

* Keep ``meta`` free of numerical implementation dependencies.
* Keep ``core`` focused on shared data contracts rather than algorithm policy.
* Do not import ``algorithms`` or ``runner`` from ``fv``, ``models``, or
  ``solvers``.
* Put logic shared by SIMPLE, PISO, and PIMPLE in ``algorithms/utils``.
  Initialization code in ``pre`` may reuse those helpers, but should not own
  a full algorithm.
* Prefer factories and dispatch tables at configuration boundaries; direct
  calls are appropriate for internal helpers.
* Avoid adding another dependency across layers when an existing field,
  equation, or model contract can carry the required information.
* Import factories from their owning modules. For example,
  ``algorithms.factory.create_algorithm`` is used by ``runner`` and is not
  re-exported from ``algorithms.__init__``.

``CellField`` calls the boundary-condition factory during configured field
construction. ``Equation`` calls ``fv.boundary_ops`` to select immersed
Dirichlet cell constraints after full assembly. Keep these integration points
narrow and avoid circular imports. ``fv.adjust_phi`` and ``fv.boundary_ops``
also depend on concrete boundary-condition types; keep those dependencies
local and explicit.

Field and topology rules
------------------------

Use grid-owned metadata for tensor allocation:

* dtype: ``grid.dtype``
* device: ``grid.device``
* cell shape: ``[grid.num_cells, k]``

Do not assume all internal faces are single-sided. ``FaceField`` separates
ordinary internal values from the upper and lower sides of immersed faces.
Use helpers in ``fv.boundary_ops`` and ``fv.kernels`` instead of reconstructing
topology masks in each operator. Prefer ``grid.fv_cache`` /
``field.fv_cache`` over module-level caches; clear them through
``invalidate_derived_caches`` when the mesh or device changes.

Configuration-driven changes
----------------------------

Most configurable features follow the same path:

.. code-block:: text

   YAML value
       -> meta/enums.py
       -> meta/config.py
       -> factory or scheme dispatch table
       -> implementation
       -> unit and integration tests

Not every internal helper needs an enum or factory entry. Add those entries
only when users must select the implementation from configuration.

Some configuration enums exist before runtime dispatch is complete. For
example, ``ddtSchemes`` is accepted by ``fvSchemesConfig``, but the current
``fvm.ddt`` implementation does not yet look it up.

For a feature selected from YAML, review all of these locations:

#. ``meta/enums.py`` for the serialized value.
#. ``meta/config.py`` for validation and feature-specific parameters.
#. The owning factory or scheme dispatch table.
#. The concrete implementation and package exports.
#. Configuration parsing, unit, and integration tests.

The task-specific pages under :doc:`../extending/index` provide checklists for
common extension points.

Differentiability
-----------------

Numerical code should keep values as PyTorch tensors on the grid's
``dtype`` and ``device`` on differentiable paths. Do not convert those tensors
to NumPy arrays or Python scalars when gradients must flow. Linear solvers
default to an implicit adjoint: the primal solve runs without gradient
tracking and ``attach_implicit_adjoint`` supplies the backward path.

Krylov solvers may also set ``grad_mode="unrolled"`` to differentiate through
iterations. That option is currently an instance attribute, not a YAML field
on ``SolverConfig``.

Tests for new operators or solvers should include a gradient check whenever
the change affects a differentiable path.
