.. _contributor-adding-fv-scheme:

Adding a finite-volume operator or scheme
=========================================

Choose the extension point by its result:

``fv/fvc``
   Explicit operators that evaluate and return tensor or field data, such as
   divergence, gradient, normal gradient, and interpolation.

``fv/fvm``
   Implicit operators that assemble and return an ``FvMatrix``.

``fv/schemes``
   User-selectable numerical kernels and their dispatch tables.

``fv/boundary_ops.py``
   Shared iteration and evaluation for domain and immersed boundary faces.

``fv/mesh_geometry.py``
   Reusable geometry coefficients independent of a particular equation.

Adding a selectable scheme
--------------------------

#. Add the YAML value to the appropriate enum in ``meta/enums.py``.
#. Confirm that the corresponding dictionary in ``fvSchemesConfig`` accepts
   that enum.
#. Implement the kernel beside related schemes.
#. Register it in the module's dispatch table, such as ``DIV_SCHEMES`` or
   ``GRAD_SCHEMES``.
#. Add focused numerical tests and a configuration parsing test.

The dispatch function type alias documents the required signature. Match its
tensor shapes exactly.

Current dispatch coverage
-------------------------

``div`` and ``grad`` schemes are selected from ``fvSchemes`` through
dispatch tables. ``fvm.ddt`` currently always uses Euler, and
``fvm.laplacian`` has a single implementation, even though the corresponding
configuration dictionaries already exist. When adding a selectable ddt or
laplacian scheme, wire the lookup in the operator as part of the same change.

Adding an explicit operator
---------------------------

An ``fvc`` operator should:

* preserve the input grid's dtype and device;
* use existing interpolation and boundary helpers;
* handle ordinary internal, domain-boundary, and double-sided immersed faces;
* document input and output shapes with jaxtyping annotations;
* remain differentiable when its inputs require gradients.

Do not flatten all face categories into one tensor unless the operation
explicitly defines a reversible mapping.

Adding an implicit operator
---------------------------

An ``fvm`` operator assembles LDU coefficients and source terms for the target
``CellField``. Follow these rules:

* ``diag`` has shape ``[C, 1]``.
* ``upper`` and ``lower`` have shape ``[F_internal, 1]``.
* ``source`` has shape ``[C, k]``.
* Boundary contributions must use the common value-fraction contract.
* Store explicit non-orthogonal face-flux terms in
  ``FvMatrix.face_flux_correction`` when the matrix flux must reproduce them.
* Matrix addition and subtraction must continue to represent equation-term
  composition.

Testing checklist
-----------------

Use the smallest mesh that exposes the behavior under test. Cover:

* constant and linear fields with known analytical results;
* scalar and multi-component fields;
* internal and domain-boundary contributions;
* upper and lower immersed faces when applicable;
* non-orthogonal geometry for corrected schemes;
* conservation or coefficient symmetry where mathematically required;
* gradient propagation for differentiable paths.

Operator integration tests belong under ``tests/integration/fv``. Pure limiter,
dispatch, or coefficient logic can be tested as a unit.
