.. _contributor-adding-fv-scheme:

Adding a finite-volume operator or scheme
=========================================

Choose the extension point by its result:

``fv/fvc``
   Explicit operators that evaluate and return tensor or field data, such as
   divergence, gradient, normal gradient, interpolation, and reconstruction.

``fv/fvm``
   Implicit operators that assemble and return an ``FvMatrix``.

``fv/schemes``
   User-selectable numerical kernels and their dispatch tables.
   Schemes must not import ``fvc`` or ``fvm``. A scheme that needs a
   configured gradient should call ``schemes.grad.eval_grad``.

``fv/kernels``
   Shared tensor kernels: interpolation weights, Green-Gauss assembly, and
   geometry coefficients.

``fv/boundary_ops.py``
   Shared iteration and evaluation for domain and immersed boundary faces.

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
dispatch tables. ``sn_grad`` uses the configured gradient via
``eval_grad``. ``fvm.ddt`` currently always uses Euler, and
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

* ``diag`` has shape ``[C]``.
* ``upper`` and ``lower`` have shape ``[F_internal]``.
* ``source`` has shape ``[C, *component_shape]``.
* Boundary contributions must use the common value-fraction contract.
  Convection uses it for both inflow and outflow; the boundary condition
  owns any flow-direction switching.
* Leave immersed cell constraints to ``equation(field, matrix)`` after all
  terms and explicit sources have been assembled. See
  :ref:`architecture-data-contracts` for constraint and flux handling.
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
