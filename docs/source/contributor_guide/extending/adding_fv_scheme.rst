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
   User-selectable numerical kernels, policy tables, and configuration
   resolution. Start at ``fv/schemes/selection.py`` for selection keys,
   built-in defaults, and aliases.
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
#. Register it in the corresponding table below. For a Laplacian variant,
   register its interpolation/correction policy in ``LAPLACIAN_SCHEMES``.
   Add alternate YAML spellings to ``LAPLACIAN_ALIASES`` in ``selection.py``;
   dispatch tables keep only canonical keys. A new time scheme registers a
   coefficient function in ``DDT_SCHEMES``. A new operator family also needs
   a typed ``search_*_scheme`` function in ``selection.py``.
#. Add focused numerical tests and a configuration parsing test.

The dispatch function type alias documents the required signature. Match its
tensor shapes exactly.

Scheme resolution and dispatch
------------------------------

All configured operators use ``fv/schemes/selection.py``. Its typed
``search_*_scheme`` functions share one lookup order: the operator-specific
key, then ``default``, then the built-in default. Alias normalization also
lives there. An absent ``divSchemes`` dictionary silently uses upwind; an
existing dictionary without a matching key or default retains its warning.

.. list-table:: Selection and implementation map
   :header-rows: 1

   * - Configuration key
     - Built-in default
     - Implementation in ``fv/schemes``
   * - ``div(<flux>, <field>)``
     - ``upwind``
     - ``div.py`` / ``DIV_SCHEMES``
   * - ``grad(<field>)``
     - ``leastsquare``
     - ``grad.py`` / ``GRAD_SCHEMES``
   * - ``laplacian(<field>)``
     - ``corrected``
     - ``laplacian.py`` / ``LAPLACIAN_SCHEMES``
   * - ``snGrad(<field>)``
     - ``corrected``
     - ``sn_grad.py`` / ``SN_GRAD_SCHEMES``
   * - ``ddt(<field>)``
     - ``euler``
     - ``ddt.py`` / ``DDT_SCHEMES``

The ``get_*_scheme`` functions map a canonical enum to a numerical function
or a Laplacian policy. YAML aliases are normalized only in
``search_laplacian_scheme``. ``fvm.laplacian`` consumes that policy when
assembling coefficients and correction fluxes; it does not interpret scheme
names. ``eval_grad`` and ``eval_sn_grad`` combine selection and evaluation
for callers that need tensor results.

The corrected normal-gradient and Laplacian paths use the shared local
least-squares hanging-face correction, independently of ``gradSchemes``.
``fvm.ddt`` and ``fvc.ddt_corr`` share time weights through
``ddt_coefficients``. History-based Euler startup lives in the
``backward`` scheme, after configuration selection.

``fvc.div`` is a face sum and ``fvc.interpolate`` currently always uses linear
interpolation, so neither requires configurable scheme resolution.
See :doc:`../../user_guide/configuration` for selection keys and defaults.

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
