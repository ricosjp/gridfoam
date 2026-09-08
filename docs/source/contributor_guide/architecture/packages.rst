.. _architecture-packages:

Packages
========

Use this page to find where a change belongs. ``__init__.py`` files are
omitted from the tree.

Source map
----------

.. code-block:: text

   src/gridfoam/
   ├── meta/            # config, enums, shared types
   ├── core/            # grid, field, fvmatrix, equation
   ├── boundaries/      # basic/ and derived/ conditions
   ├── fv/              # fvc/, fvm/, schemes/, kernels/, flux helpers
   ├── models/          # transport/ and turbulence/
   ├── solvers/         # krylov, pyamg, adjoint/
   ├── algorithms/      # simple, piso, pimple, utils/
   ├── pre/             # potential flow
   ├── post/            # diagnostics, forces
   ├── io/              # VTU export
   ├── runner.py
   ├── runtime_config.py
   └── logging.py

Responsibilities
----------------

``meta``
   Configuration models, enums, and shared type aliases. New values exposed in
   YAML normally begin here.

``core``
   Mesh interfaces, cell- and face-centred fields, finite-volume matrices, and
   equations. Grid- and field-owned FV cache containers (``fv_cache``) live
   here without importing their numerical builders. ``CellField`` constructs
   configured boundary conditions, and ``Equation`` uses ``fv.boundary_ops``
   to select immersed Dirichlet cell constraints. These are narrow integration
   points; operator assembly and algorithm policy remain in their own layers.

``boundaries``
   Boundary-condition contracts and implementations. Every condition is
   reduced to value-fraction form: ``(fraction, ref_value, ref_grad)``.

``fv``
   Finite-volume operators and scheme dispatch. ``fvc`` operators evaluate
   fields explicitly, while ``fvm`` operators assemble an ``FvMatrix``.
   Selectable discretizations live in ``schemes`` and must not import
   ``fvc`` or ``fvm``. Shared tensor kernels live in ``kernels`` and fill
   slots on ``grid.fv_cache`` / ``field.fv_cache``. Some FV helpers, such
   as ``adjust_phi``, also depend on boundary-condition types.

``models``
   Transport and turbulence models that provide physical properties and close
   the governing equations. Models currently depend on ``core`` and ``meta``;
   they do not import ``fv``.

``solvers``
   Linear solvers, preconditioners, solver-key resolution, and implicit
   adjoint support.

``algorithms``
   Pressure--velocity coupling and time-stepping algorithms such as SIMPLE,
   PISO, and PIMPLE. Shared helpers live under ``algorithms/utils``. The
   algorithm base may also emit diagnostics through ``post``.

``pre``, ``post``, and ``io``
   Potential-flow initialization, diagnostics and force evaluation, and VTU
   output. ``pre`` may reuse helpers from ``algorithms/utils`` without owning
   a full algorithm.

``runner``
   Top-level orchestration: parse configuration, create the grid and algorithm,
   run steps, collect diagnostics, and write output.
