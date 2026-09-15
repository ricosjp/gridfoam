.. _architecture:

Architecture
============

gridfoam is a differentiable finite-volume CFD solver built around PyTorch
tensors and a fluxel hierarchical mesh. Packages are split by responsibility
rather than by simulation case.

Packages at a glance
--------------------

``meta``
   Configuration models, enums, and shared types.

``core``
   Grid, fields, matrices, and equations.

``boundaries``
   Boundary-condition evaluation in value-fraction form.

``fv``
   Explicit (``fvc``) and implicit (``fvm``) finite-volume operators.

``models``
   Transport and turbulence closures.

``solvers``
   Linear solvers and implicit adjoints.

``algorithms``
   Pressure--velocity sequencing (SIMPLE, PISO, PIMPLE).

``optimize``
   Replayable step maps and implicit steady adjoints.

``pre``, ``post``, ``io``, ``runner``
   Initialization, diagnostics, VTU output, and top-level orchestration.

Runtime path
------------

#. Validate YAML as ``GridfoamConfig``.
#. Build the grid and register configured fields.
#. Optionally run potential-flow initialization.
#. Create SIMPLE, PISO, or PIMPLE and advance steps.
#. Emit diagnostics, force coefficients, and VTU output.

Go deeper
---------

.. toctree::
   :maxdepth: 1

   packages
   data_contracts
   state_and_geometry
   external_field_io
   steady_solve
   transient_steps
   guidelines
