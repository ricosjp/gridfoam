.. _user-configuration:

Configuration
=============

Simulations are driven by YAML files validated as
:class:`~gridfoam.meta.config.GridfoamConfig`. Paths inside a config file are
resolved relative to the repository root unless you change the working
directory before running a case.

Top-level structure
-------------------

A typical configuration contains:

``fluxel``
   Mesh generation: domain bounds, octree resolution, immersed-surface path,
   and IBM type.

``simulator``
   Solver settings, boundary and initial conditions, and output control.

``version``
   Configuration schema version.

Simulator blocks
----------------

``simulator.control``
   Time step, end time, write interval, output directory, and floating-point
   precision.

``simulator.fvSchemes``
   Discretization schemes for time derivatives, gradients, divergence, and
   Laplacian terms. ``divSchemes`` and ``gradSchemes`` are actively dispatched;
   ``ddtSchemes`` and ``laplacianSchemes`` are accepted but not yet wired to
   runtime operators.

``simulator.fvSolution``
   Pressure--velocity algorithm (SIMPLE, PISO, or PIMPLE), linear solvers per
   field, optional potential-flow initialization, and ``adjustPhi`` behaviour.

``simulator.conditions``
   Initial field values and boundary conditions per patch.

``simulator.properties``
   Transport and turbulence model selection.

Field and solver keys
---------------------

Field names in ``fvSolution.solvers`` follow OpenFOAM conventions. A final
pressure-correction solver can be configured with a ``pFinal`` key.

Boundary patches may use reserved domain names such as ``x_minus`` and
``x_plus``, or custom patch names for immersed surfaces.

Example excerpt
---------------

.. literalinclude:: ../../../examples/cavity/gridfoam/data/config.yaml
   :language: yaml
   :lines: 15-59

Further reading
---------------

* :doc:`running_cases` — paths and hardware notes for each bundled example
* :doc:`../contributor_guide/extending/configuration` — how to extend
  configuration models and enums
* :doc:`../api_reference/meta` — configuration and enum reference
