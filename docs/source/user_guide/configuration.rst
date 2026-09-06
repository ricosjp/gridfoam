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
   IBM type, and optional ``motion`` (``static`` or ``dynamic``).
   ``static`` (default) builds the mesh once. ``dynamic`` keeps a fluxel
   session so the IBM can be updated when the boundary moves.

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
   Discretization schemes for time derivatives, gradients, divergence,
   surface-normal gradients, and Laplacian terms. ``divSchemes``,
   ``gradSchemes`` (default ``leastsquare``), ``snGradSchemes`` and
   ``laplacianSchemes`` (default ``corrected``) are actively dispatched;
   ``ddtSchemes`` is accepted but not yet wired to a runtime operator.
   On an octree grid the ``corrected`` schemes add the explicit skewness
   correction on hanging-node (2:1) faces only; ``uncorrected`` skips it.

``simulator.fvSolution``
   Pressure--velocity algorithm (SIMPLE, PISO, or PIMPLE), linear solvers per
   field, optional potential-flow initialization, and ``adjustPhi`` behaviour.
   ``consistent: true`` selects SIMPLEC (``rAtU = 1/(1/A - H1)``).

``simulator.conditions``
   Initial field values and boundary conditions per patch.

``simulator.properties``
   Transport and turbulence model selection.

Field and solver keys
---------------------

Field names in ``fvSolution.solvers`` follow OpenFOAM conventions. A
``pFinal`` solver, if present, is used only on the last non-orthogonal
pass of the last pressure corrector (every PISO/PIMPLE outer iteration).

Boundary patches may use reserved domain names such as ``x_minus`` and
``x_plus``, or custom patch names for immersed surfaces.

Residual control
----------------

``residualControl`` lists fields and a ``tolerance`` (absolute). A float
shorthand is that absolute tolerance.

SIMPLE stops the run when every listed field is below ``tolerance``.
``rel_tolerance`` is ignored.

PIMPLE uses the same check to end the *time step*, not the run. It tests
the previous outer iteration before starting the next, skipping the first
and the scheduled last. A pass still runs one final outer iteration.
Each field converges if ``residual < tolerance``, or if
``rel_tolerance > 0`` and
``residual < rel_tolerance * residual0`` (``residual0`` is the first
solve of that time step). Pressure uses the initial residual of the last
pressure solve, including non-orthogonal corrections.

These residuals are RHS-normalized L2, not OpenFOAM's scaled L1. Do not
copy OpenFOAM tolerance values expecting the same magnitude.

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
