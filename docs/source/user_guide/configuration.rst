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
   The coupling follows the OpenFOAM ``pEqn.H`` structure: the predicted
   flux ``phiHbyA`` is built with ``constrainHbyA`` on every face block,
   ``adjustPhi`` is applied to it before the pressure solve (only when the
   pressure level is not fixed by a Dirichlet patch), PISO/PIMPLE add
   ``ddtCorr``, and ``phi = phiHbyA - pEqn.flux()`` is evaluated on internal
   and boundary faces. ``consistent: true`` selects the SIMPLEC formulation
   (``rAtU = 1/(1/A - H1)``) for SIMPLE and PIMPLE. ``residualControl`` for
   ``p`` uses the residual of the unsolved pressure equation, as in OpenFOAM.

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
