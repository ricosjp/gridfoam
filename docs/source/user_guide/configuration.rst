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
   ``laplacianSchemes`` (default ``corrected``) are actively dispatched.
   ``ddtSchemes`` selects ``euler`` (default) or ``backward`` (BDF2).
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

Field values and boundary behavior
----------------------------------

Dirichlet values apply on both inflow and outflow. Use ``inlet_outlet`` to
select a prescribed value on reverse inflow and zero gradient on outflow.
Nonzero Neumann gradients extrapolate from the adjacent cell in either
direction.

Second-order time integration
-----------------------------

Use ``backward`` for second-order backward differentiation (BDF2), following
OpenCFD OpenFOAM v2606's stationary-volume time coefficients:

.. code-block:: yaml

   simulator:
     fvSchemes:
       ddtSchemes:
         default: backward

Field-specific keys such as ``ddt(U): backward`` override ``default``.
Existing configurations retain Euler. For constant time steps BDF2 uses
``(1.5*q_new - 2*q_old + 0.5*q_older) / deltaT``; the coefficients also
account for unequal consecutive step sizes. The built-in runner still uses
a fixed ``deltaT``; adaptive time-step selection is not added by this option.

``fvm.ddt`` and PISO/PIMPLE's ``fvc.ddt_corr`` use the same time weights.
Cell fields using ``backward`` must have role ``TRANSIENT``. Two previous
levels and the previous interval are retained. The first step uses Euler
because there is only one previous level. PISO/PIMPLE update their histories
once per completed time step, regardless of the number of correctors.
Constructing a new algorithm starts fresh history from the current values.

For manual scalar solves, set the initial data and call
``field.update_history(reset=True)`` before the first step. After each
completed solve, call ``field.update_history()`` once, while ``grid.dt``
still refers to that completed step. For manual velocity/flux coupling,
advance or reset ``U`` and ``phi`` histories together. Inconsistent BDF2
histories raise an error in ``ddt_corr``.

Remeshing and IBM geometry synchronization discard the second history
level and restart with Euler. Repeated geometry changes therefore do not
retain second-order time accuracy. This implementation assumes stationary
cell volumes between stored time levels; it does not add moving-volume
ALE/GCL time terms or conservative transfer of two old mesh histories.
Stored old levels remain connected to autograd for transient sensitivities.

Discontinuous diffusion coefficients
-----------------------------------

For a scalar diffusivity that jumps across cell faces, select harmonic
interpolation for the corresponding Laplacian:

.. code-block:: yaml

   simulator:
     fvSchemes:
       laplacianSchemes:
         default: corrected
         laplacian(T): Gauss harmonic corrected

This applies to ``fvm.laplacian(gamma, T)``. The lookup key uses the name of
the transported field (``T``), since ``gamma`` is a tensor or scalar rather
than a named field in this API. The spelling of the scheme matches OpenCFD
OpenFOAM v2606; the lookup syntax remains gridfoam's existing syntax.

The face value is ``(d_O + d_N) / (d_O/gamma_O + d_N/gamma_N)``, where the
distances are measured along the face normal. Thus materials in series
carry a common diffusion flux even when their coefficients or cell widths
differ greatly. The same coefficient multiplies the implicit flux and the
explicit hanging-face correction. ``Gauss harmonic uncorrected`` omits that
correction. ``Gauss linear corrected`` and ``Gauss linear uncorrected`` are
also accepted; existing ``linear``, ``corrected`` and ``uncorrected`` values
retain linear coefficient interpolation.

Harmonic interpolation accepts finite, nonnegative scalar diffusivity.
A zero coefficient blocks its internal faces; negative or nonfinite values
raise an error. The all-zero mean has no unique derivative, and the
implementation uses a finite autograd convention at that point.
Boundary diffusivity is still the adjacent cell value. This option does
not locate an interface inside a cell or implement anisotropic diffusion
or contact resistance. The existing skewness correction also does not
reconstruct separate gradients on each side of a material interface;
general skewed discontinuous problems still require convergence checks.

Linear solver convergence
-------------------------

CG, BiCGSTAB, and PyAMG use the configured ``norm_type`` to measure the
unnormalized algebraic residual ``b - A x``. Each component converges when
its residual norm is less than ``max(tolerance, rel_tolerance * initial)``;
``initial`` is measured from that component's initial guess for this solve.
Setting ``rel_tolerance`` to zero leaves only the absolute criterion.

PyAMG checks the true residual after each V-cycle and reports actual
initial/final residuals and cycle counts per component. An initial guess
already within tolerance takes zero cycles. Exhausting ``max_iter`` without
meeting the criterion returns ``converged=False``. The same tolerance and
norm settings apply to the transpose solve used by the implicit adjoint.
These linear-solver criteria are distinct from outer ``residualControl``.

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
