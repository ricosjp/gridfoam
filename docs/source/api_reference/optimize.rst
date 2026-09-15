.. _api-optimize:

Differentiable solves and time steps
========================================

Import from the owning modules; these APIs are not re-exported by ``core``.
See :doc:`../contributor_guide/architecture/steady_solve` for the state contract,
convergence criteria, and the supported scope.

.. currentmodule:: gridfoam.optimize.simple

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   SimpleStepMap

.. currentmodule:: gridfoam.optimize.transient

See :doc:`../contributor_guide/architecture/transient_steps` for physical-time
boundaries, fixed controls, history blocks and replay guarantees.

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   TransientState
   TransientStepMap
   PisoStepMap
   PimpleStepMap

.. currentmodule:: gridfoam.optimize.steady

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   SteadyOptions
   ConvergenceError

.. autosummary::
   :toctree: generated
   :template: autosummary/function.rst

   steady_solve
