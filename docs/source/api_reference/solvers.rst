.. _api-solvers:

Linear solvers
==============

Factory
-------

.. currentmodule:: gridfoam.solvers

.. autosummary::
   :toctree: generated
   :template: autosummary/function.rst

   create_solver

Base types
----------

.. currentmodule:: gridfoam.solvers.base

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   LinearSolver
   SolveResult
   SolveStats

Adjoint utilities
-----------------

.. currentmodule:: gridfoam.solvers.adjoint

.. autosummary::
   :toctree: generated
   :template: autosummary/function.rst

   attach_implicit_adjoint
   assemble_ldu_grads
