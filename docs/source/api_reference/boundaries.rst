.. _api-boundaries:

Boundary conditions
===================

Factory
-------

.. currentmodule:: gridfoam.boundaries

.. autosummary::
   :toctree: generated
   :template: autosummary/function.rst

   create_boundary_condition

Base contract
-------------

.. currentmodule:: gridfoam.boundaries.base

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   BoundaryCondition

Basic conditions
----------------

.. currentmodule:: gridfoam.boundaries.basic

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   DirichletBC
   NeumannBC
   EmptyBC
   SlipBC

Derived conditions
------------------

.. currentmodule:: gridfoam.boundaries.derived

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   FixedFluxPressure
   InletOutletBC
