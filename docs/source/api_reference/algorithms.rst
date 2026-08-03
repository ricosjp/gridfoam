.. _api-algorithms:

Algorithms
==========

Pressure--velocity algorithms
-----------------------------

.. currentmodule:: gridfoam.algorithms

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   SIMPLE
   PISO
   PIMPLE

Reference-value helpers
-----------------------

.. autosummary::
   :toctree: generated
   :template: autosummary/function.rst

   needs_reference_value
   set_reference_value

Factory and base
----------------

.. currentmodule:: gridfoam.algorithms.factory

.. autosummary::
   :toctree: generated
   :template: autosummary/function.rst

   create_algorithm

.. currentmodule:: gridfoam.algorithms.base

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   AlgorithmBase
