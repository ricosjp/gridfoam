.. _api-algorithms:

Algorithms
==========

Pressure--velocity algorithms
-----------------------------

.. currentmodule:: gridfoam.algorithms.simple

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   SIMPLE

.. currentmodule:: gridfoam.algorithms.piso

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   PISO

.. currentmodule:: gridfoam.algorithms.pimple

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   PIMPLE

Reference-value helpers
-----------------------

.. currentmodule:: gridfoam.algorithms.utils

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

Replay checkpoints
------------------

.. currentmodule:: gridfoam.algorithms.checkpoint

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   AlgorithmCheckpoint

.. currentmodule:: gridfoam.algorithms.iteration_state

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   IterationState
