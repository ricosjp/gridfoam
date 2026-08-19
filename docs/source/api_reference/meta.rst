.. _api-meta:

Configuration and enums
=======================

Top-level configuration
-----------------------

.. currentmodule:: gridfoam.meta

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   GridfoamConfig

Enums
-----

.. currentmodule:: gridfoam.meta.enums

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   IbmType
   MeshMotion
   DeviceType
   DomainBoundaryPatch
   FieldRole
   DdtScheme
   GradScheme
   DivScheme
   LaplacianScheme
   SolverType
   NormType
   PreconditionerType
   PrecisionType
   FaceSide
   BoundaryConditionType
   ForceCoordMode
   AlgorithmType
   TransportModelType
   TurbulenceType

Field-level configuration models live in :mod:`gridfoam.meta.config` and are
validated as part of :class:`~gridfoam.meta.GridfoamConfig`. See
:doc:`../user_guide/configuration` for YAML structure.
