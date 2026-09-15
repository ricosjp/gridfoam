.. _api-reference:

API reference
=============

Public Python API for gridfoam. Pages list symbols explicitly; internal
implementation modules are omitted.

``import gridfoam`` exposes the ``core``, ``fv``, and ``meta`` namespaces.
For example, clients can use ``gridfoam.core.create_grid``,
``gridfoam.core.FieldBindings``, ``gridfoam.fv.fvc.grad``, and
``gridfoam.meta.GridfoamConfig`` without importing submodules first.
Physical-dimension helpers and enums are available through
``gridfoam.core.dimensions`` and ``gridfoam.meta.enums``.

These namespaces are imported after the runtime type-check hook is installed.
Existing direct submodule imports remain supported.

.. toctree::
   :maxdepth: 2

   runner
   core
   meta
   fv
   boundaries
   solvers
   algorithms
   optimize
   models
   pre_post_io

Package layout and runtime concepts are documented in
:doc:`../contributor_guide/architecture/index`.
