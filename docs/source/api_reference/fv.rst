.. _api-fv:

Finite-volume operators
=======================

Package namespace
-----------------

.. currentmodule:: gridfoam.fv

The ``fvc`` and ``fvm`` subpackages are re-exported from :mod:`gridfoam.fv`.

Explicit operators (``fvc``)
----------------------------

.. currentmodule:: gridfoam.fv.fvc

.. autosummary::
   :toctree: generated
   :template: autosummary/function.rst

   grad
   interpolate
   sn_grad
   div
   reconstruct

Implicit operators (``fvm``)
----------------------------

.. currentmodule:: gridfoam.fv.fvm

.. autosummary::
   :toctree: generated
   :template: autosummary/function.rst

   ddt
   div
   laplacian

Flux utilities
--------------

.. currentmodule:: gridfoam.fv.flux

.. autosummary::
   :toctree: generated
   :template: autosummary/function.rst

   correct_flux
   compute_phi_hbya
   flux_from_face_velocity

Face geometry
-------------

Static face geometry is cached on ``grid.fv_cache`` and rebuilt after
``remesh``, ``update_ib``, or ``to``.

.. currentmodule:: gridfoam.fv.kernels

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   FaceGeometry

.. autosummary::
   :toctree: generated
   :template: autosummary/function.rst

   face_geometry

GCIBM interpolation foothold
----------------------------

``InterpolationStencil`` is a donor-weighted gather and its exact transpose.
It is not wired to matrix assembly or flow solvers.

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   InterpolationStencil
