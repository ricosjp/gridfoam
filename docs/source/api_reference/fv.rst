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
   set_phi_from_matrix_flux

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
