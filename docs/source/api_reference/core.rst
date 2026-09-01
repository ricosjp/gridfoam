.. _api-core:

Core
====

.. currentmodule:: gridfoam.core

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   CellField
   FaceField
   FvMatrix
   IGridBase
   FieldNameParts

.. autosummary::
   :toctree: generated
   :template: autosummary/function.rst

   equation
   create_grid
   make_field_name
   parse_field_name

Field helpers
-------------

.. currentmodule:: gridfoam.core.field

.. autosummary::
   :toctree: generated
   :template: autosummary/function.rst

   get_or_create_cellfield
   get_or_create_facefield
   packed_face_n_rows

Physical dimensions
-------------------

.. currentmodule:: gridfoam.core.dimensions

.. autosummary::
   :toctree: generated
   :template: autosummary/function.rst

   to_dimensions
   resolve_field_dimension
   default_field_dimension
   assert_compatible
   dim_mul
   dim_div
   dim_pow
   dimension_config

Equation container
------------------

.. currentmodule:: gridfoam.core.equation

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   Equation
