.. _contributor-extending:

Extending gridfoam
==================

Pick the page that matches the change you are making. Each page is a checklist
with implementation and testing details.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Task
     - Page
   * - Add or change a YAML option
     - :doc:`configuration`
   * - Add a boundary condition
     - :doc:`adding_boundary_condition`
   * - Add an FV operator or scheme
     - :doc:`adding_fv_scheme`
   * - Add a transport or turbulence model
     - :doc:`adding_model`
   * - Add a linear solver
     - :doc:`adding_linear_solver`
   * - Add a pressure--velocity algorithm
     - :doc:`adding_algorithm`

For package ownership and dependency rules, see
:doc:`../architecture/guidelines`.

.. toctree::
   :maxdepth: 1
   :hidden:

   configuration
   adding_boundary_condition
   adding_fv_scheme
   adding_model
   adding_linear_solver
   adding_algorithm
