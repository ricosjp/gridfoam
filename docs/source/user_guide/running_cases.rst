.. _user-running-cases:

Running cases
=============

All bundled examples live under ``examples/`` and are started from the
repository root with ``uv run``.

Introductory cases (CPU-friendly)
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Case
     - Command
     - Notes
   * - Lid-driven cavity
     - ``uv run examples/cavity/gridfoam/run.py``
     - No immersed body; good first run.
   * - Cavity with baffle
     - ``uv run examples/cavity_with_baffle/gridfoam/run.py``
     - Includes a small STL immersed surface.
   * - Hagen–Poiseuille
     - ``uv run examples/hagen_poiseuille/gridfoam/run.py``
     - Channel flow validation; longer time integration.

External-flow cases
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Case
     - Command
     - Notes
   * - Cylinder
     - ``uv run examples/cylinder/gridfoam/run.py``
     - CUDA config; long run time. See :doc:`../example_gallery/cylinder`.
   * - Ahmed body
     - ``uv run examples/ahmed_body/gridfoam/run.py``
     - CUDA; requires writable output directory.
   * - MotorBike
     - ``uv run examples/motorBike/gridfoam/run.py``
     - CUDA; large immersed surface.

Optimization
------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Case
     - Command
     - Notes
   * - Inlet temperature
     - ``uv run examples/optimize/inlet_temperature/run.py``
     - Differentiable optimization example; see config in the same directory.

Outputs
-------

Each case writes VTU files to the ``output_dir`` declared in its YAML
configuration. Post-processing scripts such as
``examples/cylinder/plot_comparison.py`` are separate from the solver run and
are not executed during documentation builds.

Hardware extras
---------------

CUDA examples require a matching PyTorch extra during installation:

.. code-block:: console

   $ uv sync --extra cu124 --extra graphlow --group dev

Replace ``cu124`` with ``cu118`` when appropriate. Do not install CPU and CUDA
extras together.

See also
--------

* :doc:`quickstart`
* :doc:`../example_gallery/index`
